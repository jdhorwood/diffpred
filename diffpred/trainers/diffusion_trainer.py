from typing import Dict, List, Tuple
import pandas as pd
from diffpred.constants.metrics import (
    REGRESSION_METRICS,
    CLASSIFICATION_METRICS,
    compute_metrics,
)
from diffpred.utils.ops import is_tensor_1d
from diffpred.models.diffusion_nn import DiffusionNN
from diffpred.models.protocols import ArchitectureConfig
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from diffpred.diffusers import EDM

DIFFUSER_MAP = {"edm": EDM}

class DiffusionTrainer(pl.LightningModule):
    def __init__(
        self,
        architecture_conf: ArchitectureConfig,
        diffusion_params,
        num_pred_samples=10,
        lr=1e-2,
        use_scheduler=True,
        n_jobs=1,
        classifier=False,
        lr_schedule_factor=0.5,
        lr_schedule_patience=10,
        loss_weights: bool = True,
        weight_decay=0,
        debug_sampling=False,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.diffuser = DIFFUSER_MAP[diffusion_params.diffusion_type](
            **diffusion_params, device=device, self_conditioning=architecture_conf.self_conditioning
        )
        self.network = DiffusionNN(
            architecture_conf=architecture_conf
        )
        self.method_name = "Predictive Diffusion"
        self.lr = lr
        self.n_jobs = n_jobs
        self.automatic_optimization = (
            False if hasattr(self.network, "gp_list") else True
        )
        self.loss_weights = loss_weights
        self.use_scheduler = use_scheduler
        self.validation_step_outputs: List[Dict] = []
        self.test_step_outputs: dict = {}
        self.lr_schedule_factor = lr_schedule_factor
        self.lr_schedule_patience = lr_schedule_patience
        self.num_pred_samples = num_pred_samples
        self.classifier = classifier
        self.weight_decay = weight_decay
        self.debug_sampling = debug_sampling
        self.self_conditioning = architecture_conf.self_conditioning
        self.self_cond_prob = diffusion_params.get("self_cond_prob", 0.0)
        self.has_uncertainty = True
        self.metric_names, self.metric_fns = (
            (CLASSIFICATION_METRICS.keys(), CLASSIFICATION_METRICS)
            if classifier
            else (REGRESSION_METRICS.keys(), REGRESSION_METRICS)
        )
        self.eval_split = "val"

    def set_eval_split(self, split_name: str):
        self.eval_split = split_name

    @property
    def metric_prefix(self) -> str:
        return f"{self.eval_split}/"

    @property
    def device(self):
        try:
            return self.trainer.strategy.root_device
        except RuntimeError:
            try:
                return next(self.network.parameters())[0].device
            except AttributeError:
                return "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def sample_pred(
        self,
        x: torch.FloatTensor,
        y_dim: int,
        num_samples: int | None = None,
        device: str | None = None,
        keep_intermediates: bool = False,
        reduce: bool = True,
        y=None,
    ) -> Tuple[
        torch.FloatTensor, Dict[str, Dict[int, torch.Tensor] | None | torch.Tensor]
    ]:
        intermediates: Dict[int, torch.Tensor] = {}
        with torch.random.fork_rng():
            # Fix the seed for sampling predictions only - force consistency across epochs and runs to isolate training variability and for checkpoint reloading. 
            torch.manual_seed(0)

            num_samples = self.num_pred_samples if num_samples is None else num_samples
            device = self.device if device is None else device

            self.network.to(device)
            z = self.diffuser.sample(
                model=self.network,
                y_dim=y_dim,
                x=x,
                num_samples_per_elem=num_samples,
                keep_intermediates=keep_intermediates,
                y=y,
            )
            if keep_intermediates:
                z, intermediates = z

            z_mean = z.mean(0) if (z.ndim > 2 and reduce) else z  # Average over samples
            z_mean = self.datamodule.inverse_target_transform(z_mean)

        if self.classifier:
            preds, probas, logits = self.binarize_preds(
                z, reduce_samples=(z.ndim > 2 and reduce)
            )
            res = {"probas": probas, "logits": logits, "intermediates": intermediates}
            return preds, res

        return z_mean, {"intermediates": intermediates, "probas": None, "logits": None}

    def loss_fn(self, pred, target, weights=None, reduce: bool = True):
        assert target.shape == pred.shape
        assert target.ndim == 2
        if weights is not None and self.loss_weights:
            assert weights.shape[0] == target.shape[0]
            assert weights.ndim <= 2
            losses = self._nan_loss_fn(pred, target, weights=weights)
        else:
            losses = self._nan_loss_fn(pred, target, weights=None)

        return losses.mean(0) if reduce else losses

    def _nan_loss_fn(self, pred, target, weights=None):
        if torch.isnan(target).any():
            mask = ~torch.isnan(target)
            weights = torch.ones_like(target) * weights if weights is not None else None
            pred, target = pred[mask], target[mask]
            weights = weights[mask] if weights is not None else None
            losses = F.mse_loss(target, pred, reduction="none")
        else:
            losses = F.mse_loss(target, pred, reduction="none").mean(-1)

        if weights is not None:
            weights = weights.squeeze() if is_tensor_1d(losses) else weights
            losses = losses * weights

        return losses

    def binarize_preds(self, y_pred, threshold=0.5, reduce_samples=True):
        y_proba = torch.sigmoid(y_pred)
        if reduce_samples:
            y_proba = self.pred_mean_from_samples(y_proba)
            y_pred = self.pred_mean_from_samples(y_pred)

        preds = torch.where(y_proba > threshold, 1, 0)

        return preds, y_proba, y_pred

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule

    def predict_samples(
        self,
        x: torch.FloatTensor,
        num_samples: int | None = None,
        device: str | None = None,
        reduce: bool = True,
        y=None,
    ) -> torch.FloatTensor:
        (
            preds,
            _,
        ) = self.sample_pred(
            x=x,
            y_dim=self.network.target_dim,
            num_samples=(self.num_pred_samples if num_samples is None else num_samples),
            device=device,
            reduce=reduce,
            y=y,
        )
        # Backward-compatible API: keep returning only predictions.
        return preds

    def compute_pred_with_uncertainty(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return mean prediction and 95% prediction interval as uncertainty from sampled predictions.
        """
        pred = self.predict_samples(x, reduce=False)
        pred_mean = self.pred_mean_from_samples(pred)
        unc = self.get_uncertainty(pred)
        return pred_mean, unc

    @staticmethod
    def pred_mean_from_samples(pred: torch.Tensor) -> torch.Tensor:
        return pred.mean(0)
    
    @staticmethod
    def get_uncertainty(pred: torch.FloatTensor, interval: float = 0.95) -> torch.Tensor:
        lower = torch.quantile(pred, (1 - interval) / 2, dim=0)
        upper = torch.quantile(pred, 1 - (1 - interval) / 2, dim=0)
        return (upper - lower).abs()

    def training_step(
        self,
        batch: Tuple[torch.FloatTensor, torch.FloatTensor],
        batch_idx: torch.LongTensor,
    ):
        y, y_k, noise_level, out, losses, delta = self._batch_step(batch)
        loss = losses.mean(0)

        with torch.no_grad():
            y_pred = self.diffuser.get_y_pred(out, y_k, noise_level)
            y_pred_inv, y_inv = (
                self.datamodule.inverse_target_transform(y_pred),
                self.datamodule.inverse_target_transform(y),
            )
            train_mse = F.mse_loss(y_pred_inv, y_inv)
            self.log("train_mse", train_mse.item(), prog_bar=True)

        self.log("train/loss", loss.item(), prog_bar=True)
        self.log_dict({f"train/{k}": v for k, v in delta.items()})
        return loss

    def _batch_step(self, batch, sigma: float | None = None):
        x, y = batch
        y_k, target, noise_level = self.diffuser.forward_process_sample(y, sigma=sigma)

        self_cond = None
        if self.self_conditioning:
            run_self_cond = (
                self.training
                and torch.rand(1, device=y.device).item() < self.self_cond_prob
            )
            if run_self_cond:
                with torch.no_grad():
                    # Generate a full-batch y_0 estimate when self-conditioning is enabled.
                    self_cond = self.diffuser.forward(
                        network=self.network,
                        x=x,
                        y_k=y_k,
                        noise_level=noise_level,
                        self_cond=None,
                    ).detach()

        if self.diffuser.guidance:
            if self.training:
                # dropout x with probabilisty probability p
                dropout_mask = torch.bernoulli(
                    torch.ones(y.shape[0], device=y.device)
                    * self.diffuser.guidance_dropout(noise_level)
                )
                x = self.diffuser._mask_features(x, dropout_mask).to(y.device)
                # mask self-conditioning information with the same dropout mask to prevent leakage of self-conditioning information in the dropped-out samples
                if self_cond is not None:
                    self_cond = self.diffuser._mask_features(
                        self_cond, dropout_mask
                    ).to(y.device)

                out = self.diffuser.forward(
                    network=self.network,
                    x=x,
                    y_k=y_k,
                    noise_level=noise_level,
                    self_cond=self_cond,
                )
            else:
                out = self.diffuser.forward(
                    network=self.network,
                    x=x,
                    y_k=y_k,
                    noise_level=noise_level,
                    self_cond=self_cond,
                )
                unconditional_out = self.diffuser.forward(
                    network=self.network,
                    x=self.diffuser._mask_features(
                        x, dropout_mask=torch.ones(y.shape[0])
                    ).to(y.device),
                    y_k=y_k,
                    noise_level=noise_level,
                    self_cond=None,
                )
                out = (
                    self.diffuser.guidance * out
                    + (1 - self.diffuser.guidance) * unconditional_out
                )  # Guidance term, equivalent to doing weighting directly on the score
        else:
            out = self.diffuser.forward(
                network=self.network,
                x=x,
                y_k=y_k,
                noise_level=noise_level,
                self_cond=self_cond,
            )
        weights = self.diffuser.get_loss_weights(noise_level)
        losses = self.loss_fn(out, target, weights=weights, reduce=False)
        delta = self._check_loss_delta(y, y_k, noise_level, out)
        return y, y_k, noise_level, out, losses, delta

    @torch.no_grad()
    def _check_loss_delta(self, y, y_k, noise_level, out):
        int_noise_idx = torch.where((noise_level > 0.01) & (noise_level < 0.1))[0]
        int_noise_loss = F.mse_loss(y[int_noise_idx], out[int_noise_idx])
        int_baseline_loss = F.mse_loss(y[int_noise_idx], y_k[int_noise_idx])
        int_delta = int_noise_loss - int_baseline_loss

        low_noise_idx = torch.where(noise_level < 0.01)[0]
        low_noise_loss = F.mse_loss(y[low_noise_idx], out[low_noise_idx])
        low_baseline_loss = F.mse_loss(y[low_noise_idx], y_k[low_noise_idx])
        low_delta = low_noise_loss - low_baseline_loss

        high_noise_idx = torch.where(noise_level > 0.1)[0]
        high_noise_loss = F.mse_loss(y[high_noise_idx], out[high_noise_idx])
        high_baseline_loss = F.mse_loss(y[high_noise_idx], y_k[high_noise_idx])
        high_delta = high_noise_loss - high_baseline_loss

        delta = {
            "delta_int": int_delta.item(),
            "delta_low": low_delta.item(),
            "delta_high": high_delta.item(),
        }
        return delta

    def _train_diffusion_step(self, x, k, eps, z):
        epsilon = eps[:, :, k]
        y_k = z[:, :, k]
        y_hat = self.network(x, y_k, k)
        loss_k = self.loss_fn(y_hat, epsilon, k)
        # losses.append(loss_k)

        optimizer = self.optimizers()[k]
        optimizer.zero_grad()
        self.manual_backward(loss_k)
        optimizer.step()
        self.log(f"loss_gp_step_{k}", loss_k.item())
        self.log(f"lengthscale_{k}", self.network.gp_list[k].lengthscale)
        self.log(f"likelihood_noise_{k}", self.network.gp_list[k].noise)
        self.log(f"amplitude_{k}", self.network.gp_list[k].kernel.outputscale)
        return (k, loss_k)

    def validation_step(
        self, batch: Tuple[torch.FloatTensor, ...], batch_idx: torch.LongTensor
    ):
        x, y = batch
        y, y_k, noise_level, out, losses, delta = self._batch_step(batch)
        val_loss = losses.mean(0)

        y_pred, pred_dict = self.sample_pred(
            x=x, y_dim=y.shape[-1], num_samples=self.num_pred_samples
        )
        y_proba, y_logits = pred_dict["probas"], pred_dict["logits"]

        val_res = {
            f"{self.metric_prefix}loss": val_loss.item(),
        }

        y = self.datamodule.inverse_target_transform(y)
        self.log_dict({f"val/{k}": v for k, v in delta.items()})
        self.log_dict(val_res, prog_bar=True)
        self.validation_step_outputs.append(
            {
                "y": y,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "y_pred_logits": y_logits,
            }
        )

    def on_validation_epoch_end(self):
        y_preds = torch.cat(
            [output["y_pred"] for output in self.validation_step_outputs]
        )
        if self.validation_step_outputs[0]["y_proba"] is not None:
            y_probas = torch.cat(
                [output["y_proba"] for output in self.validation_step_outputs]
            )
        else:
            y_probas = None

        y_trues = torch.cat([output["y"] for output in self.validation_step_outputs])

        res = compute_metrics(
            y_true=y_trues,
            y_pred=y_preds,
            y_proba=y_probas,
            metric_fns=self.metric_fns,
            split_name=self.metric_prefix,
        )

        self.log_dict(res, prog_bar=True, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        for dataloader_idx, test_step_outputs in self.test_step_outputs.items():
            y_preds = torch.cat([output["y_pred"] for output in test_step_outputs])
            if test_step_outputs[0]["y_proba"] is not None:
                y_probas = torch.cat(
                    [output["y_proba"] for output in test_step_outputs]
                )
            else:
                y_probas = None

            y_trues = torch.cat([output["y"] for output in test_step_outputs])

            res = compute_metrics(
                y_true=y_trues,
                y_pred=y_preds,
                y_proba=y_probas,
                metric_fns=self.metric_fns,
                split_name=f"test_{dataloader_idx}/",
            )
            self.test_step_outputs[dataloader_idx].clear()
            self.log_dict(res, on_step=False, on_epoch=True)
        if hasattr(self.datamodule, "use_benchmark") and self.datamodule.use_benchmark:
            benchmark_res = self.datamodule.benchmark_eval(y_preds)
            print(f"Benchmark evaluation: {benchmark_res}")
            pd.DataFrame(benchmark_res).to_csv(
                f"benchmark_results_{self.datamodule.dataset.replace('/', '_')}.csv"
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y = self.datamodule.inverse_target_transform(y)
        y_pred, pred_dict = self.sample_pred(
            x=x, y_dim=y.shape[-1], num_samples=self.num_pred_samples
        )
        y_proba, y_logits = pred_dict["probas"], pred_dict["logits"]

        if dataloader_idx not in self.test_step_outputs:
            self.test_step_outputs[dataloader_idx] = []

        self.test_step_outputs[dataloader_idx].append(
            {
                "y": y,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "y_pred_logits": y_logits,
            }
        )

    def predict_step(self, batch, batch_idx=None):
        x, y = batch
        y_inv = self.datamodule.inverse_target_transform(y)
        sampling_y = y if self.debug_sampling else None
        y_pred, pred_dict = self.sample_pred(
            x=x,
            y_dim=y.shape[-1],
            num_samples=self.num_pred_samples,
            y=sampling_y,
        )
        res = {
            "y_pred": y_pred,
            "y": y_inv,
            "y_proba": pred_dict["probas"],
            "y_pred_logits": pred_dict["logits"],
        }

        return res

    def configure_optimizers(self):
        if hasattr(self.network, "gp_list"):
            return [
                torch.optim.Adam(
                    gp.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
                for gp in self.network.gp_list
            ]
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            if self.use_scheduler:
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.lr_schedule_factor,
                    patience=self.lr_schedule_patience,
                    min_lr=1e-5,
                )
                scheduler_config = {
                    "scheduler": lr_scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": self.trainer.check_val_every_n_epoch,
                }

                return [optimizer], [scheduler_config]

            return optimizer
