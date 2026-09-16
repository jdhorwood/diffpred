from typing import Any, Dict, List, Tuple
import pandas as pd
from lightning.pytorch.utilities.types import STEP_OUTPUT
from diffpred.constants.metrics import (
    REGRESSION_METRICS,
    CLASSIFICATION_METRICS,
    compute_metrics,
)
from diffpred.models.networks.factories.nn_factory import baseline_net_factory
from diffpred.models.protocols import ArchitectureConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


class NetTrainer(pl.LightningModule):
    def __init__(
        self,
        architecture_conf: ArchitectureConfig,
        lr=1e-2,
        classifier=False,
        weight_decay=0,
        lr_schedule_factor=0.5,
        lr_schedule_patience=10,
        num_pred_samples: int = 50,
        use_scheduler: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.network = self.create_network(architecture_conf)
        self.lr = lr
        self.lr_schedule_factor = lr_schedule_factor
        self.lr_schedule_patience = lr_schedule_patience
        self.validation_step_outputs: List[Dict] = []
        self.test_step_outputs: dict = {}
        self.num_pred_samples = num_pred_samples
        self.weight_decay = weight_decay
        self.classifier = classifier
        self._loss_fn = self.create_loss_fn()
        self.use_scheduler = use_scheduler
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

    def create_network(self, architecture_conf: ArchitectureConfig):
        return baseline_net_factory(conf=architecture_conf)

    def create_loss_fn(self):
        if hasattr(self.network, "loss_fn"):
            return self.network.loss_fn
        if self.classifier:
            return F.binary_cross_entropy_with_logits
        else:
            return F.mse_loss

    def loss_fn(self, pred, y):
        if not torch.isnan(y).any():
            return self._loss_fn(pred, y)
        else:
            # compute masked loss
            mask = ~torch.isnan(y)
            if pred.shape != y.shape:
                if y.shape[-1] != 1:
                    raise ValueError(
                        f"Prediction shape {pred.shape} does not match target shape {y.shape}"
                    )
                else:
                    masked_pred = torch.stack(
                        [pred[mask.squeeze(), i] for i in range(pred.shape[1])], dim=1
                    )
                    masked_y = y[mask.squeeze()]
            else:
                masked_pred = pred[mask]
                masked_y = y[mask]
            return self._loss_fn(masked_pred, masked_y)

    def forward(self, input) -> Any:
        return self.network(input)

    def training_step(
        self,
        batch: Tuple[torch.FloatTensor, torch.FloatTensor],
        batch_idx: torch.LongTensor,
    ) -> STEP_OUTPUT:
        x, y = batch
        pred = self.network(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(
        self,
        batch: Tuple[torch.FloatTensor, torch.FloatTensor],
        batch_idx: torch.LongTensor,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        x, y = batch
        out = self.network(x)
        test_loss = self.loss_fn(out, y.float())

        res = self.predict_step(batch)
        self.log_dict({"test_loss": test_loss}, prog_bar=True)
        if dataloader_idx not in self.test_step_outputs:
            self.test_step_outputs[dataloader_idx] = []

        self.test_step_outputs[dataloader_idx].append(
            {
                "y_pred": res["y_pred"],
                "y": res["y"],
                "y_proba": res.get("y_proba", None),
                "y_pred_logits": res.get("y_pred_logits", None),
            }
        )

        return res

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

        self.log_dict(res, prog_bar=True)
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
            self.log_dict(res)
        if hasattr(self.datamodule, "use_benchmark") and self.datamodule.use_benchmark:
            benchmark_res = self.datamodule.benchmark_eval(y_preds)
            print(f"Benchmark evaluation: {benchmark_res}")
            pd.DataFrame(benchmark_res).to_csv(
                f"benchmark_results_{self.datamodule.dataset.replace('/', '_')}.csv"
            )

    def predict_step(
        self,
        batch: Tuple[torch.FloatTensor, torch.FloatTensor],
        batch_idx: torch.LongTensor | None = None,
    ) -> STEP_OUTPUT:
        x, y = batch
        y_pred, y_proba = self.predict(x, return_proba=True)
        y = self.datamodule.inverse_target_transform(y)
        res = {"y_pred": y_pred, "y": y, "y_proba": y_proba, "y_pred_logits": None}
        return res

    @torch.no_grad()
    def predict(
        self, x: torch.FloatTensor, return_proba: bool = False
    ) -> torch.FloatTensor:
        pred = self.network(x)
        pred = self.datamodule.inverse_target_transform(pred)
        pred, proba = self.process_pred(pred)
        if return_proba:
            return pred, proba
        return pred

    def compute_pred_with_uncertainty(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if not self.has_uncertainty:
            raise ValueError("Model does not have uncertainty")
        pred, lower, upper = self.network.posterior_prediction(
            x, num_samples=self.num_pred_samples, return_std=True
        )
        pred, lower, upper = (
            self.datamodule.inverse_target_transform(pred),
            self.datamodule.inverse_target_transform(lower),
            self.datamodule.inverse_target_transform(upper),
        )

        return pred, lower, upper

    @property
    def has_uncertainty(self):
        return False

    @torch.no_grad()
    def validation_step(
        self,
        batch: Tuple[torch.FloatTensor, torch.FloatTensor],
        batch_idx: torch.LongTensor,
    ) -> STEP_OUTPUT:
        x, y = batch
        out = self.network(x)
        val_loss = self.loss_fn(out, y.float())
        self.log_dict({f"{self.metric_prefix}loss": val_loss}, prog_bar=True)
        res = self.predict_step(batch)
        self.validation_step_outputs.append(
            {
                "y_pred": res["y_pred"],
                "y": res["y"],
                "y_proba": res.get("y_proba", None),
                "y_pred_logits": res.get("y_pred_logits", None),
            }
        )

        return res

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_schedule_factor,
                patience=self.lr_schedule_patience,
                min_lr=1e-4,
            )
            scheduler_config = {
                "optimizer": optimizer,
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": self.trainer.check_val_every_n_epoch,
            }
            return [optimizer], [scheduler_config]
        else:
            return optimizer

    def process_pred(
        self, y_pred: torch.FloatTensor, clf_threshold: float = 0.5
    ) -> torch.FloatTensor | Tuple[torch.FloatTensor, torch.FloatTensor]:
        if self.classifier:
            y_proba = torch.sigmoid(y_pred)
            return torch.where(y_proba > clf_threshold, 1, 0), y_proba
        return y_pred, None

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule


class MVETrainer(NetTrainer):
    def __init__(
        self,
        architecture_conf: ArchitectureConfig,
        lr=1e-3,
        classifier=False,
        weight_decay=0,
        lr_schedule_factor=0.5,
        lr_schedule_patience=20,
        num_pred_samples: int = 50,
        use_scheduler: bool = True,
        warmup_ratio: float = 0,
        warmup_variance: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            architecture_conf=architecture_conf,
            lr=lr,
            classifier=classifier,
            weight_decay=weight_decay,
            lr_schedule_factor=lr_schedule_factor,
            lr_schedule_patience=lr_schedule_patience,
            num_pred_samples=num_pred_samples,
            use_scheduler=use_scheduler,
            **kwargs,
        )

        self.warmup_ratio = warmup_ratio
        self.warmup_variance = warmup_variance
        if self.classifier:
            raise ValueError("MVETrainer does not support classification tasks")

    def create_network(self, architecture_conf):
        # create the MVE network architeccture
        return baseline_net_factory(conf=architecture_conf, use_mve=True)

    def create_loss_fn(self):
        return self.mve_loss_fn

    def mve_loss_fn(
        self, pred: torch.FloatTensor, y: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Implement the MVE loss function here
        assert pred.ndim == 2 and len(pred) == len(y)

        # First column is the mean, second column is the variance
        mean = pred[:, 0].unsqueeze(-1)
        log_std = pred[:, 1].unsqueeze(-1)
        log_std = torch.clamp(log_std, min=-5, max=5)  # Clamp for numerical stability
        variance = torch.exp(2 * log_std)
        eps = 1e-7  # Small constant to avoid log(0) and division by zero
        loss = 0.5 * torch.mean(
            ((y - mean) ** 2 / (variance + eps)) + torch.log(variance + eps)
        )
        return loss

    def get_warmup_weight(self) -> float:
        """
        Returns a weight for blending fixed and predicted variance:
        - 1.0 during full warmup
        - Gradually decreases to 0.0 during transition period
        - 0.0 after transition is complete
        """
        if self.max_epochs is None:
            raise ValueError(
                "Trainer is not initialized yet but tried to establish warmup"
            )

        warmup_epochs = int(self.max_epochs * self.warmup_ratio)
        transition_epochs = int(
            warmup_epochs * 0.5
        )  # Transition over 50% of warmup time

        if self.current_epoch < warmup_epochs:
            # Full warmup
            return 1.0
        elif self.current_epoch < warmup_epochs + transition_epochs:
            # Transition period - linear decay
            progress = (self.current_epoch - warmup_epochs) / transition_epochs
            return 1.0 - progress
        else:
            # After transition
            return 0.0

    @property
    def has_uncertainty(self):
        return True

    @torch.no_grad()
    def predict(
        self, x: torch.FloatTensor, return_proba: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        out = self.network(x)
        pred = out[:, 0].unsqueeze(-1)
        std = torch.exp(out[:, 1].unsqueeze(-1))
        pred = self.datamodule.inverse_target_transform(pred)
        std = (
            std * self.datamodule.data_std
        )  # Scale std by data std to match original scale
        if return_proba:
            return pred, std
        return pred

    @torch.no_grad()
    def compute_pred_with_uncertainty(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, std = self.predict(x, return_proba=True)
        return pred, std.abs()

    @property
    def max_epochs(self):
        return self.trainer.max_epochs if self.trainer else None

    @property
    def in_warmup(self) -> bool:
        if self.max_epochs is None:
            raise ValueError(
                "Trainer is not initialized yet but tried to establish warmup"
            )
        return self.current_epoch < int(self.max_epochs * self.warmup_ratio)


class MCDropoutTrainer(NetTrainer):
    def __init__(
        self,
        architecture_conf: ArchitectureConfig,
        lr=1e-3,
        classifier=False,
        weight_decay=0,
        lr_schedule_factor=0.5,
        lr_schedule_patience=10,
        num_pred_samples: int = 1000,
        use_scheduler: bool = True,
        **kwargs,
    ):
        if getattr(architecture_conf, "mc_dropout", 0) == 0:
            # Legacy checkpoints/configs may omit mc_dropout; use historical default.
            architecture_conf.mc_dropout = 0.2
            print(
                "Warning: mc_dropout was missing/zero for MCDropoutTrainer. "
                "Applying legacy default mc_dropout=0.2."
            )

        super().__init__(
            architecture_conf=architecture_conf,
            lr=lr,
            classifier=classifier,
            weight_decay=weight_decay,
            lr_schedule_factor=lr_schedule_factor,
            lr_schedule_patience=lr_schedule_patience,
            num_pred_samples=num_pred_samples,
            use_scheduler=use_scheduler,
            **kwargs,
        )

        if self.classifier:
            raise ValueError("MCDropoutTrainer is currently for regression only")

    def create_network(self, architecture_conf):
        return baseline_net_factory(conf=architecture_conf)

    @property
    def has_uncertainty(self):
        return True

    def _enable_dropout(self):
        """Enable dropout at test time for MC Dropout."""
        for m in self.network.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def predict(
        self,
        x: torch.FloatTensor,
        return_samples: bool = False,
        return_proba: bool = False,
    ) -> torch.FloatTensor | Tuple[torch.FloatTensor, torch.FloatTensor]:
        self.network.eval()
        self._enable_dropout()

        preds: list[torch.Tensor] = []
        for _ in range(self.num_pred_samples):
            out = self.network(x)
            preds.append(out)

        preds_tensor = torch.stack(preds, dim=0)  # [num_samples, batch, output_dim]
        preds_tensor = self.datamodule.inverse_target_transform(
            preds_tensor
        )  # rescale to original scale

        if return_samples:
            return preds_tensor  # [num_samples, batch, output_dim]

        mean = preds_tensor.mean(dim=0)
        std = preds_tensor.std(dim=0)
        if return_proba:
            return mean, std
        return mean

    @torch.no_grad()
    def compute_pred_with_uncertainty(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, std = self.predict(x, return_proba=True)
        return pred, std.abs()


class DeepEnsembleTrainer(NetTrainer):
    def __init__(
        self,
        architecture_conf: ArchitectureConfig,
        lr=1e-3,
        classifier=False,
        weight_decay=0,
        lr_schedule_factor=0.5,
        lr_schedule_patience=10,
        ensemble_size: int = 5,
        use_scheduler: bool = True,
        **kwargs,
    ):
        super().__init__(
            architecture_conf=architecture_conf,
            lr=lr,
            classifier=classifier,
            weight_decay=weight_decay,
            lr_schedule_factor=lr_schedule_factor,
            lr_schedule_patience=lr_schedule_patience,
            use_scheduler=use_scheduler,
            **kwargs,
        )
        self.automatic_optimization = False  # Required for manual optimization
        self.ensemble_size = ensemble_size
        self.ensemble = nn.ModuleList(
            [
                MVETrainer(
                    architecture_conf=architecture_conf,
                    lr=lr,
                    classifier=classifier,
                    weight_decay=weight_decay,
                    lr_schedule_factor=lr_schedule_factor,
                    lr_schedule_patience=lr_schedule_patience,
                    use_scheduler=use_scheduler,
                    **kwargs,
                )
                for _ in range(ensemble_size)
            ]
        )
        if classifier:
            raise ValueError("This class supports regression only")

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule
        for model in self.ensemble:
            model.set_datamodule(datamodule)

    def forward(self, x):
        preds, stds = [], []
        for model in self.ensemble:
            model.eval()
            pred, std = model.predict(x, return_proba=True)
            preds.append(pred)
            stds.append(std)
        preds = torch.stack(preds, dim=0)  # [E, B, 1]
        stds = torch.stack(stds, dim=0)  # [E, B, 1]
        mean_ensemble = preds.mean(dim=0)
        var_aleatoric = stds.pow(2).mean(dim=0)
        var_epistemic = preds.var(dim=0, unbiased=False)
        total_variance = var_aleatoric + var_epistemic
        std_total = total_variance.sqrt()
        return mean_ensemble, std_total

    @torch.no_grad()
    def predict(self, x: torch.FloatTensor, return_proba: bool = False):
        mean, std = self.forward(x)
        if return_proba:
            return mean, std
        return mean

    @torch.no_grad()
    def compute_pred_with_uncertainty(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, std = self.predict(x, return_proba=True)
        return pred, std.abs()

    def training_step(
        self,
        batch: Tuple[torch.FloatTensor, torch.FloatTensor],
        batch_idx: torch.LongTensor,
    ):
        # Access all optimizers for ensemble members
        optimizers = self.optimizers()
        losses = []
        for i, model in enumerate(self.ensemble):
            model = model.to(self.device)
            optimizer = optimizers[i]
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            if isinstance(loss, dict):
                loss = loss["loss"] if "loss" in loss else list(loss.values())[0]
            # Backward and step manually
            self.manual_backward(loss)
            optimizer.step()
            losses.append(loss.detach())
        loss = torch.stack(
            [
                torch.tensor(member_loss)
                if not torch.is_tensor(member_loss)
                else member_loss
                for member_loss in losses
            ]
        ).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.FloatTensor, torch.FloatTensor],
        batch_idx: torch.LongTensor,
    ):
        x, y = batch
        mean, std = self.forward(x)
        val_loss = F.mse_loss(mean, self.datamodule.inverse_target_transform(y))
        self.log_dict({"val/loss": val_loss}, prog_bar=True)
        output = {
            "y_pred": mean,
            "y": self.datamodule.inverse_target_transform(y),
            "y_proba": std,
            "y_pred_logits": None,
        }
        self.validation_step_outputs.append(output)
        return output

    def test_step(
        self,
        batch: Tuple[torch.FloatTensor, torch.FloatTensor],
        batch_idx: torch.LongTensor,
        dataloader_idx: int = 0,
    ):
        x, y = batch
        mean, std = self.forward(x)
        test_loss = F.mse_loss(mean, self.datamodule.inverse_target_transform(y))
        self.log_dict({"test_loss": test_loss}, prog_bar=True)
        output = {
            "y_pred": mean,
            "y": self.datamodule.inverse_target_transform(y),
            "y_proba": std,
            "y_pred_logits": None,
        }
        # Ensure test_step_outputs is a dict of lists, keyed by dataloader_idx
        if not hasattr(self, "test_step_outputs"):
            self.test_step_outputs = {}
        if dataloader_idx not in self.test_step_outputs:
            self.test_step_outputs[dataloader_idx] = []
        self.test_step_outputs[dataloader_idx].append(output)
        return output

    def configure_optimizers(self):
        # Return optimizers for all ensemble members
        optimizers = []
        for model in self.ensemble:
            opt = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            optimizers.append(opt)
        return optimizers
