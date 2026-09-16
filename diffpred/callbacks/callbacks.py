from typing import Literal, Dict, Tuple
import os
import joblib
import fsspec
import torch
import numpy as np
from torch.utils.data import DataLoader
from diffpred.viz.viz_edm import viz_sigma_losses
from diffpred.data.mpp_datamodule import MPPDatamodule
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from diffpred.uq.predictive_dist import (
    calibration_plots,
)
from diffpred.trainers import DiffusionTrainer
from diffpred.utils.checkpoint_reload import CheckpointLoader, CheckpointResolver


def _reload_model(trainer, pl_module, criteria: Literal["best", "last"]):
    dm = pl_module.datamodule
    checkpoint_callback = trainer.checkpoint_callback
    reload_path = CheckpointResolver.from_checkpoint_callback(
        checkpoint_callback=checkpoint_callback,
        criteria=criteria,
        prefer_ema=isinstance(pl_module, DiffusionTrainer),
    )
    if not reload_path.is_dir():
        pl_module = CheckpointLoader.load_model(
            model=pl_module,
            checkpoint_path=reload_path,
            datamodule=dm,
            ensure_compatibility=True,
        )
    elif hasattr(pl_module, "set_datamodule"):
        pl_module.set_datamodule(dm)
    return pl_module

class DatamoduleCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        # save initial model
        pl_module.set_datamodule(trainer.datamodule)

    def on_validation_start(self, trainer, pl_module):
        # save model before validation
        pl_module.set_datamodule(trainer.datamodule)

    def on_test_start(self, trainer, pl_module):
        # save model after validation
        pl_module.set_datamodule(trainer.datamodule)

class CalibrationPlotCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        pl_module = _reload_model(trainer, pl_module, "best")
        output_path = Path.cwd().joinpath("viz")
        os.makedirs(output_path, exist_ok=True)
        calibration_plots(
            model=pl_module, datamodule=pl_module.datamodule, output_dir=output_path
        )

class SigmaLossesCallback(Callback):
    num_samples: int = 5000

    def on_train_start(self, trainer, pl_module):
        with torch.no_grad():
            self.initial_result_train = self._compute_sigma_losses(
                pl_module.diffuser, trainer.datamodule, pl_module, train=True
            )
            self.initial_result_val = self._compute_sigma_losses(
                pl_module.diffuser, trainer.datamodule, pl_module, train=False
            )
            with fsspec.open("sigma_losses_initial_train.pkl", "wb") as f:
                joblib.dump(self.initial_result_train, f)

    def on_train_end(self, trainer, pl_module):
        pl_module = _reload_model(trainer, pl_module, "last")
        with torch.no_grad():
            self.after_result_train = self._compute_sigma_losses(
                pl_module.diffuser, trainer.datamodule, pl_module
            )
            self.after_result_val = self._compute_sigma_losses(
                pl_module.diffuser, trainer.datamodule, pl_module, train=False
            )
            with fsspec.open("sigma_losses_after_train.pkl", "wb") as f:
                joblib.dump(self.after_result_train, f)

            with fsspec.open("sigma_losses_after_val.pkl", "wb") as f:
                joblib.dump(self.after_result_val, f)

        sampled_sigmas = pl_module.diffuser.sample_sigma(shape=10000)
        # visualize the losses and sigma sampling density
        viz_sigma_losses(
            initial_losses=self.initial_result_train,
            post_losses=self.after_result_train,
            sampled_sigmas=sampled_sigmas,
            output_dir=Path.cwd(),
            split="train",
        )
        viz_sigma_losses(
            initial_losses=self.initial_result_val,
            post_losses=self.after_result_val,
            sampled_sigmas=sampled_sigmas,
            output_dir=Path.cwd(),
            split="val",
        )

    def _compute_sigma_losses(self, diffuser, dm, model, train: bool = True):
        sigmas = np.linspace(
            np.log(diffuser.sigma_min), np.log(diffuser.sigma_max), 200
        )
        sigmas = np.exp(sigmas)
        sigmas = np.clip(sigmas, diffuser.sigma_min, diffuser.sigma_max)
        result: Dict[float, Tuple[float, float, float]] = {}
        dataloader = self._sample_data(dm, train=train)
        for noise_level in sigmas:
            # compute model loss
            losses = self._get_losses(dataloader, model, noise_level)
            # store mean, std
            weight = model.diffuser.get_loss_weights(noise_level)
            if torch.is_tensor(weight):
                weight = weight.cpu()
            mean = losses.mean().item()
            std = losses.std().item()
            result[noise_level] = (mean, std, weight)
        return result

    def _sample_data(self, dm: MPPDatamodule, train: bool = True) -> DataLoader:
        if train:
            dt = dm.train_dt
        else:
            dt = dm.val_dt
        indices = np.random.randint(0, len(dt), size=self.num_samples)
        if isinstance(dt, (list, torch.utils.data.dataset.Subset)):
            x, y = zip(*[dt[i] for i in indices])
        else:
            x, y = dt[indices]
        dataset = dm._convert_to_dataset(x, y)
        dataloader_cls = dm.get_dataloader_cls()
        return dataloader_cls(dataset, batch_size=dm.batch_size, shuffle=True)

    def _get_losses(self, dataloader, model, sigma: float):
        losses = []
        for ix, batch in enumerate(dataloader):
            batch[0] = batch[0].to(model.device)
            batch[1] = batch[1].to(model.device)
            y, y_k, _, out, loss, _ = model._batch_step(batch, sigma=sigma)
            losses.append(loss)
        return torch.cat(losses)