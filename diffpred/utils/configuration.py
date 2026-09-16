from diffpred.callbacks.callbacks import (
    CalibrationPlotCallback,
    DatamoduleCallback,
    SigmaLossesCallback,
)
from diffpred.callbacks.ema import EMA
from diffpred.callbacks.ema_checkpoint import ModelCheckpoint
from optuna_integration import PyTorchLightningPruningCallback


class CallbackConfigurer:
    def __init__(self, exp_params, model, datamodule, baseline, trial):
        self.exp_params = exp_params
        self.model = model
        self.datamodule = datamodule
        self.baseline = baseline
        self.trial = trial
        self.all_callbacks = []
        self.checkpoint_callback = None

        self.ema_decay = self.exp_params.get("ema_decay", 0.99)
        self._configure_callbacks()

    def _configure_callbacks(self):
        monitor_metric, self.checkpoint_callback = self.create_checkpoint_callback()
        self._configure_general_callbacks(monitor_metric, self.checkpoint_callback)
        self._configure_diffusion_callbacks()

    def _configure_general_callbacks(self, monitor_metric, checkpoint_callback):
        self.all_callbacks.extend([checkpoint_callback, DatamoduleCallback()])

        if (
            "calibration_plot" in self.exp_params.callbacks
            or "all_plotting" in self.exp_params.callbacks
        ):
            self.all_callbacks.append(CalibrationPlotCallback())

        if "sigma_losses" in self.exp_params.callbacks or "all_plotting" in self.exp_params.callbacks:
            self.all_callbacks.append(SigmaLossesCallback())

        if self.trial is not None:
            self.all_callbacks.append(
                PyTorchLightningPruningCallback(self.trial, monitor=monitor_metric)
            )

    def _configure_diffusion_callbacks(self):
        if (
            not self.baseline
            and self.model.hparams.get("diffusion_params") is not None
            and self.model.hparams.diffusion_params.diffusion_type == "edm"
        ):
            self.all_callbacks.append(EMA(decay=self.ema_decay))

    def create_checkpoint_callback(self):
        monitor_metric, monitor_mode = (
            self.exp_params.monitor_metric,
            self.exp_params.monitor_mode,
        )
        monitor_metric_name = monitor_metric.replace("val/", "")
        if (
            monitor_metric_name not in (self.model.metric_names)
            and monitor_metric_name != "loss"
        ):
            raise ValueError(
                f"Monitor metric {monitor_metric_name} not in model metric names. Options are {self.model.metric_names} and loss."
            )

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            save_last=self.exp_params.save_last_ckpt,
        )

        return monitor_metric, checkpoint_callback
