import os
import yaml
import joblib
import logging
import torch
from pathlib import Path
from dataclasses import asdict
import hydra
import optuna
from omegaconf import OmegaConf, DictConfig
import fsspec
from diffpred.utils.configuration import CallbackConfigurer
from diffpred.utils.ops import configure_runtime_precision, seed_everything
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from diffpred.utils.reloader import get_datamodule
from diffpred.utils.reloader import get_trainer
from diffpred.utils.checkpoint_reload import (
    ConfigAlignment,
    CheckpointLoader,
    CheckpointResolver,
)


def _persist_target_columns(datamodule) -> None:
    target_columns = getattr(datamodule, "target_columns", None)
    if not isinstance(target_columns, list) or not target_columns:
        return

    path = Path.cwd() / ".hydra" / "target_columns.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump({"target_columns": target_columns}, f, sort_keys=False)


def run_experiment(
    exp_params, model_params, logger, model, datamodule, baseline, trial=None
):
    """
    Execute fit and train for the PL model with the appropriate experiment datamodule. This method should be agnostic to the specific experiment.
    """
    if isinstance(logger, WandbLogger) and isinstance(model, pl.LightningModule):
        logger.watch(model)

    callback_config = CallbackConfigurer(exp_params, model, datamodule, baseline, trial)

    max_epochs = model_params.training.get("max_epochs", 10)
    console_out = False if (exp_params.tune or not exp_params.console_out) else True
    trainer = pl.Trainer(
        max_epochs=(10 if exp_params.get("debug", False) else max_epochs),
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=(exp_params.get("check_val_freq", 10)),
        callbacks=callback_config.all_callbacks,
        accelerator=(exp_params.device if exp_params.device is not None else "auto"),
        limit_train_batches=(1 if exp_params.get("debug", False) else None),
        enable_model_summary=console_out,
        enable_progress_bar=console_out,
        limit_val_batches=(
            1
            if exp_params.get("debug", False)
            else model_params.training.get("val_limit", None)
        ),
    )

    trainer.fit(model=model, datamodule=datamodule)
    if exp_params.use_ema_on_test:
        reload_path = CheckpointResolver.from_checkpoint_callback(
            checkpoint_callback=callback_config.checkpoint_callback,
            criteria="best",
            prefer_ema=True,
        )
    else:
        reload_path = CheckpointResolver.from_checkpoint_callback(
            checkpoint_callback=callback_config.checkpoint_callback,
            criteria="best",
            prefer_ema=False,
        )
    if not reload_path.is_dir() and not exp_params.use_last_ckpt:
        model = CheckpointLoader.load_model(
            model=model,
            checkpoint_path=reload_path,
            datamodule=datamodule,
            ensure_compatibility=True,
        )

    ConfigAlignment.set_eval_split(model, "train")
    train_res = trainer.validate(
        model=model, dataloaders=datamodule.train_dataloader(), verbose=False
    )
    ConfigAlignment.set_eval_split(model, "val")
    val_res = trainer.validate(model=model, datamodule=datamodule)
    test_res = trainer.test(model=model, datamodule=datamodule)

    # save results
    results = {"train": train_res, "val": val_res, "test": test_res}
    with fsspec.open(f"{exp_params.output_dir}/results.yaml", "w") as f:
        yaml.dump(results, f)

    # save the wandb run metadata if using wandb
    if isinstance(logger, WandbLogger):
        # Extract the important wandb metadata
        wandb_metadata = {
            "run_id": logger.experiment.id,
            "project": logger.experiment.project,
            "entity": logger.experiment.entity,
        }

        # Save the metadata to a custom YAML file
        with fsspec.open(f"{exp_params.output_dir}/wandb_run_metadata.yaml", "w") as f:
            yaml.dump(wandb_metadata, f)

        # ensure that the wandb run is finished properly
        logger.finalize(status="success")
        logger.experiment.finish()

    return results


class MPPExperiment:
    def __init__(self, config: DictConfig):
        self.full_cfg = config
        self.exp_params = config.experiment
        self.model_params = config.model
        self.data_params = config.data

        # check diffusion type to determine if baseline
        self.is_baseline = ConfigAlignment.infer_baseline_and_align_config(config)

        # set up wandb logger
        self._setup_wandb_logger()

        # Keep runtime precision consistent with eval/reload codepaths.
        configure_runtime_precision("medium")

    def _setup_wandb_logger(self):
        wandb_mode = self._get_wandb_mode()
        if wandb_mode == "disabled":
            logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

        self.logger = WandbLogger(
            project=self.exp_params.get("wandb_project", "diffpred"),
            name=self.exp_params.exp_name,
            save_dir=self.exp_params.output_dir,
            config=OmegaConf.to_container(
                self.full_cfg, resolve=True, throw_on_missing=True
            ),
            mode=wandb_mode,
        )

    def _get_wandb_mode(self):
        if self.exp_params.get("debug", False):
            return "disabled"
        elif self.exp_params.get("tune", False):
            return "disabled"
        elif self.exp_params.get("wandb_mode", None) is not None:
            return self.exp_params.wandb_mode
        else:
            return None

    def __call__(self, trial=None):

        ### set the seed for the run for reproducibility
        seed_everything(self.exp_params.seed)

        ### setup datamodule
        self.datamodule = get_datamodule(
            batch_size=self.model_params.training.batch_size,
            **self.data_params,
            featurizer=self.model_params.featurizer,
            seed=self.exp_params.seed,
            net_type=self.model_params.hparams.get("net_type", None),
        )
        self.datamodule.setup()
        _persist_target_columns(self.datamodule)

        ### setup model
        self.model = get_trainer(
            self.model_params,
            self.data_params,
            self.exp_params,
            self.datamodule,
            is_baseline=self.is_baseline,
        )

        ### run experiment
        result = run_experiment(
            exp_params=self.exp_params,
            model_params=self.model_params,
            logger=self.logger,
            model=self.model,
            datamodule=self.datamodule,
            baseline=self.is_baseline,
            trial=trial,
        )
        # save the results for the trial in user_attrs
        if trial is not None:
            for k, v in result.items():
                trial.set_user_attr(k, v)

        metric = result["val"][0][self.exp_params.monitor_metric]
        return metric

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.data.name in [
        "tdc",
        "molecule_ace",
        "polaris",
        "dockstring",
        "dataframe_masked_imputation",
    ]:
        experiment = MPPExperiment(cfg)
        experiment()
    else:
        raise ValueError(f"Experiment for {cfg.data.name} not supported")


if __name__ == "__main__":
    main()
