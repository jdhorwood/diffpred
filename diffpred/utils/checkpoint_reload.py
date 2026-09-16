from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from omegaconf import OmegaConf, open_dict

from diffpred.utils.reloader import get_datamodule, get_trainer, load_config
from diffpred.utils.ops import configure_runtime_precision


class ConfigAlignment:
    @staticmethod
    def _run_has_legacy_module_paths(run_dir: Path | None) -> bool:
        if run_dir is None:
            return False

        marker = b"mf_prediction"
        for ckpt_path in run_dir.rglob("*.ckpt"):
            if "/.hydra/" in str(ckpt_path):
                continue
            try:
                with ckpt_path.open("rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        if marker in chunk:
                            return True
            except OSError:
                continue
        return False

    @staticmethod
    def apply_legacy_model_defaults(cfg, run_dir: Path | None = None) -> None:
        if not ConfigAlignment._run_has_legacy_module_paths(run_dir):
            return

        model_params = cfg.model
        hparams = model_params.hparams
        model_name = model_params.get("name", None)

        uses_uncertainty_baseline = any(
            model_params.get(flag, False)
            for flag in ("use_mcdropout", "use_mve", "use_deepensemble", "use_deep_ensemble")
        )
        if not uses_uncertainty_baseline:
            return

        has_mc_dropout = False
        try:
            has_mc_dropout = "mc_dropout" in hparams
        except Exception:
            has_mc_dropout = hasattr(hparams, "mc_dropout")

        mc_dropout_value = float(getattr(hparams, "mc_dropout", 0.0) or 0.0)
        dropout_value = float(getattr(hparams, "dropout", 0.0) or 0.0)

        def _set_mc_and_dropout(mc_value: float, dropout_new: float | None = None) -> None:
            if OmegaConf.is_config(hparams):
                with open_dict(hparams):
                    hparams.mc_dropout = mc_value
                    if dropout_new is not None:
                        hparams.dropout = dropout_new
            else:
                hparams.mc_dropout = mc_value
                if dropout_new is not None:
                    hparams.dropout = dropout_new


        # To preserve layer indexing for old checkpoints,
        # migrate dropout -> mc_dropout when needed.
        if model_name == "mlp" and dropout_value > 0.0 and mc_dropout_value == 0.0:
            _set_mc_and_dropout(mc_value=dropout_value, dropout_new=0.0)
            print(
                "Applied legacy MLP uncertainty-dropout migration for checkpoint reload: "
                f"mc_dropout={dropout_value}, dropout=0.0"
            )
            return

        # Legacy MC Dropout configs could omit mc_dropout entirely.
        if model_params.get("use_mcdropout", False) and not has_mc_dropout:
            _set_mc_and_dropout(mc_value=0.2, dropout_new=0.0)
            print("Applied legacy mc_dropout default (0.2) for MC Dropout checkpoint reload.")

    @staticmethod
    def set_eval_split(model, split_name: str) -> None:
        if hasattr(model, "set_eval_split"):
            model.set_eval_split(split_name)

    @staticmethod
    def infer_baseline_and_align_config(cfg, run_dir: Path | None = None) -> bool:
        ConfigAlignment.apply_legacy_model_defaults(cfg, run_dir=run_dir)
        data_params = cfg.data
        model_params = cfg.model
        exp_params = cfg.experiment

        if data_params.diffusion_params.diffusion_type is not None:
            is_baseline = False
        else:
            arch_type = model_params.hparams.get("arch_type", None)
            if arch_type is not None and arch_type in ("baseline", "mp-net"):
                is_baseline = False
                data_params.diffusion_params.diffusion_type = "edm"
            else:
                is_baseline = True
                exp_params.use_ema_on_test = False

        data_params.diffusion_params.refinement_steps = (
            0 if is_baseline else data_params.diffusion_params.refinement_steps
        )

        data_params.standardize_targets = True
        if data_params.get("classification", False):
            if is_baseline:
                data_params.standardize_targets = False
                data_params.symmetric_binary_targets = False
            else:
                data_params.symmetric_binary_targets = True

        return is_baseline


class CheckpointResolver:
    @staticmethod
    def resolve_checkpoint(
        run_dir: Path,
        ckpt_path: str | None,
        use_ema: bool,
    ) -> Path:
        if ckpt_path is not None:
            path = Path(ckpt_path)
            if not path.is_absolute():
                path = run_dir / path
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return path

        ckpts = [
            path
            for path in run_dir.rglob("*.ckpt")
            if "/.hydra/" not in str(path)
        ]
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found under {run_dir}")

        ema_ckpts = [path for path in ckpts if path.name.endswith("-EMA.ckpt")]
        non_ema_ckpts = [path for path in ckpts if not path.name.endswith("-EMA.ckpt")]

        if use_ema and ema_ckpts:
            candidates = ema_ckpts
        elif non_ema_ckpts:
            candidates = non_ema_ckpts
        else:
            candidates = ema_ckpts

        candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
        if len(candidates) > 1:
            print(
                f"WARNING: Multiple checkpoint candidates found ({len(candidates)}). "
                f"Using most recent: {candidates[0]}"
            )
        return candidates[0]

    @staticmethod
    def from_checkpoint_callback(
        checkpoint_callback,
        criteria: Literal["best", "last"] = "best",
        prefer_ema: bool = False,
    ) -> Path:
        reload_path = (
            checkpoint_callback.best_model_path
            if criteria == "best"
            else checkpoint_callback.last_model_path
        )
        path = Path(reload_path)
        if prefer_ema and hasattr(checkpoint_callback, "_ema_format_filepath"):
            path = Path(checkpoint_callback._ema_format_filepath(str(path)))
        return path


class CheckpointLoader:
    LEGACY_MODULE_ALIASES = {"mf_prediction": "diffpred"}

    class _ModuleAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def __init__(self, aliases: dict[str, str]):
            self.aliases = aliases

        def _map_name(self, fullname: str) -> str | None:
            for source, target in self.aliases.items():
                if fullname == source:
                    return target
                prefix = f"{source}."
                if fullname.startswith(prefix):
                    return f"{target}.{fullname[len(prefix):]}"
            return None

        def find_spec(self, fullname, path, target=None):
            mapped_name = self._map_name(fullname)
            if mapped_name is None:
                return None

            mapped_spec = importlib.util.find_spec(mapped_name)
            if mapped_spec is None:
                return None

            is_package = mapped_spec.submodule_search_locations is not None
            return importlib.util.spec_from_loader(fullname, self, is_package=is_package)

        def create_module(self, spec):
            return None

        def exec_module(self, module) -> None:
            mapped_name = self._map_name(module.__name__)
            if mapped_name is None:
                raise ModuleNotFoundError(module.__name__)
            mapped_module = importlib.import_module(mapped_name)
            sys.modules[module.__name__] = mapped_module

    @staticmethod
    @contextmanager
    def _legacy_module_aliases():
        # Some legacy checkpoints were created under `mf_prediction.*`.
        aliases = CheckpointLoader.LEGACY_MODULE_ALIASES
        finder = CheckpointLoader._ModuleAliasFinder(aliases)
        initial_modules = set(sys.modules.keys())

        # Seed top-level aliases so package imports resolve consistently.
        for source, target in aliases.items():
            try:
                sys.modules.setdefault(source, importlib.import_module(target))
            except ModuleNotFoundError:
                continue

        sys.meta_path.insert(0, finder)
        try:
            yield
        finally:
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)
            for module_name in list(sys.modules.keys()):
                if module_name in initial_modules:
                    continue
                if any(
                    module_name == source or module_name.startswith(f"{source}.")
                    for source in aliases
                ):
                    sys.modules.pop(module_name, None)

    @staticmethod
    def load_model(
        model,
        checkpoint_path: str | Path,
        datamodule=None,
        ensure_compatibility: bool = True,
    ):
        configure_runtime_precision("medium")
        checkpoint_path = Path(checkpoint_path)
        if ensure_compatibility:
            CheckpointLoader.prepare_network_for_checkpoint(model.network, checkpoint_path)

        try:
            with CheckpointLoader._legacy_module_aliases():
                reloaded = model.__class__.load_from_checkpoint(
                    str(checkpoint_path),
                    model=model.network,
                )
        except RuntimeError:

            # Fallback path for legacy checkpoints whose saved hyperparameters
            # reconstruct a different module layout. We trust the already-built
            # model instance (from aligned config) and load only the state_dict.
            with CheckpointLoader._legacy_module_aliases():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            if not isinstance(state_dict, dict):
                raise
            model.load_state_dict(state_dict, strict=True)
            reloaded = model
            print(
                "WARNING: load_from_checkpoint failed with state_dict mismatch; "
                "used direct state_dict loading fallback for compatibility."
            )

        if datamodule is not None and hasattr(reloaded, "set_datamodule"):
            reloaded.set_datamodule(datamodule)
        return reloaded

    @staticmethod
    def prepare_network_for_checkpoint(network, checkpoint_path: Path) -> None:
        inferred_output_mlp_type = CheckpointLoader._infer_output_mlp_type(checkpoint_path)
        if inferred_output_mlp_type is None:
            return

        # Update modules recursively where output_mlp_type controls readout shape.
        CheckpointLoader._set_output_mlp_type_recursive(network, inferred_output_mlp_type)

    @staticmethod
    def _infer_output_mlp_type(checkpoint_path: Path) -> str | None:
        if not checkpoint_path.exists():
            return None

        with CheckpointLoader._legacy_module_aliases():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            return None

        keys = list(state_dict.keys())
        has_residual = any(".out_mlp.blocks." in key or ".mlp_readout.blocks." in key for key in keys)
        if has_residual:
            return "residual"

        has_linear = any(
            ".out_mlp.0.weight" in key or ".mlp_readout.0.weight" in key for key in keys
        )
        if has_linear:
            return "linear"

        return None

    @staticmethod
    def _set_output_mlp_type_recursive(module, output_mlp_type: str) -> None:
        if (
            hasattr(module, "output_mlp_type")
            and hasattr(module, "_build_output_mlp")
            and hasattr(module, "_get_output_mlp_in_dim")
        ):
            current = getattr(module, "output_mlp_type", None)
            if current != output_mlp_type:
                module.output_mlp_type = output_mlp_type
                in_dim = module._get_output_mlp_in_dim()
                module.out_mlp = module._build_output_mlp(in_dim)

        for child in module.children():
            CheckpointLoader._set_output_mlp_type_recursive(child, output_mlp_type)


class RunReconstructor:
    TARGET_COLUMNS_FILE = Path(".hydra/target_columns.yaml")

    def __init__(self, run_dir: Path, cfg=None):
        self.run_dir = Path(run_dir)
        self.cfg: Any = load_config(str(run_dir)) if cfg is None else cfg
        self.is_baseline = ConfigAlignment.infer_baseline_and_align_config(
            self.cfg,
            run_dir=self.run_dir,
        )

    def resolved_use_ema(self, use_ema: bool | None = None) -> bool:
        if use_ema is not None:
            return use_ema
        return self.cfg.experiment.get("use_ema_on_test", False)

    def build_datamodule(self):
        datamodule = get_datamodule(
            batch_size=self.cfg.model.training.batch_size,
            **self.cfg.data,
            featurizer=self.cfg.model.featurizer,
            seed=self.cfg.experiment.seed,
            net_type=self.cfg.model.hparams.get("net_type", None),
        )
        datamodule.setup()
        self._apply_saved_target_column_order(datamodule)
        return datamodule

    def _target_columns_path(self) -> Path:
        return self.run_dir / self.TARGET_COLUMNS_FILE

    def _load_saved_target_columns(self) -> list[str] | None:
        path = self._target_columns_path()
        if not path.exists():
            return None
        with path.open("r") as f:
            payload = yaml.safe_load(f)
        if not isinstance(payload, dict):
            return None
        target_columns = payload.get("target_columns")
        if not isinstance(target_columns, list):
            return None
        if not all(isinstance(col, str) for col in target_columns):
            return None
        return target_columns

    def _apply_saved_target_column_order(self, datamodule) -> None:
        saved_target_columns = self._load_saved_target_columns()
        if not saved_target_columns:
            return
        RunReconstructor.apply_target_column_order(datamodule, saved_target_columns)

    @staticmethod
    def apply_target_column_order(datamodule, target_columns: list[str]) -> bool:
        if not hasattr(datamodule, "target_columns"):
            return False

        current_target_columns = list(getattr(datamodule, "target_columns", []))
        if current_target_columns == target_columns:
            return False

        if set(current_target_columns) != set(target_columns):
            print(
                "WARNING: Saved target column set does not match datamodule target columns. "
                "Skipping target order restoration."
            )
            return False

        reorder_indices = [current_target_columns.index(col) for col in target_columns]
        for split_name in ("train_dt", "val_dt", "test_dt"):
            split_dataset = getattr(datamodule, split_name, None)
            reordered = RunReconstructor._reorder_dataset_targets(split_dataset, reorder_indices)
            if reordered is not None:
                setattr(datamodule, split_name, reordered)

        RunReconstructor._reorder_target_stats(datamodule, reorder_indices)

        datamodule.target_columns = target_columns
        print("Restored target column order from saved training metadata.")
        return True

    @staticmethod
    def _reorder_target_stats(datamodule, reorder_indices: list[int]) -> None:
        for name in ("data_mean", "data_std"):
            tensor = getattr(datamodule, name, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim != 1:
                continue
            if tensor.shape[0] != len(reorder_indices):
                continue
            setattr(datamodule, name, tensor[reorder_indices])

    @staticmethod
    def _reorder_dataset_targets(dataset, reorder_indices: list[int]):
        if dataset is None:
            return None

        if hasattr(dataset, "tensors"):
            tensors = list(dataset.tensors)
            if len(tensors) < 2:
                return dataset
            targets = tensors[1]
            if isinstance(targets, torch.Tensor) and targets.ndim >= 2:
                tensors[1] = targets[:, reorder_indices]
                dataset.tensors = tuple(tensors)
            return dataset

        if isinstance(dataset, list):
            reordered_dataset = []
            for sample in dataset:
                if not (isinstance(sample, tuple) and len(sample) >= 2):
                    reordered_dataset.append(sample)
                    continue
                features, targets = sample[0], sample[1]
                if isinstance(targets, torch.Tensor) and targets.ndim >= 1:
                    targets = targets[reorder_indices]
                if len(sample) == 2:
                    reordered_dataset.append((features, targets))
                else:
                    reordered_dataset.append((features, targets, *sample[2:]))
            return reordered_dataset

        return dataset

    def build_trainer(self, datamodule):
        model = get_trainer(
            self.cfg.model,
            self.cfg.data,
            self.cfg.experiment,
            datamodule,
            is_baseline=self.is_baseline,
        )
        model.to(self.cfg.experiment.device)
        return model

    def load_checkpointed_model(
        self,
        datamodule,
        ckpt_path: str | None = None,
        use_ema: bool | None = None,
    ):
        checkpoint = CheckpointResolver.resolve_checkpoint(
            run_dir=self.run_dir,
            ckpt_path=ckpt_path,
            use_ema=self.resolved_use_ema(use_ema),
        )
        model = self.build_trainer(datamodule)
        model = CheckpointLoader.load_model(
            model=model,
            checkpoint_path=checkpoint,
            datamodule=datamodule,
            ensure_compatibility=True,
        )
        return model, checkpoint

    @staticmethod
    def apply_inference_overrides(
        model,
        *,
        disable_guidance: bool = False,
        num_refinement_steps: int | None = None,
        num_pred_samples: int | None = None,
        deterministic_sampling: bool | None = None,
    ):
        """Apply optional inference-time overrides in a model-agnostic way.

        Overrides are only applied when corresponding attributes exist.
        This keeps the function safe for both baseline and diffusion models.
        """
        if num_pred_samples is not None and hasattr(model, "num_pred_samples"):
            model.num_pred_samples = int(num_pred_samples)

        diffuser = getattr(model, "diffuser", None)
        if diffuser is None:
            return model

        if num_refinement_steps is not None and hasattr(diffuser, "refinement_steps"):
            diffuser.refinement_steps = int(num_refinement_steps)

        if disable_guidance and hasattr(diffuser, "guidance"):
            diffuser.guidance = 0.0

        if deterministic_sampling is not None and hasattr(diffuser, "deterministic_sampling"):
            diffuser.deterministic_sampling = bool(deterministic_sampling)

        return model
