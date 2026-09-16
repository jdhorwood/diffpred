from pathlib import Path
import re
import pandas as pd
import glob
import yaml
import os.path as osp
from omegaconf import OmegaConf
from diffpred.data.moleculeace_data import MoleculeAceDatamodule
from diffpred.data.tdc_data import TDCDatamodule
from diffpred.data.polaris import PolarisDatamodule
from diffpred.data.dockstring import DockstringDatamodule
from diffpred.data.masked_dataframe_datamodule import (
    MaskedDataframeImputationDatamodule,
)
from diffpred.models.protocols import ArchitectureConfig
from diffpred.trainers import (
    DiffusionTrainer,
    NetTrainer,
    MVETrainer,
    MCDropoutTrainer,
    DeepEnsembleTrainer,
)


def __get_results_path(dir_path: str) -> str | None:
    recomputed_paths = glob.glob(
        "**/recomputed_results.yaml",
        root_dir=dir_path,
        recursive=True,
        include_hidden=True,
    )
    if len(recomputed_paths) > 1:
        raise ValueError(f"Multiple recomputed results files found in {dir_path}")
    if len(recomputed_paths) == 1:
        return osp.join(dir_path, recomputed_paths[0])

    paths = glob.glob(
        "**/results.yaml", root_dir=dir_path, recursive=True, include_hidden=True
    )
    if len(paths) == 0:
        # raise a warning
        print(f"WARNING: No results file found in {dir_path}")
        return None
    if len(paths) > 1:
        raise ValueError(f"Multiple results files found in {dir_path}")

    return osp.join(dir_path, paths[0])

def get_config_path_from_dir(dir_path: str) -> str:
    paths = glob.glob(
        "**/.hydra/config.yaml", root_dir=dir_path, recursive=True, include_hidden=True
    )
    if len(paths) == 0:
        raise ValueError(f"No config file found in {dir_path}")
    if len(paths) > 1:
        raise ValueError(f"Multiple config files found in {dir_path}")

    return osp.join(dir_path, paths[0])

def load_config(dir_path: str) -> OmegaConf:
    """
    Load the config file from the given directory path.

    Args:
        dir_path (str): The directory where the config file is located.

    Returns:
        OmegaConf: The loaded config.
    """
    config_path = get_config_path_from_dir(dir_path)
    return OmegaConf.load(config_path)

def load_results(dir_path: str) -> dict | None:
    """
    Load the results yaml file from the given directory path.

    Args:
        dir_path (str): The directory where the results file is located.

    Returns:
        dict: The loaded results.
    """
    results_path = __get_results_path(dir_path)
    if results_path is None:
        return None

    with open(results_path, "r") as f:
        results = yaml.safe_load(f)

    return results

def get_trainer(model_params, data_params, exp_params, datamodule, is_baseline):
    architecture_conf = _get_arch_conf(model_params, datamodule)

    if not _is_uncertainty_baseline(model_params) and architecture_conf.mc_dropout > 0:
        architecture_conf.mc_dropout = 0.0

    if is_baseline:
        if model_params.get("use_deepensemble", False):
            # Use DeepEnsembleTrainer
            return DeepEnsembleTrainer(
                architecture_conf=architecture_conf,
                lr=model_params.training.get("lr", 1e-3),
                classifier=data_params.get("classification", False),
                weight_decay=model_params.training.get("weight_decay", 0),
                use_scheduler=model_params.training.get("use_scheduler", True),
                ensemble_size=model_params.training.get("ensemble_size", 5),
            )
        if model_params.get("use_mve", False):
            return MVETrainer(
                architecture_conf=architecture_conf,
                lr=model_params.training.get("lr", 1e-2),
                classifier=data_params.get("classification", False),
                weight_decay=model_params.training.get("weight_decay", 0),
                use_scheduler=model_params.training.get("use_scheduler", True),
            )
        if model_params.get("use_mcdropout", False):
            if architecture_conf.mc_dropout == 0:
                print(
                    "Warning: MCDropoutTrainer requires a non-zero mc_dropout rate in the architecture configuration. Modifying mc_dropout to a default of 0.2."
                )
                architecture_conf.mc_dropout = 0.2
            return MCDropoutTrainer(
                architecture_conf=architecture_conf,
                lr=model_params.training.get("lr", 1e-2),
                classifier=data_params.get("classification", False),
                weight_decay=model_params.training.get("weight_decay", 0),
                use_scheduler=model_params.training.get("use_scheduler", True),
            )
        return NetTrainer(
            architecture_conf=architecture_conf,
            lr=model_params.training.get("lr", 1e-3),
            classifier=data_params.get("classification", False),
            weight_decay=model_params.training.get("weight_decay", 0),
            use_scheduler=model_params.training.get("use_scheduler", True),
            lr_schedule_factor=model_params.training.get("lr_schedule_factor", 0.5),
            lr_schedule_patience=model_params.training.get("lr_schedule_patience", 20),
        )
    else:
        trainer = DiffusionTrainer(
            architecture_conf=architecture_conf,
            diffusion_params=data_params.diffusion_params,
            num_pred_samples=model_params.training.num_pred_samples,
            lr=model_params.training.lr,
            use_scheduler=model_params.training.get("use_scheduler", True),
            n_jobs=model_params.training.get("n_jobs", 1),
            classifier=data_params.get("classification", False),
            weight_decay=model_params.training.get("weight_decay", 0),
            loss_weights=model_params.training.get("loss_weights", True),
            debug_sampling=model_params.training.get("debug_sampling", False),
            device=exp_params.device,
        )

    return trainer

def _is_uncertainty_baseline(model_params) -> bool:
    return (
        model_params.get("use_mve", False)
        or model_params.get("use_deepensemble", False)
        or model_params.get("use_mcdropout", False)
    )

def _get_arch_conf(model_params, datamodule) -> ArchitectureConfig:
    feature_dim = datamodule.feature_dim
    edge_dim = datamodule.edge_feat_dim
    target_dim = datamodule.target_dim

    return ArchitectureConfig(
        input_dim=feature_dim,
        edge_dim=edge_dim,
        output_dim=target_dim,
        degree=datamodule.degree_histogram,
        pe_dim=(datamodule.min_num_nodes - 1)
        if datamodule.min_num_nodes is not None
        else None,
        **model_params.hparams,
    )

def get_datamodule(
    batch_size,
    featurizer,
    seed: int = 0,
    classification=False,
    net_type: str | None = None,
    **data_params,
):
    data_name = data_params.pop("name")
    params = {k: v for k, v in data_params.items() if k != "diffusion_params"}

    if data_name == "tdc":
        datamodule = TDCDatamodule(
            batch_size=batch_size, featurizer=featurizer, seed=seed, **params
        )
    elif data_name == "molecule_ace":
        datamodule = MoleculeAceDatamodule(
            batch_size=batch_size, featurizer=featurizer, seed=seed, **params
        )
    elif data_name == "polaris":
        datamodule = PolarisDatamodule(
            batch_size=batch_size, featurizer=featurizer, seed=seed, **params
        )
    elif data_name == "dockstring":
        datamodule = DockstringDatamodule(
            batch_size=batch_size, featurizer=featurizer, seed=seed, **params
        )
    elif data_name == "dataframe_masked_imputation":
        datamodule = MaskedDataframeImputationDatamodule(
            batch_size=batch_size, featurizer=featurizer, seed=seed, **params
        )
    else:
        raise ValueError(f"Datamodule for {data_name} not supported")

    datamodule.net_type = net_type
    return datamodule

def extract_run_results(run_path: str, metrics: list[str] | None = None) -> list | None:
    """
    Load the results and config for a single run from a run root path to a dict with tuple entries of the form (dataset, model, seed, metric) : value.

    Args:
        results (list): The results as a list of dataframes.

    Returns:
        pd.DataFrame: The results as a DataFrame.
    """
    if metrics is None:
        metrics = ["mae", "rmse", "pearsonr", "spearmanr"]

    try:
        config = load_config(run_path)
        results = load_results(run_path)
    except ValueError:
        return None
    if results is None:
        return []
    dataset = config.data.dataset
    model_hparams = {
        k: v
        for k, v in config.model.hparams.items()
        if not (OmegaConf.is_list(v) or OmegaConf.is_dict(v))
    }
    train_hparams = {
        k: v
        for k, v in config.model.training.items()
        if not (OmegaConf.is_list(v) or OmegaConf.is_dict(v))
    }
    diffusion_params = {
        k: v
        for k, v in config.data.diffusion_params.items()
        if not (OmegaConf.is_list(v) or OmegaConf.is_dict(v))
    }
    exp_params = {
        k: v
        for k, v in config.experiment.items()
        if not (OmegaConf.is_list(v) or OmegaConf.is_dict(v))
    }

    if "net_type" not in model_hparams:
        model_hparams["net_type"] = "mlp"

    dataloader_res = []
    for ix, test_res in enumerate(results["test"]):
        test_res = {
            **test_res,
            **model_hparams,
            **diffusion_params,
            **exp_params,
            **train_hparams,
        }
        test_res["dataset"] = dataset
        test_df = pd.DataFrame([test_res])
        test_df = test_df.rename(
            mapper={
                "test_0_mae": "test_0/mae",
                "test_0_pearsonr": "test_0/pearsonr",
                "test_0_spearmanr": "test_0/spearmanr",
                "test_0_rmse": "test_0/rmse",
            },
            axis="columns",
        )
        train_results = results.get("train")
        if isinstance(train_results, list) and len(train_results) > 0:
            # Train metrics are shared across test dataloader indices.
            train_df = pd.DataFrame([train_results[0]])
            test_df = pd.concat([test_df, train_df], axis=1)
        elif isinstance(train_results, dict) and len(train_results) > 0:
            # Prefer key 0 when present, else use the first available value.
            train_entry = (
                train_results[0] if 0 in train_results else next(iter(train_results.values()))
            )
            train_df = pd.DataFrame([train_entry])
            test_df = pd.concat([test_df, train_df], axis=1)

        val_results = results.get("val")
        if isinstance(val_results, list) and len(val_results) > 0:
            # Validation is shared across test dataloader indices.
            val_df = pd.DataFrame([val_results[0]])
            test_df = pd.concat([test_df, val_df], axis=1)
        elif isinstance(val_results, dict) and len(val_results) > 0:
            # Prefer key 0 when present, else use the first available value.
            val_entry = val_results[0] if 0 in val_results else next(iter(val_results.values()))
            val_df = pd.DataFrame([val_entry])
            test_df = pd.concat([test_df, val_df], axis=1)

        # Filter out duplicate column names so downstream row-wise concat is safe.
        test_df = test_df.loc[:, ~test_df.columns.duplicated()]
       
        dataloader_res.append(test_df)

    return dataloader_res


def batch_extract_run_results(
    root_dir: str, metrics: list[str] | None = None
) -> dict[int, pd.DataFrame]:
    """
    Extract the results for all runs in the given root directory and return them as a DataFrame.

    Args:
        root_dir (str): The root directory where the results are located.
        metrics (list[str]): The metrics to extract from the results.

    Returns:
        pd.DataFrame: The results as a DataFrame.
    """

    if metrics is None:
        metrics = ["mae", "rmse", "pearsonr", "spearmanr"]

    # Discover Hydra run roots recursively so nested sweep layouts are supported.
    config_paths = glob.glob(
        f"{root_dir}/**/.hydra/config.yaml",
        recursive=True,
        include_hidden=True,
    )
    result_dirs = sorted({osp.dirname(osp.dirname(path)) for path in config_paths})
    all_results: dict = {}

    for run_path in result_dirs:
        if osp.isdir(run_path):
            run_result_df = extract_run_results(run_path, metrics)
            if run_result_df is None:
                continue
            for idx, df in enumerate(run_result_df):
                if idx not in all_results:
                    all_results[idx] = []
                all_results[idx].append(df)
    for idx in all_results:
        all_results[idx] = pd.concat(all_results[idx]).reset_index(drop=True)

    return all_results

def extract_tag_from_run_path(run_dir: Path, tag: str) -> str | None:
    run_str = str(run_dir)
    matches = re.findall(rf"{re.escape(tag)}=([^,/]+)", run_str)
    if not matches:
        return None
    return matches[-1]

def _is_true_tag(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "y"}

def resolve_uncertainty_method(run_dir: Path) -> str | None:
    model_name = extract_tag_from_run_path(run_dir, "model")
    if model_name is not None and model_name.startswith("simple_diff"):
        return "DIFFPRED"
    if _is_true_tag(extract_tag_from_run_path(run_dir, "model.use_mve")):
        return "MVE"
    if _is_true_tag(extract_tag_from_run_path(run_dir, "model.use_mcdropout")):
        return "MCDropout"
    if _is_true_tag(extract_tag_from_run_path(run_dir, "model.use_deepensemble")):
        return "DeepEnsemble"
    if _is_true_tag(extract_tag_from_run_path(run_dir, "model.use_deep_ensemble")):
        return "DeepEnsemble"
    return None