import os
from scipy import stats
from scipy.stats import norm, linregress
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from diffpred.utils.mmd import mmd
from diffpred.trainers import (
    DiffusionTrainer,
    MVETrainer,
    MCDropoutTrainer,
    DeepEnsembleTrainer,
)
from diffpred.viz.utils import set_mpl_settings


def _get_split_dataloader(datamodule: pl.LightningDataModule, split: str = "test"):
    if split == "test":
        test_dl = datamodule.test_dataloader()
        if isinstance(test_dl, list):
            test_dl = test_dl[0]  # molecule ACE returns a list, take the first for viz
        return test_dl
    elif split == "val":
        val_dl = datamodule.val_dataloader()
        if isinstance(val_dl, list):
            val_dl = val_dl[0]
        return val_dl
    elif split == "train":
        train_dl = datamodule.train_dataloader()
        if isinstance(train_dl, list):
            train_dl = train_dl[0]
        return train_dl
    else:
        raise ValueError(f"Invalid split {split}")


def get_cdf_distribution(pred_samples, y_test, y_std: torch.FloatTensor | None = None):
    # pred_samples is of shape num_samples x batch
    cdf_samples = []

    if y_std is None:
        for ix in range(y_test.shape[0]):
            ecdf = stats.ecdf(pred_samples[:, ix])
            cdf_sample = ecdf.cdf.evaluate(y_test[ix])
            cdf_samples.append(cdf_sample)
    else:
        # filter nans from y_test
        nan_mask = torch.isnan(y_test)
        y_test = y_test[~nan_mask]
        pred_samples = pred_samples[~nan_mask]
        y_std = y_std[~nan_mask]

        for ix in range(y_test.shape[0]):
            dist = torch.distributions.Normal(
                loc=pred_samples[ix].squeeze(), scale=y_std[ix].squeeze()
            )
            cdf_sample = dist.cdf(y_test[ix])
            cdf_samples.append(cdf_sample)

    return np.array(cdf_samples).squeeze()


def get_mc_dropout_samples(
    model: MCDropoutTrainer,
    datamodule: pl.LightningDataModule,
    split: str = "test",
):
    dataloader = _get_split_dataloader(datamodule, split=split)
    all_samples = []
    all_targets = []
    for batch in dataloader:
        x, y = batch
        samples = model.predict(
            x.to(model.device),
            return_samples=True,
        )  # [num_samples, batch, output_dim]
        all_samples.append(samples)
        all_targets.append(y)
    # Concatenate along batch dimension
    samples = torch.cat(all_samples, dim=1)  # [num_samples, total_batch, output_dim]
    targets = torch.cat(all_targets, dim=0)
    targets = datamodule.inverse_target_transform(targets)
    samples = samples.detach().cpu()
    targets = targets.detach().cpu()
    # Squeeze last dim if output_dim == 1
    if samples.shape[-1] == 1:
        samples = samples.squeeze(-1)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.squeeze(-1)
    return samples, targets


def _get_dataloader_preds(
    model: DiffusionTrainer, dataloader: DataLoader, num_samples: int = 100
):
    all_preds = []
    all_targets = []
    for batch in dataloader:
        x, y = batch
        preds = model.predict_samples(
            x.to(model.device),
            device=model.device,
            reduce=False,
            num_samples=num_samples,
        )
        assert (
            preds.ndim == 3
            and preds.shape[0] == num_samples
            and preds.shape[1] == y.shape[0]
            and preds.shape[2] == y.shape[1]
        )
        all_preds.append(preds)
        all_targets.append(y)

    y_pred = torch.cat(all_preds, dim=1)  # shape: (num_samples, batch_size, 1)
    targets = torch.cat(all_targets)

    return y_pred, targets


def get_test_samples(
    model: DiffusionTrainer,
    datamodule: pl.LightningDataModule,
    num_samples: int = 1000,
    split: str = "test",
):
    dataloader = _get_split_dataloader(datamodule, split=split)
    y_pred, targets = _get_dataloader_preds(model, dataloader, num_samples=num_samples)
    targets = datamodule.inverse_target_transform(targets)
    # Move to CPU for numpy operations
    y_pred = y_pred.detach().cpu()
    targets = targets.detach().cpu()

    # Check that the samples are in the expected dim
    assert y_pred.shape[0] == num_samples

    # Assume shape: (num_samples, batch_size, 1) or (num_samples, batch_size)
    if y_pred.ndim == 3:
        if y_pred.shape[2] != 1:
            raise ValueError("Expected last dimension to be 1")
        y_pred = y_pred.squeeze(-1)

    # Assume shape: (batch_size, 1) or (batch_size)
    if targets.ndim == 2:
        if targets.shape[1] != 1:
            raise ValueError("Expected last dimension to be 1")
        targets = targets.squeeze(-1)

    return y_pred, targets


def get_baseline_preds_with_uncertainty(
    model: MVETrainer,
    datamodule: pl.LightningDataModule,
    split: str = "test",
    return_samples: bool = False,
    num_samples: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
    dataloader = _get_split_dataloader(datamodule, split=split)
    all_preds = []
    all_std = []
    all_targets = []
    for batch in dataloader:
        x, y = batch
        preds, std = model.predict(
            x.to(model.device),
            return_proba=True,  # return the mean and std
        )
        all_preds.append(preds)
        all_std.append(std)
        all_targets.append(y)

    y_pred = torch.cat(all_preds, dim=0)  # shape: (batch_size, 1)
    std = torch.cat(all_std, dim=0)  # shape: (batch_size, 1)
    targets = torch.cat(all_targets, dim=0)  # shape: (batch_size, 1)
    targets = datamodule.inverse_target_transform(targets)

    # Move to CPU for numpy operations
    y_pred = y_pred.detach().cpu()
    std = std.detach().cpu()
    targets = targets.detach().cpu()

    if not return_samples:
        return y_pred, std, targets
    else:
        # Sample from the normal distribution
        eps = torch.randn((num_samples, std.shape[0], 1))
        pred_samples = y_pred + std * eps
        pred_samples = pred_samples.detach().cpu()
        return pred_samples, targets


def calibration_plots(
    model: DiffusionTrainer | MVETrainer | MCDropoutTrainer,
    datamodule: pl.LightningDataModule,
    output_dir: str,
):
    # set color palette
    set_mpl_settings()
    sns.set_palette("colorblind")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(model, DiffusionTrainer):
        test_pred, test_targets = get_test_samples(model, datamodule, split="test")
        val_pred, val_targets = get_test_samples(model, datamodule, split="val")

    elif isinstance(model, (MVETrainer, DeepEnsembleTrainer, MCDropoutTrainer)):
        test_pred, test_std, test_targets = get_baseline_preds_with_uncertainty(
            model, datamodule, split="test"
        )
        val_pred, val_std, val_targets = get_baseline_preds_with_uncertainty(
            model, datamodule, split="val"
        )
    else:
        return

    ### ECDF calibration plot
    ecdf_calibration_fig_val, mmd_val = ecdf_calibration_plot_fn(
        val_pred,
        val_targets,
        compute_calibration=False,
        y_std=(
            val_std.abs()
            if isinstance(model, (MVETrainer, DeepEnsembleTrainer, MCDropoutTrainer))
            else None
        ),
    )
    ecdf_calibration_fig, mmd_test = ecdf_calibration_plot_fn(
        test_pred,
        test_targets,
        y_std=(
            test_std.abs()
            if isinstance(model, (MVETrainer, DeepEnsembleTrainer, MCDropoutTrainer))
            else None
        ),
    )
    # Save
    save_path = os.path.join(output_dir, "ecdf_calibration.png")
    save_path_val = os.path.join(output_dir, "ecdf_calibration_val.png")
    save_path_pdf = os.path.join(output_dir, "ecdf_calibration.pdf")
    save_path_pdf_val = os.path.join(output_dir, "ecdf_calibration_val.pdf")

    # save pdf
    ecdf_calibration_fig_val.savefig(save_path_pdf_val)
    ecdf_calibration_fig.savefig(save_path_pdf)

    # save png
    ecdf_calibration_fig_val.savefig(save_path_val)
    ecdf_calibration_fig.savefig(save_path)
    plt.close(ecdf_calibration_fig)
    plt.close(ecdf_calibration_fig_val)

    # save preds and targets
    save_path_preds = os.path.join(output_dir, "calibration_preds.npy")
    save_path_targets = os.path.join(output_dir, "calibration_targets.npy")
    if isinstance(model, (MVETrainer, DeepEnsembleTrainer)):
        save_path_std = os.path.join(output_dir, "calibration_std.npy")
        np.save(save_path_std, test_std.numpy())

    # Save MMD values for stacking
    mmd_array = np.array([mmd_test, mmd_val])
    pd.Series(mmd_array, index=["test", "val"]).to_csv(
        os.path.join(output_dir, "ecdf_mmd_values.csv")
    )

    np.save(save_path_preds, test_pred.numpy())
    np.save(save_path_targets, test_targets.numpy())


def ecdf_calibration_plot_fn(
    pred_samples: torch.Tensor,
    target: torch.Tensor,
    compute_calibration: bool = False,
    calibration: float | None = None,
    y_std: torch.FloatTensor | None = None,
) -> plt.Figure:
    cdf_samples = get_cdf_distribution(pred_samples, target, y_std=y_std)

    # Convert CDF samples to standard normal via inverse quantile transform
    normal_samples = norm.ppf(cdf_samples)

    # Sort the normal samples
    normal_samples_sorted = np.sort(normal_samples)
    n = len(normal_samples_sorted)

    # Theoretical quantiles
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = norm.ppf(probs)

    mmd_value = mmd(normal_samples_sorted, theoretical_quantiles)

    if compute_calibration:
        # Compute the calibration factor as inverse of slope
        # remove any inf, -inf values from normal samples
        inf_mask = np.isinf(normal_samples_sorted)
        calibration_samples = normal_samples_sorted[~inf_mask]
        calibration = (
            1 / linregress(theoretical_quantiles[~inf_mask], calibration_samples).slope
        )

    fig, ax = plt.subplots(figsize=(7, 5))

    # Main plot
    if calibration is not None:
        ax.plot(
            theoretical_quantiles,
            normal_samples_sorted * calibration,
            "o",
            label="Empirical*calibration",
            alpha=0.5,
        )
        mmd_calibrated = mmd(normal_samples_sorted * calibration, theoretical_quantiles)
        ax.annotate(
            f"MMD (calibrated): {mmd_calibrated:.2f}",
            (0.05, 0.75),
            xycoords="axes fraction",
        )

    ax.plot(
        theoretical_quantiles, normal_samples_sorted, "o", label="Empirical", alpha=0.5
    )
    ax.plot(theoretical_quantiles, theoretical_quantiles, "r--", label="Ideal")

    ax.annotate(f"MMD: {mmd_value:.2f}", (0.05, 0.85), xycoords="axes fraction")
    ax.set_xlabel("Theoretical Quantiles")  # Remove x-label from top plot
    ax.set_ylabel("Sample Quantiles")
    # set x tick size
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="lower right")
    # Adjust spacing between subplots
    plt.tight_layout()

    if compute_calibration:
        return fig, mmd_value, calibration
    return fig, mmd_value
