import torch
import torchmetrics.functional as metrics
from functools import partial
import torchmetrics.functional.classification as clf_metrics
from diffpred.utils.ops import (
    is_tensor_1d,
    is_tensor_symmetric_binary,
    convert_to_binary,
)


def bce_metric(preds, target):
    """
    Binary Cross-Entropy metric.
    """
    assert is_tensor_1d(preds) and is_tensor_1d(
        target
    ), "Both preds and target should be 1D tensors."
    assert (
        preds.shape == target.shape
    ), "Predictions and targets must have the same shape."
    assert (
        0 <= preds.min() and preds.max() <= 1
    ), "Predictions should be in the range [0, 1]."
    return -torch.nn.functional.binary_cross_entropy(
        input=preds, target=target.to(preds.dtype)
    )


def compute_metrics(
    y_true,
    y_pred,
    metric_fns,
    y_proba=None,
    split_name: str = "",
):
    """
    Compute metrics for a given set of metric functions.
    """

    if is_tensor_symmetric_binary(y_true):
        # put targets in [0, 1] format for binary classification metrics
        y_true = convert_to_binary(y_true)

    if is_tensor_symmetric_binary(y_pred):
        # put predictions in [0, 1] format for binary classification metrics
        y_pred = convert_to_binary(y_pred)

    res = {}
    for metric_name, metric_fn in metric_fns.items():
        if not is_tensor_1d(y_true):
            assert y_true.shape == y_pred.shape
            metric_values = []
            for i in range(y_true.shape[1]):
                val = _compute_single_column_metric(
                    y_true[:, i],
                    y_pred[:, i],
                    metric_name=metric_name,
                    metric_fn=metric_fn,
                    y_proba=(y_proba[:, i] if y_proba is not None else None),
                )
                val_tensor = torch.as_tensor(val, dtype=torch.float32).reshape(())
                metric_values.append(val_tensor)
                res[split_name + metric_name + f"_task_{i}"] = val_tensor.item()
            res[split_name + metric_name] = torch.stack(metric_values).mean().item()
        else:
            out = _compute_single_column_metric(
                y_true,
                y_pred,
                metric_name,
                metric_fn,
                y_proba,
            )

            res[split_name + metric_name] = (
                torch.as_tensor(out, dtype=torch.float32).reshape(()).item()
            )

    return res


def _compute_single_column_metric(
    y_true, y_pred, metric_name, metric_fn, y_proba, ignore_nans=True
) -> float | dict:
    assert is_tensor_1d(y_true) and is_tensor_1d(y_pred)

    if torch.isnan(y_true).any() and ignore_nans:
        mask = ~torch.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        y_proba = y_proba[mask] if y_proba is not None else None

    if metric_name in ("roc_auc", "pr_auc", "bce"):
        y_true = y_true.int()
        val = metric_fn(target=y_true, preds=y_proba)
    else:
        val = metric_fn(target=y_true, preds=y_pred)

    return val


REGRESSION_METRICS = {
    "mae": metrics.mean_absolute_error,
    "pearsonr": metrics.pearson_corrcoef,
    "spearmanr": metrics.spearman_corrcoef,
    "rmse": partial(metrics.mean_squared_error, squared=False),
    "r2_score": metrics.r2_score,
}

CLASSIFICATION_METRICS = {
    "accuracy": clf_metrics.binary_accuracy,
    "precision": clf_metrics.binary_precision,
    "recall": clf_metrics.binary_recall,
    "f1": clf_metrics.binary_f1_score,
    "roc_auc": clf_metrics.binary_auroc,
    "pr_auc": clf_metrics.binary_average_precision,
    "mcc": clf_metrics.binary_matthews_corrcoef,
    "bce": bce_metric,
}
