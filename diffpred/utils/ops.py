import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import ToSparseTensor
from lightning.pytorch import seed_everything as pl_seed_everything


def configure_runtime_precision(matmul_precision: str = "medium"):
    """Set global torch matmul precision used by training/evaluation codepaths."""
    torch.set_float32_matmul_precision(matmul_precision)


def concat_sources(*tensors: torch.Tensor):
    """
    Concatenate tensors along the last dimension. Used for combining the feature, noisy target and time embeddings into a single input tensor.

    """
    tensor_list = [t for t in tensors if t is not None]
    inputs = torch.cat(tensor_list, dim=-1).float()
    return inputs


def match_dims(tensor: torch.Tensor, reference_tensor: torch.Tensor):
    """
    Repeat the tensor along the first dimension to match the shape of the reference tensor.
    """

    if tensor is not None and tensor.ndim != reference_tensor.ndim:
        tensor = tensor.repeat(reference_tensor.shape[0], 1, 1)
    return tensor


def seed_everything(seed: int):
    pl_seed_everything(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_runtime_precision("medium")
    # torch.use_deterministic_algorithms(True)
    return seed


def is_tensor_1d(tensor: torch.Tensor):
    return tensor.ndim == 1 or (tensor.ndim == 2 and tensor.shape[1] == 1)

def is_array_1d(array: np.ndarray):
    return array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1)

def is_tensor_dicrete(tensor: torch.Tensor, ignore_nans=True):
    if ignore_nans:
        mask = ~torch.isnan(tensor)
        tensor = tensor[mask]
    return torch.all(tensor == tensor.round())


def is_tensor_binary(tensor: torch.Tensor, ignore_nans=True):
    if ignore_nans:
        mask = ~torch.isnan(tensor)
        tensor = tensor[mask]
    return is_tensor_dicrete(tensor, ignore_nans) and torch.all(
        (tensor == 0) | (tensor == 1)
    )


def convert_to_symmetric_binary(tensor: torch.Tensor):
    if not is_tensor_binary(tensor):
        raise ValueError("Input tensor is not binary")

    # convert 0 to -1 and 1 to 1
    tensor = 2 * tensor - 1
    return tensor


def is_tensor_symmetric_binary(tensor: torch.Tensor, ignore_nans=True):
    if ignore_nans:
        mask = ~torch.isnan(tensor)
        tensor = tensor[mask]
    return torch.all((tensor == -1) | (tensor == 1))


def convert_to_binary(tensor: torch.Tensor):
    if not is_tensor_symmetric_binary(tensor):
        raise ValueError("Input tensor is not symmetric binary")

    # convert -1 to 0 and 1 to 1
    tensor = (tensor + 1) / 2
    return tensor


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output
