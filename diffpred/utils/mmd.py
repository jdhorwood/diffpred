import torch


def mmd(x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
    """
    Computes the Maximum Mean Discrepancy (MMD) between two distributions using the RBF kernel with variance 1.
    Args:
        x: First distribution samples.
        y: Second distribution samples.
    Returns:
        MMD value.
    """
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)

    # filter out infinite values
    x = x[~torch.isinf(x).any(dim=1)]
    y = y[~torch.isinf(y).any(dim=1)]
    # filter out NaN values
    x = x[~torch.isnan(x).any(dim=1)]
    y = y[~torch.isnan(y).any(dim=1)]

    n = x.size(0)
    x_kernel = torch.exp(-0.5 * torch.cdist(x, x, p=2) ** 2)
    y_kernel = torch.exp(-0.5 * torch.cdist(y, y, p=2) ** 2)
    xy_kernel = torch.exp(-0.5 * torch.cdist(x, y, p=2) ** 2)

    # unbiased MMD
    x_kernel = (x_kernel.sum() - n) / (n * (n - 1))
    y_kernel = (y_kernel.sum() - n) / (n * (n - 1))
    xy_kernel = (xy_kernel.sum()) / (n * n)

    mmd_value = (x_kernel + y_kernel - 2 * xy_kernel).clamp(
        min=0
    )  # clamp to avoid negative values
    mmd_value = torch.sqrt(mmd_value)
    return mmd_value
