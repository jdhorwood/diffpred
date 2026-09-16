import torch
import torch.nn as nn
from diffpred.models.protocols import ArchitectureConfig
from diffpred.models.networks.factories.diffnet_factory import (
    diffusion_network_factory,
)


class DiffusionNN(nn.Module):
    def __init__(
        self,
        architecture_conf: ArchitectureConfig,
        classifier: bool = False,
    ):
        super().__init__()

        self.architecture_conf = architecture_conf
        self.net = diffusion_network_factory(architecture_conf, classifier=classifier)

    def forward(
        self,
        x: torch.FloatTensor,
        y_k: torch.FloatTensor | None = None,
        k: torch.LongTensor | None = None,
        self_cond: torch.FloatTensor | None = None,
    ):
        """
        The forward pass outputs the prediction of the noise at step k.

        x: torch.FloatTensor
            Input tensor corresponding to the features.
        y_k: torch.FloatTensor
            Noisy target value at step k.
        k: torch.LongTensor
            The step k.
        self_cond: torch.FloatTensor
            Optional self-conditioning signal.
        """
        features, targets, k = x, y_k, k
        if y_k is not None and k is not None:
            if (k.ndim == 0) or (k.shape[0] != len(x)):
                k = k * torch.ones(len(x), 1, device=k.device)
            return self.net.forward(features, targets, k, self_cond=self_cond)
        else:
            return self.net(x)

    @property
    def target_dim(self):
        return self.net.target_dim
