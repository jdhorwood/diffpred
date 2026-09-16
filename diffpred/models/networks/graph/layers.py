from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, PNAConv, GINConv, GINEConv, GPSConv
from diffpred.constants.activations import ACTIVATIONS


class BaseGraphLayer(nn.Module, ABC):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        residual: bool | None = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._init_layer_attrs(activation, dropout, batch_norm, residual)
        self.conv = self.create_conv()

    def get_residual_proj(self, x):
        if self.residual and self.in_channels != self.out_channels:
            return self.residual_proj(x)
        return x

    def forward(self, x, edge_index, edge_attr=None | torch.Tensor):
        if not self.supports_edge_attr():
            h = self.conv(x, edge_index)
        else:
            h = self.conv(x, edge_index, edge_attr)
        if self.batch_norm:
            h = self.bnorm(h)
        h = self.activation(h)
        h = self.dropout(h)

        if self.residual:
            h = h + self.get_residual_proj(x)

        return h

    def _init_layer_attrs(
        self,
        activation: str,
        dropout: float,
        batch_norm: bool,
        residual: bool | None = None,
    ):
        activation_fn = (
            ACTIVATIONS[activation] if activation is not None else nn.Identity
        )
        self.activation = activation_fn()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.residual = residual
        if self.residual and self.in_channels != self.out_channels:
            self.residual_proj = nn.Linear(self.in_channels, self.out_channels)

        self.bnorm = nn.BatchNorm1d(self.out_channels)

    @abstractmethod
    def create_conv(self):
        pass

    @abstractmethod
    def supports_edge_attr(self) -> bool:
        pass


class GCNLayer(BaseGraphLayer):
    def create_conv(self):
        return GCNConv(self.in_channels, self.out_channels)

    def supports_edge_attr(self) -> bool:
        return False


class PNALayer(BaseGraphLayer):
    AGGREGATORS = ["mean", "min", "max", "std"]
    SCALERS = ["identity", "amplification", "attenuation"]

    def __init__(
        self,
        in_channels,
        out_channels,
        degree,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        residual=True,
    ):
        self.degree = degree  # Keep track of the degree for PNA
        super().__init__(
            in_channels,
            out_channels,
            activation,
            dropout,
            batch_norm,
            residual,
        )

    def create_conv(self):
        return PNAConv(
            self.in_channels,
            self.out_channels,
            aggregators=self.AGGREGATORS,
            scalers=self.SCALERS,
            deg=self.degree,
        )

    def supports_edge_attr(self):
        return False


class GINLayer(BaseGraphLayer):
    def create_conv(self):
        layer_net = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )
        return GINConv(nn=layer_net)

    def supports_edge_attr(self) -> bool:
        return False


class GINELayer(BaseGraphLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        residual=True,
    ):
        self.edge_dim = edge_dim
        super().__init__(
            in_channels,
            out_channels,
            activation,
            dropout,
            batch_norm,
            residual,
        )

    def create_conv(self):
        layer_net = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )
        return GINEConv(nn=layer_net, edge_dim=self.edge_dim)

    def supports_edge_attr(self) -> bool:
        return True


class GPSLayer(BaseGraphLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        heads: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        norm: str = "batch_norm",
    ):
        self.edge_dim = edge_dim
        self.heads = heads
        self.norm = norm
        super().__init__(in_channels, out_channels, activation, dropout, residual=None)
        self.output_layer = self._create_output_layer()
        self.dropout_layer = nn.Dropout(dropout)  # Explicit dropout for GPSLayer

    def create_conv(self):
        self.mpnn = self._create_mpnn()
        return GPSConv(
            channels=self.in_channels,
            conv=self.mpnn,
            attn_type="performer",  # for scalability
            norm=self.norm,
            heads=self.heads,
            act=self.activation,
            dropout=self.dropout,
        )

    def _create_mpnn(self):
        mpnn_net = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.ReLU(),
            nn.Linear(self.in_channels, self.in_channels),
        )
        return GINEConv(nn=mpnn_net, edge_dim=self.edge_dim)

    def _init_layer_attrs(self, activation: str, dropout: float, *args, **kwargs):
        activation_fn = (
            ACTIVATIONS[activation] if activation is not None else nn.Identity
        )
        self.activation = activation_fn()
        self.dropout = dropout

    def supports_edge_attr(self) -> bool:
        return True

    def _create_output_layer(self):
        return (
            nn.Linear(self.in_channels, self.out_channels)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = self.output_layer(h)
        h = self.activation(h)
        h = self.dropout_layer(h)  # Dropout after activation and output layer
        return h
