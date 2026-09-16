from abc import ABC, abstractmethod
from typing import Literal, Protocol, cast
import torch
import torch.nn as nn
from torch_geometric.data import Data
from chemprop import nn as chemprop_nn
from diffpred.models.networks.graph.layers import (
    GCNLayer,
    PNALayer,
    GINLayer,
    GINELayer,
    GPSLayer,
)
from diffpred.models.networks.graph.graphormer_helpers.data import (
    GraphormerBatch,
    build_graphormer_batch,
)
from diffpred.models.networks.graph.graphormer_helpers.layers import (
    GraphormerGraphEncoder,
)
from diffpred.models.networks.nn_arch import MLP, ResidualOutNet
from diffpred.constants.activations import ACTIVATIONS

from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import (
    GraphMultisetTransformer,
    global_add_pool,
    global_mean_pool,
)


def pool(x, data, pooling: str, pooling_nn=None):
    prev_device = x.device
    x = x.to("cpu")
    data = data.to("cpu")
    if pooling == "mean":
        out = global_mean_pool(x, data.batch)
    elif pooling == "sum":
        out = global_add_pool(x, data.batch)
    elif pooling == "gmt":
        if pooling_nn is None:
            raise ValueError("Expected network for pooling for multiset transformer")
        out = pooling_nn(x, data.batch)
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

    data = data.to(prev_device)
    out = out.to(prev_device)
    return out


def _build_readout_mlp(
    *,
    output_mlp_type: str,
    num_output_layers: int,
    output_hidden: int,
    output_dim: int | None,
    activation,
    in_dim: int,
) -> nn.Module:
    if output_dim is None:
        raise ValueError("output_dim must be set to build readout")

    out_mlp = nn.ModuleList()
    if output_mlp_type == "linear":
        for i in range(num_output_layers):
            out_mlp.append(nn.Linear(in_dim, output_hidden // 2**i))
            out_mlp.append(activation())
            in_dim = output_hidden // 2**i
        out_mlp.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*out_mlp)

    if output_mlp_type == "residual":
        return ResidualOutNet(
            num_hiddens=num_output_layers,
            input_dim=in_dim,
            hidden_dim=output_hidden,
            out_dim=output_dim,
            activation=activation,
        )

    raise ValueError(
        f"Invalid output_mlp_type: {output_mlp_type}. "
        "Choose either 'linear' or 'residual'."
    )


class _ChempropBatchLike(Protocol):
    V: torch.Tensor
    E: torch.Tensor
    edge_index: torch.Tensor
    rev_edge_index: torch.Tensor
    batch: torch.Tensor


class ChempropMPNNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        edge_dim: int | None,
        output_dim=None,
        activation="relu",
        dropout=0.0,
        pooling: str = "sum",
        output_hidden: int | None = None,
        num_output_layers: int = 3,
        output_mlp_type: Literal["linear", "residual"] = "linear",
        message_passing: Literal["bond", "atom"] = "bond",
        aggregation: Literal["mean", "sum", "norm"] = "mean",
        depth: int = 3,
    ):
        super().__init__()
        if len(hidden_channels) == 0:
            raise ValueError("ChempropMPNNet requires at least one hidden channel")
        if edge_dim is None:
            raise ValueError("ChempropMPNNet requires edge_dim for molecular graphs")

        self.hiddens = hidden_channels
        self.output_dim = output_dim
        self.output_mlp_type = output_mlp_type
        self.num_output_layers = num_output_layers
        self.activation = ACTIVATIONS[activation]
        self.output_hidden = output_hidden
        self.pooling = pooling
        self.message_passing = message_passing

        d_h = hidden_channels[-1]
        mp = getattr(
            chemprop_nn,
            "BondMessagePassing" if message_passing == "bond" else "AtomMessagePassing",
        )
        self.message_passing_layer = mp(
            d_v=in_channels,
            d_e=edge_dim,
            d_h=d_h,
            depth=depth,
            dropout=dropout,
            activation=self.activation(),
        )
        self.aggregation_layer = self._build_aggregation(aggregation)
        if output_dim is not None:
            in_dim = self._get_output_mlp_in_dim()
            self.mlp_readout = self._build_output_mlp(in_dim)
        else:
            self.mlp_readout = None

    def _build_aggregation(self, aggregation: str):
        if aggregation == "mean":
            return chemprop_nn.MeanAggregation()
        if aggregation == "sum":
            return chemprop_nn.SumAggregation()
        if aggregation == "norm":
            return chemprop_nn.NormAggregation()
        raise ValueError(f"Unknown Chemprop aggregation: {aggregation}")

    def _get_output_mlp_in_dim(self):
        return self.hiddens[-1]

    def _build_output_mlp(self, in_dim: int) -> nn.Module:
        if self.output_hidden is None:
            self.output_hidden = in_dim
        return _build_readout_mlp(
            output_mlp_type=self.output_mlp_type,
            num_output_layers=self.num_output_layers,
            output_hidden=self.output_hidden,
            output_dim=self.output_dim,
            activation=self.activation,
            in_dim=in_dim,
        )

    class _TorchBatchMolGraph:
        # Minimal BatchMolGraph-compatible container used by Chemprop message passing.
        def __init__(self, V, E, edge_index, rev_edge_index, batch):
            self.V = V
            self.E = E
            self.edge_index = edge_index
            self.rev_edge_index = rev_edge_index
            self.batch = batch

        def __len__(self) -> int:
            return int(self.batch.max().item()) + 1 if self.batch.numel() > 0 else 0

        def to(self, device: str | torch.device):
            self.V = self.V.to(device)
            self.E = self.E.to(device)
            self.edge_index = self.edge_index.to(device)
            self.rev_edge_index = self.rev_edge_index.to(device)
            self.batch = self.batch.to(device)
            return self

    def _build_reverse_edge_index(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        keys = src * num_nodes + dst
        rev_keys = dst * num_nodes + src

        sorted_keys, perm = torch.sort(keys)
        positions = torch.searchsorted(sorted_keys, rev_keys)
        valid = (positions < sorted_keys.shape[0]) & (sorted_keys[positions] == rev_keys)

        rev_edge_index = torch.arange(edge_index.shape[1], device=edge_index.device)
        rev_edge_index[valid] = perm[positions[valid]]
        return rev_edge_index.long()

    def _to_batch_mol_graph(self, data: Data) -> _ChempropBatchLike:
        if data.x is None:
            raise ValueError("ChempropMPNNet expects node features in data.x")
        if data.edge_index is None:
            raise ValueError("ChempropMPNNet expects edge_index in graph batch")
        if data.edge_attr is None:
            raise ValueError("ChempropMPNNet expects edge features for molecular graphs")

        batch = data.batch
        if batch is None:
            batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)

        rev_edge_index = self._build_reverse_edge_index(
            edge_index=data.edge_index,
            num_nodes=int(data.x.shape[0]),
        )

        return ChempropMPNNet._TorchBatchMolGraph(
            V=data.x.float(),
            E=data.edge_attr.float(),
            edge_index=data.edge_index.long(),
            rev_edge_index=rev_edge_index,
            batch=batch.long(),
        )

    def forward(self, data):
        if isinstance(data, list):
            out = [self.forward(d) for d in data]
            return torch.cat(out, dim=0)

        batch_mol_graph = self._to_batch_mol_graph(data)
        node_embeddings = self.message_passing_layer(batch_mol_graph)
        out = self.aggregation_layer(node_embeddings, batch_mol_graph.batch)
        if self.mlp_readout is not None:
            out = self.mlp_readout(out)
        return out


class GraphormerNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        edge_dim: int | None,
        output_dim: int | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        output_hidden: int | None = None,
        num_output_layers: int = 3,
        output_mlp_type: Literal["linear", "residual"] = "linear",
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_embedding_dim: int | None = None,
        edge_type: Literal["multi_hop", "single_hop"] = "multi_hop",
        multi_hop_max_dist: int = 5,
        num_spatial: int = 512,
        num_in_degree: int = 512,
        num_out_degree: int = 512,
        num_edge_dis: int = 128,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        pre_layernorm: bool = False,
    ):
        super().__init__()
        if len(hidden_channels) == 0:
            raise ValueError("GraphormerNet requires at least one hidden channel")
        if edge_dim is None:
            raise ValueError("GraphormerNet requires edge_dim for graph features")

        embedding_dim = hidden_channels[-1]
        ffn_embedding_dim = (
            embedding_dim if ffn_embedding_dim is None else ffn_embedding_dim
        )
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"Graphormer embedding dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hiddens = hidden_channels
        self.output_dim = output_dim
        self.output_mlp_type = output_mlp_type
        self.num_output_layers = num_output_layers
        self.activation = ACTIVATIONS[activation]
        self.output_hidden = output_hidden
        self.edge_type: Literal["multi_hop", "single_hop"] = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist

        num_atoms = max(512, in_channels * 512)
        num_edges = max(512, edge_dim * 512)
        self.graphormer_encoder = GraphormerGraphEncoder(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            num_encoder_layers=num_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation,
            pre_layernorm=pre_layernorm,
        )

        if output_dim is not None:
            in_dim = self._get_output_mlp_in_dim()
            self.mlp_readout = self._build_output_mlp(in_dim)
        else:
            self.mlp_readout = None

    def _get_output_mlp_in_dim(self):
        return self.hiddens[-1]

    def _build_output_mlp(self, in_dim: int) -> nn.Module:
        if self.output_hidden is None:
            self.output_hidden = in_dim
        return _build_readout_mlp(
            output_mlp_type=self.output_mlp_type,
            num_output_layers=self.num_output_layers,
            output_hidden=self.output_hidden,
            output_dim=self.output_dim,
            activation=self.activation,
            in_dim=in_dim,
        )

    def forward(self, data):
        if isinstance(data, list):
            out = [self.forward(d) for d in data]
            return torch.cat(out, dim=0)

        if isinstance(data, GraphormerBatch):
            batched_data = data
        else:
            batched_data = build_graphormer_batch(
                data,
                edge_type=self.edge_type,
                multi_hop_max_dist=self.multi_hop_max_dist,
            )
        out = self.graphormer_encoder(batched_data)
        if self.mlp_readout is not None:
            out = self.mlp_readout(out)
        return out


class BaseGraphNet(nn.Module, ABC):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        output_dim=None,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        pooling: str = "sum",
        output_hidden: int | None = None,
        num_output_layers: int = 3,
        output_mlp_type: Literal["linear", "residual"] = "linear",
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.mlp_readout: nn.Module | None = None
        self.output_dim = output_dim
        self.output_mlp_type = output_mlp_type
        self.num_output_layers = num_output_layers
        self.activation = ACTIVATIONS[activation]
        self.hiddens = hidden_channels
        self.input_dim = in_channels
        self.uses_atom_encoder = (
            True if in_channels == 9 else False
        )  # Account for AtomEncoder - assumes dim is only 9 for OGB datasets (which use AtomEncoder)
        in_channels = in_channels if not self.uses_atom_encoder else hidden_channels[0]
        self.encoder = AtomEncoder(emb_dim=hidden_channels[0])
        self.pooling = pooling
        self.pooling_nn = (
            GraphMultisetTransformer(channels=hidden_channels[-1], k=1)
            if pooling == "gmt"
            else None
        )
        self.output_hidden = (
            output_hidden
            if output_hidden is not None
            else self._get_output_mlp_in_dim()
        )
        self.layers = nn.ModuleList()
        for _, hidden in enumerate(hidden_channels):
            self.layers.append(
                self.create_layer(
                    in_channels,
                    hidden,
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                )
            )
            in_channels = hidden
        self.layers = nn.Sequential(*self.layers)
        if output_dim is not None:
            in_dim = self._get_output_mlp_in_dim()
            self.mlp_readout = self._build_output_mlp(in_dim)
        else:
            self.mlp_readout = None

    @property
    @abstractmethod
    def supports_sparse_tensor(self):
        pass

    @abstractmethod
    def create_layer(self, in_channels, out_channels, activation, dropout, batch_norm):
        pass

    def __forward(self, data) -> torch.Tensor:
        feats = data.x if not self.uses_atom_encoder else self.encoder(data.x)
        feats = self._pre_feature_handling(feats, data)
        graph_connectivity = (
            data.adj_t
            if self.supports_sparse_tensor and hasattr(data, "adj_t")
            else data.edge_index
        )

        for layer in self.layers:
            feats = layer(
                feats,
                edge_index=graph_connectivity,
                edge_attr=data.edge_attr,
            )

        out = pool(feats, data, self.pooling, self.pooling_nn)
        if self.mlp_readout is not None:
            out = self.mlp_readout(out)
        return out

    def _pre_feature_handling(self, feats, data):
        return feats

    def forward(self, data):
        if isinstance(data, list):
            out = [self.__forward(d) for d in data]
            return torch.cat(out)
        return self.__forward(data)

    def _get_output_mlp_in_dim(self):
        return self.hiddens[-1] if len(self.hiddens) > 0 else self.input_dim

    def _build_output_mlp(self, in_dim: int) -> nn.Module:
        if self.output_hidden is None:
            raise ValueError("output_hidden must be set to build readout")
        
        return _build_readout_mlp(
            output_mlp_type=self.output_mlp_type,
            num_output_layers=self.num_output_layers,
            output_hidden=self.output_hidden,
            output_dim=self.output_dim,
            activation=self.activation,
            in_dim=in_dim,
        )


class GCNNet(BaseGraphNet):
    def create_layer(self, in_channels, out_channels, activation, dropout, batch_norm):
        return GCNLayer(
            in_channels,
            out_channels,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    @property
    def supports_sparse_tensor(self):
        return True


class PNANet(BaseGraphNet):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        degree,
        output_dim=None,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        pooling: str = "sum",
        output_hidden: int | None = None,
        output_mlp_type: Literal["linear", "residual"] = "linear",
        num_output_layers: int = 3,
    ):
        self.degree = degree
        super().__init__(
            in_channels,
            hidden_channels,
            output_dim,
            activation,
            dropout,
            batch_norm,
            pooling,
            num_output_layers=num_output_layers,
            output_hidden=output_hidden,
            output_mlp_type=output_mlp_type,
        )

    @property
    def supports_sparse_tensor(self):
        return False

    def create_layer(self, in_channels, out_channels, activation, dropout, batch_norm):
        return PNALayer(
            in_channels,
            out_channels,
            self.degree,
            activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )


class GINNet(BaseGraphNet):
    @property
    def supports_sparse_tensor(self):
        return True

    def create_layer(self, in_channels, out_channels, activation, dropout, batch_norm):
        return GINLayer(
            in_channels,
            out_channels,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )


class GINENet(BaseGraphNet):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        edge_dim,
        output_dim=None,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        pooling: str = "sum",
        output_hidden: int | None = None,
        num_output_layers: int = 3,
        output_mlp_type: Literal["linear", "residual"] = "residual",
    ):
        self.edge_dim = edge_dim
        super().__init__(
            in_channels,
            hidden_channels,
            output_dim,
            activation,
            dropout,
            batch_norm,
            pooling,
            output_hidden=output_hidden,
            num_output_layers=num_output_layers,
            output_mlp_type=output_mlp_type,
        )

    @property
    def supports_sparse_tensor(self):
        return True

    def create_layer(self, in_channels, out_channels, activation, dropout, batch_norm):
        return GINELayer(
            in_channels,
            out_channels,
            edge_dim=self.edge_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )


class GPSNet(BaseGraphNet):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        edge_dim,
        output_dim=None,
        activation="relu",
        dropout=0.0,
        norm: str = "batch_norm",
        pooling: str = "sum",
        pe_dim: int = 6,
        num_rw_steps: int = 10,
        encoding_dim: int = 16,
        output_hidden: int | None = None,
        output_mlp_type: Literal["linear", "residual"] = "residual",
        num_output_layers: int = 3,
    ):
        self.edge_dim = edge_dim
        self.norm = norm
        self.encoding_dim = encoding_dim
        self.pe_dim = pe_dim
        self.num_rw_steps = num_rw_steps
        in_channels = in_channels + self.encoding_dim
        super().__init__(
            in_channels,
            hidden_channels,
            output_dim,
            activation,
            dropout,
            pooling=pooling,
            output_hidden=output_hidden,
            output_mlp_type=output_mlp_type,
            num_output_layers=num_output_layers,
        )

        # PE_LAYERS
        self.pre_mlps = nn.ModuleDict()
        self.pre_mlps["rwse"] = nn.Sequential(
            nn.BatchNorm1d(num_rw_steps),
            nn.Linear(num_rw_steps, self.encoding_dim),
        )

    @property
    def supports_sparse_tensor(self):
        return True

    def create_layer(
        self, in_channels, out_channels, activation, dropout, *args, **kwargs
    ):
        return GPSLayer(
            in_channels,
            out_channels,
            edge_dim=self.edge_dim,
            activation=activation,
            dropout=dropout,
            norm=self.norm,
        )

    def _pre_feature_handling(self, feats: torch.Tensor, data: Data):
        if len(self.pre_mlps) == 0:
            return data

        pe_feats = []
        for pe in self.pre_mlps:
            pe_feats.append(self.pre_mlps[pe](data[pe]))
        pe_feats = torch.cat(pe_feats, dim=-1)
        feats = torch.cat([feats, pe_feats], dim=-1)
        return feats
