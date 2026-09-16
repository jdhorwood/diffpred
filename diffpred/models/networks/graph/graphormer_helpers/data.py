# Core Graphormer components adapted from the official implementation: https://github.com/microsoft/Graphormer/tree/main
 

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch_geometric.data import Data
from ogb.utils.mol import smiles2graph


class OGBGraphormerFeaturizer:
    """SMILES -> PyG Data using OGB's canonical molecular graph featurization."""

    def __call__(self, smiles_list):
        if isinstance(smiles_list, str):
            smiles_iter = [smiles_list]
        else:
            smiles_iter = smiles_list

        out = []
        for smiles in smiles_iter:
            graph = smiles2graph(smiles)
            out.append(
                Data(
                    x=torch.from_numpy(graph["node_feat"]).long(),
                    edge_index=torch.from_numpy(graph["edge_index"]).long(),
                    edge_attr=torch.from_numpy(graph["edge_feat"]).long(),
                    num_nodes=int(graph["num_nodes"]),
                )
            )
        return out
@dataclass
class GraphormerBatch:
    x: torch.Tensor
    in_degree: torch.Tensor
    out_degree: torch.Tensor
    attn_bias: torch.Tensor
    spatial_pos: torch.Tensor
    attn_edge_type: torch.Tensor
    edge_input: torch.Tensor

    def __len__(self):
        return self.x.size(0)

    def to(self, *args, **kwargs) -> "GraphormerBatch":
        self.x = self.x.to(*args, **kwargs)
        self.in_degree = self.in_degree.to(*args, **kwargs)
        self.out_degree = self.out_degree.to(*args, **kwargs)
        self.attn_bias = self.attn_bias.to(*args, **kwargs)
        self.spatial_pos = self.spatial_pos.to(*args, **kwargs)
        self.attn_edge_type = self.attn_edge_type.to(*args, **kwargs)
        self.edge_input = self.edge_input.to(*args, **kwargs)
        return self


def _pad_1d_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def _pad_2d_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def _pad_attn_bias_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def _pad_edge_type_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def _pad_spatial_pos_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def _pad_3d_unsqueeze(
    x: torch.Tensor,
    padlen1: int,
    padlen2: int,
    padlen3: int,
) -> torch.Tensor:
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def collate_graphormer_items(
    items: list[tuple[Data, torch.Tensor]],
) -> tuple[GraphormerBatch, torch.Tensor]:
    if len(items) == 0:
        raise ValueError("Cannot collate an empty Graphormer batch")

    graphs, targets = zip(*items)
    packed = []

    for graph in graphs:
        if graph.x is None or graph.edge_attr is None or graph.edge_index is None:
            raise ValueError("Graphormer collate requires x, edge_index, and edge_attr")

        has_structural = bool(getattr(graph, "graphormer_structural_precomputed", False))
        if not has_structural:
            graph = preprocess_graphormer_item(
                graph,
                edge_type="multi_hop",
                multi_hop_max_dist=5,
            )

        if graph.in_degree is None or graph.out_degree is None or graph.attn_bias is None:
            raise ValueError("Missing Graphormer structural tensors in collate")
        if graph.spatial_pos is None or graph.attn_edge_type is None or graph.edge_input is None:
            raise ValueError("Missing Graphormer structural tensors in collate")

        g_x = graph.x
        g_in_degree = graph.in_degree
        g_out_degree = graph.out_degree
        g_attn_bias = graph.attn_bias
        g_spatial_pos = graph.spatial_pos
        g_attn_edge_type = graph.attn_edge_type
        g_edge_input = graph.edge_input
        if g_x is None:
            raise ValueError("Missing node features in Graphormer collate")

        packed.append(
            (
                g_attn_bias.float(),
                g_attn_edge_type.long(),
                g_spatial_pos.long(),
                g_in_degree.long(),
                g_out_degree.long(),
                g_x.long(),
                g_edge_input.long(),
                torch.as_tensor(targets[len(packed)]),
            )
        )

    (
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
    ) = zip(*packed)

    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    x = torch.cat([_pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [_pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [_pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [_pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [_pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([_pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([_pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])

    target_tensor = torch.stack([torch.as_tensor(y) for y in ys]).float()
    return (
        GraphormerBatch(
            x=x,
            in_degree=in_degree,
            out_degree=in_degree, # Graphormer uses in_degree for both in and out degree features - for undirected graphs.
            attn_bias=attn_bias,
            spatial_pos=spatial_pos,
            attn_edge_type=attn_edge_type,
            edge_input=edge_input,
        ),
        target_tensor,
    )


class GraphormerPreprocessTransform:
    """Preprocess node/edge categorical inputs for Graphormer at dataset transform time."""

    supports_progress = True

    def __init__(
        self,
        edge_type: Literal["multi_hop", "single_hop"] = "multi_hop",
        multi_hop_max_dist: int = 5,
        precompute_structural: bool = False,
    ):
        self.edge_type: Literal["multi_hop", "single_hop"] = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.precompute_structural = precompute_structural

    def __call__(self, data: Data) -> Data:
        if self.precompute_structural:
            return preprocess_graphormer_item(
                data,
                edge_type=self.edge_type,
                multi_hop_max_dist=self.multi_hop_max_dist,
            )

        if data.x is None or data.edge_attr is None:
            raise ValueError("Graphormer preprocessing requires x and edge_attr")

        x = data.x
        edge_attr = data.edge_attr
        if len(x.size()) == 1:
            x = x[:, None]
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        data.x = convert_to_single_emb(x)
        data.edge_attr = convert_to_single_emb(edge_attr) + 1

        data.graphormer_preprocessed = True
        return data


def preprocess_graphormer_item(
    data: Data,
    *,
    edge_type: Literal["multi_hop", "single_hop"],
    multi_hop_max_dist: int,
) -> Data:
    if data.x is None or data.edge_attr is None or data.edge_index is None:
        raise ValueError("Graphormer preprocessing requires x, edge_attr, and edge_index")

    edge_attr, edge_index, x = data.edge_attr, data.edge_index, data.x
    if len(x.size()) == 1:
        x = x[:, None]
    n = x.size(0)
    x_single = convert_to_single_emb(x)

    adj = torch.zeros([n, n], dtype=torch.bool, device=edge_index.device)
    adj[edge_index[0, :], edge_index[1, :]] = True
    in_degree = adj.long().sum(dim=1)

    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    edge_attr_single = convert_to_single_emb(edge_attr) + 1
    edge_feat_dim = edge_attr.size(-1)
    attn_edge_type = torch.zeros(
        (n, n, edge_feat_dim),
        dtype=torch.long,
        device=edge_index.device,
    )
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr_single

    attn_bias = torch.zeros([n + 1, n + 1], dtype=torch.float, device=x_single.device)
    dist, next_hop = floyd_warshall(adj)
    spatial_pos = dist
    if edge_type == "multi_hop":
        n_nodes, _, edge_feat_dim = attn_edge_type.shape
        max_dist = max(1, multi_hop_max_dist)
        edge_input = torch.zeros(
            (n_nodes, n_nodes, max_dist, edge_feat_dim),
            dtype=torch.long,
            device=attn_edge_type.device,
        )
        for i in range(n_nodes):
            for j in range(n_nodes):
                d = int(dist[i, j].item())
                if i == j or d <= 0 or d >= 510:
                    continue
                cur = i
                for step in range(min(d, max_dist)):
                    nxt = int(next_hop[cur, j].item())
                    if nxt < 0:
                        break
                    edge_input[i, j, step, :] = attn_edge_type[cur, nxt, :]
                    cur = nxt
    else:
        max_dist = max(1, multi_hop_max_dist)
        edge_input = torch.zeros(
            (n, n, max_dist, edge_feat_dim),
            dtype=torch.long,
            device=x_single.device,
        )

    data.x = x_single
    data.edge_attr = edge_attr_single
    data.in_degree = in_degree
    data.out_degree = in_degree
    data.attn_bias = attn_bias
    data.spatial_pos = spatial_pos
    data.attn_edge_type = attn_edge_type
    data.edge_input = edge_input
    data.graphormer_preprocessed = True
    data.graphormer_structural_precomputed = True
    return data


def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    feature_num = x.size(1) if x.dim() > 1 else 1
    feature_offset = 1 + torch.arange(
        0,
        feature_num * offset,
        offset,
        dtype=torch.long,
        device=x.device,
    )
    return x + feature_offset


def floyd_warshall(adj: torch.BoolTensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute all-pairs shortest-path distances and next hops from adjacency.

    This is the standard Floyd-Warshall dynamic-programming algorithm used by the
    Graphormer preprocessing path (via its original algos/collator utilities) to
    construct shortest-path based structural features.
    """
    n = adj.shape[0]
    dist = torch.full((n, n), 510, dtype=torch.long, device=adj.device)
    next_hop = torch.full((n, n), -1, dtype=torch.long, device=adj.device)

    dist[adj] = 1
    next_hop[adj] = torch.arange(n, device=adj.device).unsqueeze(0).expand(n, n)[adj]
    diag_idx = torch.arange(n, device=adj.device)
    dist[diag_idx, diag_idx] = 0
    next_hop[diag_idx, diag_idx] = diag_idx

    for k in range(n):
        through = dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0)
        update = through < dist
        dist = torch.where(update, through, dist)
        via_k = next_hop[:, k].unsqueeze(1).expand(n, n)
        next_hop = torch.where(update, via_k, next_hop)

    dist = dist.clamp(max=510)
    return dist, next_hop


def build_graphormer_batch(
    data: Data,
    *,
    edge_type: Literal["multi_hop", "single_hop"],
    multi_hop_max_dist: int,
) -> GraphormerBatch:
    data_list = data.to_data_list() if hasattr(data, "to_data_list") else [data]
    per_graph = []
    max_nodes = 0
    max_dist = max(1, multi_hop_max_dist)
    edge_feat_dim = 1
    for graph in data_list:
        if graph.x is None or graph.edge_attr is None or graph.edge_index is None:
            raise ValueError("GraphormerNet requires x, edge_index, and edge_attr in graph batch")
        has_structural = bool(getattr(graph, "graphormer_structural_precomputed", False))
        if not has_structural:
            graph = preprocess_graphormer_item(
                graph,
                edge_type=edge_type,
                multi_hop_max_dist=multi_hop_max_dist,
            )

        x_single = graph.x
        in_degree = graph.in_degree
        out_degree = graph.out_degree
        attn_bias = graph.attn_bias
        dist = graph.spatial_pos
        attn_edge_type = graph.attn_edge_type
        edge_input = graph.edge_input
        if x_single is None or in_degree is None or out_degree is None:
            raise ValueError("Missing Graphormer tensors in graph batch")
        if attn_bias is None or dist is None or attn_edge_type is None or edge_input is None:
            raise ValueError("Missing Graphormer tensors in graph batch")

        n = x_single.shape[0]
        max_nodes = max(max_nodes, n)
        max_dist = max(max_dist, int(edge_input.shape[-2]))
        edge_feat_dim = int(edge_input.shape[-1])

        per_graph.append(
            {
                "x": x_single.long(),
                "in_degree": in_degree.long(),
                "out_degree": out_degree.long(),
                "attn_bias": attn_bias.float(),
                "spatial_pos": dist.long(),
                "attn_edge_type": attn_edge_type.long(),
                "edge_input": edge_input.long(),
            }
        )

    batch_size = len(per_graph)
    x = torch.zeros(
        (batch_size, max_nodes, per_graph[0]["x"].shape[-1]),
        dtype=torch.long,
        device=per_graph[0]["x"].device,
    )
    in_degree = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=x.device)
    out_degree = torch.zeros((batch_size, max_nodes), dtype=torch.long, device=x.device)
    attn_bias = torch.zeros((batch_size, max_nodes + 1, max_nodes + 1), dtype=torch.float, device=x.device)
    spatial_pos = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.long, device=x.device)
    attn_edge_type = torch.zeros((batch_size, max_nodes, max_nodes, edge_feat_dim), dtype=torch.long, device=x.device)
    edge_input = torch.zeros((batch_size, max_nodes, max_nodes, max_dist, edge_feat_dim), dtype=torch.long, device=x.device)

    for idx, graph_data in enumerate(per_graph):
        n = graph_data["x"].shape[0]
        x[idx, :n, :] = graph_data["x"]
        in_degree[idx, :n] = graph_data["in_degree"]
        out_degree[idx, :n] = graph_data["out_degree"]
        attn_bias[idx, : n + 1, : n + 1] = graph_data["attn_bias"]
        spatial_pos[idx, :n, :n] = graph_data["spatial_pos"]
        attn_edge_type[idx, :n, :n, :] = graph_data["attn_edge_type"]
        edge_input[idx, :n, :n, :, :] = graph_data["edge_input"]

    return GraphormerBatch(
        x=x,
        in_degree=in_degree,
        out_degree=out_degree,
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        attn_edge_type=attn_edge_type,
        edge_input=edge_input,
    )
