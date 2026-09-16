# Core Graphormer components adapted from the official implementation: https://github.com/microsoft/Graphormer/tree/main

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn as nn
from diffpred.constants.activations import ACTIVATIONS



class GraphormerBatchLike(Protocol):
    x: torch.Tensor
    in_degree: torch.Tensor
    out_degree: torch.Tensor
    attn_bias: torch.Tensor
    spatial_pos: torch.Tensor
    attn_edge_type: torch.Tensor
    edge_input: torch.Tensor


def init_graphormer_params(module: nn.Module, n_layers: int) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class GraphNodeFeature(nn.Module):
    # Adapted from the official Graphormer implementation
    def __init__(
        self,
        num_heads: int,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)

    def forward(self, batched_data: GraphormerBatchLike) -> torch.Tensor:
        x = batched_data.x
        in_degree = batched_data.in_degree
        out_degree = batched_data.out_degree
        n_graph = x.size(0)

        node_feature = self.atom_encoder(x).sum(dim=-2)
        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        return torch.cat([graph_token_feature, node_feature], dim=1)


class GraphAttnBias(nn.Module):
    # Adapted from Graphormer attention-bias path; quantization/fairseq extras removed.
    def __init__(
        self,
        num_heads: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_spatial = num_spatial
        self.num_edges = num_edges
        self.multi_hop_max_dist = multi_hop_max_dist
        self.edge_type = edge_type

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        if edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        else:
            self.edge_dis_encoder = None

    def forward(self, batched_data: GraphormerBatchLike) -> torch.Tensor:
        attn_bias = batched_data.attn_bias
        spatial_pos = batched_data.spatial_pos
        x = batched_data.x
        attn_edge_type = batched_data.attn_edge_type
        edge_input = batched_data.edge_input

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        if self.edge_type == "multi_hop":
            if self.edge_dis_encoder is None:
                raise ValueError("edge_dis_encoder is required for multi_hop edge_type")
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]

            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / spatial_pos_.float().unsqueeze(-1)).permute(
                0, 3, 1, 2
            )
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)
        return graph_attn_bias


class MultiheadAttention(nn.Module):
    # This follows the core Graphormer/fairseq attention math using plain PyTorch modules.
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        attn_bias: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        tgt_len, bsz, embed_dim = query.size()

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(query)
        v = self.v_proj(query)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights + attn_bias.view(bsz * self.num_heads, tgt_len, tgt_len)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, tgt_len)

        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        return self.out_proj(attn)


class GraphormerGraphEncoderLayer(nn.Module):
    # Encoder block mirrors the reference sequence of pre/post-layernorm and FFN.
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: str,
        pre_layernorm: bool,
    ):
        super().__init__()
        self.pre_layernorm = pre_layernorm

        self.dropout_module = nn.Dropout(dropout)
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.activation_fn = ACTIVATIONS[activation_fn]()

        self.self_attn = MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(query=x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x


class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: str,
        pre_layernorm: bool,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
        )
        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
        )
        self.layers = nn.ModuleList(
            [
                GraphormerGraphEncoderLayer(
                    embedding_dim=embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.apply(lambda module: init_graphormer_params(module, num_encoder_layers))

    def forward(self, batched_data: GraphormerBatchLike) -> torch.Tensor:
        data_x = batched_data.x
        n_graph = data_x.size(0)
        padding_mask = data_x[:, :, 0].eq(0)
        padding_mask_cls = torch.zeros(
            n_graph,
            1,
            device=padding_mask.device,
            dtype=padding_mask.dtype,
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        x = self.graph_node_feature(batched_data)
        attn_bias = self.graph_attn_bias(batched_data)

        x = self.dropout(x)
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, key_padding_mask=padding_mask)

        return x[0, :, :]
