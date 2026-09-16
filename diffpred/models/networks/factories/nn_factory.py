from diffpred.models.networks.graph.graph_nets import (
    GCNNet,
    PNANet,
    GINNet,
    GINENet,
    GPSNet,
    ChempropMPNNet,
    GraphormerNet,
)
from diffpred.models.protocols import ArchitectureConfig
from diffpred.models.networks.nn_arch import MLP
from diffpred.constants.activations import ACTIVATIONS
import torch.nn as nn


def create_feature_network(conf: ArchitectureConfig) -> nn.Module:
    if conf.net_type == "mlp":
        return MLP(
            input_dim=conf.input_dim,
            hiddens=conf.feature_hiddens,
            activation=ACTIVATIONS[conf.activation],
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
            dropout=conf.dropout,
            mc_dropout=conf.mc_dropout,
        )
    elif conf.net_type == "gcn":
        return GCNNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            activation=conf.activation,
            dropout=conf.dropout,
            batch_norm=conf.batch_norm,
            pooling=conf.pooling,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
        )
    elif conf.net_type == "gine":
        return GINENet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            activation=conf.activation,
            edge_dim=conf.edge_dim,
            dropout=conf.dropout,
            batch_norm=conf.batch_norm,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
        )
    elif conf.net_type == "gin":
        return GINNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            activation=conf.activation,
            dropout=conf.dropout,
            batch_norm=conf.batch_norm,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
        )
    elif conf.net_type == "pna":
        return PNANet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            activation=conf.activation,
            degree=conf.degree,
            dropout=conf.dropout,
            batch_norm=conf.batch_norm,
            pooling=conf.pooling,
        )
    elif conf.net_type == "gps":
        return GPSNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            edge_dim=conf.edge_dim,
            norm=conf.norm,
            pe_dim=conf.pe_dim,
            num_rw_steps=conf.num_rw_steps,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
            dropout=conf.dropout,
        )
    elif conf.net_type == "chemprop":
        return ChempropMPNNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            edge_dim=conf.edge_dim,
            activation=conf.activation,
            dropout=conf.dropout,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
            message_passing=conf.chemprop_message_passing,
            aggregation=conf.chemprop_aggregation,
            depth=conf.chemprop_depth,
        )
    elif conf.net_type == "graphormer":
        return GraphormerNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            edge_dim=conf.edge_dim,
            activation=conf.activation,
            dropout=conf.dropout,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
            num_layers=conf.graphormer_num_layers,
            num_heads=conf.graphormer_num_heads,
            ffn_embedding_dim=conf.graphormer_ffn_embed_dim,
            edge_type=conf.graphormer_edge_type,
            multi_hop_max_dist=conf.graphormer_multi_hop_max_dist,
            num_spatial=conf.graphormer_num_spatial,
            num_in_degree=conf.graphormer_num_in_degree,
            num_out_degree=conf.graphormer_num_out_degree,
            num_edge_dis=conf.graphormer_num_edge_dis,
            attention_dropout=conf.graphormer_attention_dropout,
            activation_dropout=conf.graphormer_activation_dropout,
            pre_layernorm=conf.graphormer_pre_layernorm,
        )
    else:
        raise ValueError


def baseline_net_factory(conf: ArchitectureConfig, use_mve: bool = False) -> nn.Module:
    output_dim = (
        conf.output_dim if not use_mve else conf.output_dim * 2
    )  # for mean and variance of each output
    if conf.net_type == "mlp":
        return MLP(
            input_dim=conf.input_dim,
            hiddens=conf.feature_hiddens,
            activation=ACTIVATIONS[conf.activation],
            output_dim=output_dim,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
            dropout=conf.dropout,
            mc_dropout=conf.mc_dropout,
        )
    elif conf.net_type == "gcn":
        return GCNNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            activation=conf.activation,
            dropout=(conf.mc_dropout if conf.mc_dropout > 0 else conf.dropout),
            batch_norm=conf.batch_norm,
            output_dim=output_dim,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
        )
    elif conf.net_type == "gin":
        return GINNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            activation=conf.activation,
            dropout=(conf.mc_dropout if conf.mc_dropout > 0 else conf.dropout),
            batch_norm=conf.batch_norm,
            output_dim=output_dim,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
        )

    elif conf.net_type == "gine":
        return GINENet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            activation=conf.activation,
            edge_dim=conf.edge_dim,
            dropout=(conf.mc_dropout if conf.mc_dropout > 0 else conf.dropout),
            batch_norm=conf.batch_norm,
            output_dim=output_dim,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
        )

    elif conf.net_type == "pna":
        return PNANet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            activation=conf.activation,
            degree=conf.degree,
            dropout=(conf.mc_dropout if conf.mc_dropout > 0 else conf.dropout),
            batch_norm=conf.batch_norm,
            output_dim=output_dim,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
        )
    elif conf.net_type == "gps":
        return GPSNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            output_dim=output_dim,
            dropout=(conf.mc_dropout if conf.mc_dropout > 0 else conf.dropout),
            edge_dim=conf.edge_dim,
            norm=conf.norm,
            pe_dim=conf.pe_dim,
            num_rw_steps=conf.num_rw_steps,
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
        )
    elif conf.net_type == "chemprop":
        return ChempropMPNNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            edge_dim=conf.edge_dim,
            output_dim=output_dim,
            activation=conf.activation,
            dropout=(conf.mc_dropout if conf.mc_dropout > 0 else conf.dropout),
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
            message_passing=conf.chemprop_message_passing,
            aggregation=conf.chemprop_aggregation,
            depth=conf.chemprop_depth,
        )
    elif conf.net_type == "graphormer":
        return GraphormerNet(
            in_channels=conf.input_dim,
            hidden_channels=conf.feature_hiddens,
            edge_dim=conf.edge_dim,
            output_dim=output_dim,
            activation=conf.activation,
            dropout=(conf.mc_dropout if conf.mc_dropout > 0 else conf.dropout),
            num_output_layers=conf.num_hiddens,
            output_mlp_type=conf.output_mlp_type,
            num_layers=conf.graphormer_num_layers,
            num_heads=conf.graphormer_num_heads,
            ffn_embedding_dim=conf.graphormer_ffn_embed_dim,
            edge_type=conf.graphormer_edge_type,
            multi_hop_max_dist=conf.graphormer_multi_hop_max_dist,
            num_spatial=conf.graphormer_num_spatial,
            num_in_degree=conf.graphormer_num_in_degree,
            num_out_degree=conf.graphormer_num_out_degree,
            num_edge_dis=conf.graphormer_num_edge_dis,
            attention_dropout=conf.graphormer_attention_dropout,
            activation_dropout=conf.graphormer_activation_dropout,
            pre_layernorm=conf.graphormer_pre_layernorm,
        )
    else:
        raise ValueError

