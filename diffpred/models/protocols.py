from typing import Literal, Protocol, List
from dataclasses import dataclass, field
import torch
@dataclass
class ConditioningConfig:
    """
    Config which determines the type of conditioning used in the diffusion model.
    """

    fusion_gating: bool = False  # whether to use gating mechanism for fusion-based conditioning at each block
    fusion_loc: Literal["pre-branch", "post-branch"] = (
        "post-branch"  # whether to apply fusion at each block or layer. Only applies if fusion_type is not dual-stage.
    )
    fusion_type: Literal["none", "concat", "add", "gated"] = (
        "none"  # type of fusion used for conditioning, dual-stage refers to first concatenating and then applying a scaling layer from timestep. use_timestep_scaling does not apply in this case (alway applies)
    )
    use_timestep_gating: bool = False  # whether to use timestep conditioning in the form of scaling/shift on conditioning target features


@dataclass
class ArchitectureConfig:
    """
    Config which determines the type and architecture used to process the inputs in the diffusion model.
    """

    input_dim: int
    output_dim: int
    edge_dim: int | None = None  # Optional edge dimension for graph models
    feature_hiddens: List[int] = field(
        default_factory=lambda: [256, 128]
    )  # separate hidden layers for an initial feature extractor (for vector representation of features)
    arch_type: Literal["baseline"] | None = None
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_hiddens: int = 3
    activation: str = "silu"
    mc_dropout: float = 0.0
    dropout: float = 0.0
    batch_norm: bool = False
    degree: torch.Tensor | None = None
    pooling: str = "sum"
    net_type: Literal["mlp", "gcn", "pna", "gin", "gine", "gps", "chemprop", "graphormer"] = "mlp"
    chemprop_message_passing: Literal["bond", "atom"] = "bond"
    chemprop_aggregation: Literal["mean", "sum", "norm"] = "norm"
    chemprop_depth: int = 3
    graphormer_num_layers: int = 12
    graphormer_num_heads: int = 8
    graphormer_ffn_embed_dim: int = 128
    graphormer_edge_type: Literal["multi_hop", "single_hop"] = "multi_hop"
    graphormer_multi_hop_max_dist: int = 5
    graphormer_num_spatial: int = 512
    graphormer_num_in_degree: int = 512
    graphormer_num_out_degree: int = 512
    graphormer_num_edge_dis: int = 128
    graphormer_attention_dropout: float = 0.1
    graphormer_activation_dropout: float = 0.1
    graphormer_pre_layernorm: bool = False
    use_rwse: bool = False
    use_lap_pe: bool = False
    layer_output_norm: bool = False
    norm: str = "batch_norm"
    pe_dim: int = 6
    num_rw_steps: int = 10
    num_output_layers: int = 3
    output_mlp_type: Literal["linear", "residual"] = "linear"
    cond_config: ConditioningConfig = field(default_factory=ConditioningConfig)
    self_conditioning: bool = False



class DiffNetProtocol(Protocol):
    def forward(self, x, z, t): ...

    def __call__(self, inputs):
        return self.forward(inputs)
