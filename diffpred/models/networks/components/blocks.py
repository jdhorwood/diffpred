import torch
import torch.nn as nn
from typing import Any, cast
from omegaconf import OmegaConf, open_dict
from diffpred.models.protocols import ConditioningConfig


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU):
        super().__init__()
        self.block_net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            activation(),
            torch.nn.Linear(out_dim, out_dim),
        )
        if in_dim != out_dim:
            self.residual_proj = torch.nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            return self.residual_proj(x) + self.block_net(x)
        return x + self.block_net(x)


class BaseBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        input_dim,
        out_dim,
        activation,
        conditioning: bool = True,
        cond_conf: ConditioningConfig | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.conditioning = conditioning
        self.activation = activation
        self.skip_connection = (
            nn.Linear(input_dim, out_dim) if input_dim != out_dim else nn.Identity()
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, out_dim),
            self.activation,
            nn.Linear(out_dim, out_dim),
            (
                nn.LayerNorm(out_dim, elementwise_affine=False, bias=False)
                if conditioning
                else nn.Identity()  # affine=False because FiLM is used after
            ),  # no extra norm needed if conditioning is not used
        )
        self.linear_out = nn.Linear(out_dim, out_dim)
        self.cond_conf = cond_conf if cond_conf is not None else ConditioningConfig()
        self._init_cond_layers()

    def _init_cond_layers(self):
        if self.conditioning:
            self.embed_linear = nn.Linear(self.embed_dim, self.out_dim * 2)

        # Legacy naming support: migrate use_timestep_film -> use_timestep_gating in-place.
        if hasattr(self.cond_conf, "use_timestep_film"):
            legacy_val = bool(getattr(self.cond_conf, "use_timestep_film"))
            if OmegaConf.is_config(self.cond_conf):
                with open_dict(cast(Any, self.cond_conf)):
                    setattr(self.cond_conf, "use_timestep_gating", legacy_val)
            else:
                setattr(self.cond_conf, "use_timestep_gating", legacy_val)

        if (
            self.cond_conf.use_timestep_gating or self.cond_conf.fusion_type == "gated"
        ):  # for gated fusion, always use timestep gating
            self._timestep_gating_lin = nn.Linear(self.embed_dim, 1)

        merge_dim = (
            self.out_dim
            if self.cond_conf.fusion_loc == "post-branch"
            else self.input_dim
        )
        if self.cond_conf.fusion_type == "concat":
            self.aggr_proj = nn.Linear(merge_dim + self.embed_dim, merge_dim)
        elif self.cond_conf.fusion_type in ("add", "gated"):
            self.aggr_proj = (
                nn.Linear(self.embed_dim, merge_dim)
                if self.embed_dim != merge_dim
                else nn.Identity()
            )

    def forward(
        self,
        x,
        embedding: torch.Tensor | None,
        target_emb: torch.Tensor | None,
        t: torch.Tensor | None = None,
    ):
        self._verify_embedding(embedding)
        self._verify_embedding(target_emb)
        self._verify_embedding(t)
        residual = self._block_forward(x, embedding=embedding)
        if self.conditioning and self.cond_conf.fusion_loc == "pre-branch":
            x = self._target_fusion(target_emb, x, t)
        out = self._merge_residual(residual, x)
        if self.conditioning and self.cond_conf.fusion_loc == "post-branch":
            out = self._target_fusion(target_emb, out, t)
        return out

    def _verify_embedding(self, embedding: torch.Tensor | None):
        if embedding is None and self.conditioning:
            raise ValueError(
                "Embedding is required for conditioning (set to True), but it has value of None."
            )
        if embedding is not None and not self.conditioning:
            raise ValueError(
                "Conditioning was disabled, but an embedding vector was provided."
            )
        return

    def _block_forward(self, input, embedding: torch.Tensor | None):
        self._verify_embedding(embedding)
        if self.conditioning:  # this is for the standard conditioning within the block
            scale_shift = self.embed_linear(embedding)
            h = self.mlp(input)
            h = self._scale_shift_cond(h, scale_shift)
            h = self.activation(h)
        else:
            h = self.activation(self.mlp(input))
        h = self.linear_out(h)
        return h

    def _merge_residual(self, residual, main_path_input):
        return residual + self.skip_connection(main_path_input)

    def _scale_shift_cond(self, h, embedding):
        # scale and shift conditioning - akin to adaptive normalization/FiLM
        scale, shift = embedding.chunk(2, dim=-1)
        h = h * scale + shift
        return h

    def _target_fusion(
        self,
        target_emb: torch.FloatTensor,
        h: torch.FloatTensor,
        t: torch.FloatTensor | None = None,
    ):
        """
        Implements a fusion-based conditioning mechanism for the main flow path of the network.
        The target embedding is fused with the main flow path using the specified fusion type.
        Args:
            target_emb (torch.FloatTensor): The target embedding to be fused.
            h (torch.FloatTensor): The main flow feature tensor.
            t (torch.FloatTensor | None): Optional time tensor for conditioning.
        """
        if not self.conditioning:
            raise ValueError(
                "Target fusion is not possible when conditioning is disabled."
            )
        if self.cond_conf.fusion_type == "concat":
            if h.ndim == target_emb.ndim - 1:
                # Expand h to match the target embedding dimension
                h = h.unsqueeze(0).expand(target_emb.shape[0], -1, -1)
            if self.cond_conf.use_timestep_gating:
                z = self._timestep_gating(t)
                target_emb = z * target_emb
            h = torch.cat([h, target_emb], dim=-1)
            h = self.aggr_proj(h)
        elif self.cond_conf.fusion_type == "add":
            if self.cond_conf.use_timestep_gating:
                z = self._timestep_gating(t)
                target_emb = z * target_emb
            target_emb = self.aggr_proj(target_emb)
            h = h + target_emb
        elif self.cond_conf.fusion_type == "gated":
            z = self._timestep_gating(t)
            target_emb = self.aggr_proj(target_emb)
            h = (1 - z) * h + (z * target_emb)
        elif self.cond_conf.fusion_type == "none":
            pass
        else:
            raise ValueError(
                f"Unknown fusion type: {self.cond_conf.fusion_type}. "
                "Valid options are: 'concat', 'add', or 'none'."
            )
        return h

    def _timestep_gating(self, time_emb: torch.FloatTensor):
        return torch.sigmoid(self._timestep_gating_lin(time_emb))
