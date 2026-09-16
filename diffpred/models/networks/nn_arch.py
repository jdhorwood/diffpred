from typing import Literal
import torch.nn as nn
from diffpred.constants.activations import ACTIVATIONS
from diffpred.models.networks.components.blocks import ResidualBlock
from diffpred.models.networks.components.blocks import BaseBlock


class ResidualOutNet(nn.Module):
    """
    Residual output network for baseline models which mirrors diffusion architecture.

    """

    def __init__(
        self, num_hiddens, input_dim, hidden_dim, out_dim, activation, mc_dropout=0.0,
    ):
        super().__init__()
        self.num_blocks = num_hiddens
        self.blocks = nn.ModuleList()
        self.mc_dropouts = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(
                BaseBlock(
                    embed_dim=None,
                    input_dim=input_dim,
                    out_dim=(hidden_dim // 2**i),
                    activation=activation(),
                    conditioning=False,
                )
            )
            self.mc_dropouts.append(nn.Dropout(mc_dropout) if mc_dropout > 0 else nn.Identity())
            input_dim = hidden_dim // 2**i
        self.out_layer = nn.Linear(input_dim, out_dim)

    def forward(self, features):
        for block, mc_dropout in zip(self.blocks, self.mc_dropouts):
            features = block(x=features, embedding=None, target_emb=None, t=None)
            features = mc_dropout(features)
        out = self.out_layer(features)
        return out


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hiddens=[64],
        activation=nn.ReLU,
        residual=False,
        dropout=0.0,
        mc_dropout=0.0,
        num_output_layers: int = 3,
        output_hidden: int | None = None,
        output_mlp_type: Literal["linear", "residual"] = "linear",
    ):
        super().__init__()
        if (dropout > 0.0 and mc_dropout > 0.0):
            raise RuntimeError("Must use either standard dropout or mc_dropout, not both.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hiddens
        self.out_mlp = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.residual = residual
        self.mc_dropout = mc_dropout
        self.dropout = dropout
        self.num_output_layers = num_output_layers
        self.output_mlp_type = output_mlp_type
        self.output_hidden = (
            output_hidden
            if output_hidden is not None
            else self._get_output_mlp_in_dim()
        )

        # Build network layers
        self._build_network()

    def _build_network(self):
        current_dim = self.input_dim
        # Hidden layers
        for hidden in self.hiddens:
            if self.residual:
                self.layers.append(
                    ResidualBlock(
                        in_dim=current_dim, out_dim=hidden, activation=self.activation
                    )
                )
                self.layers.append(self.activation())
            else:
                self.layers.append(nn.Linear(current_dim, hidden))
                self.layers.append(self.activation())
            if self.mc_dropout > 0:
                self.layers.append(nn.Dropout(self.mc_dropout))
            current_dim = hidden

        if self.dropout > 0 and len(self.hiddens) > 0:
            self.layers.append(nn.Dropout(self.dropout))

        self.layers = nn.Sequential(*self.layers)
        # Output layers if specified
        if self.output_dim is not None:
            in_dim = self._get_output_mlp_in_dim()
            self.out_mlp = self._build_output_mlp(in_dim)

    def _get_output_mlp_in_dim(self):
        return self.hiddens[-1] if len(self.hiddens) > 0 else self.input_dim

    def _build_output_mlp(self, in_dim: int) -> nn.Sequential:
        if self.output_mlp_type == "linear":
            for i in range(self.num_output_layers):
                self.out_mlp.append(nn.Linear(in_dim, self.output_hidden // 2**i))
                self.out_mlp.append(self.activation())
                # Dropout if specified
                if self.mc_dropout > 0:
                    self.out_mlp.append(nn.Dropout(self.mc_dropout))
                in_dim = self.output_hidden // 2**i
            self.out_mlp.append(nn.Linear(in_dim, self.output_dim))
            return nn.Sequential(*self.out_mlp)
        elif self.output_mlp_type == "residual":
            return ResidualOutNet(
                num_hiddens=self.num_output_layers,
                input_dim=in_dim,
                hidden_dim=self.output_hidden,
                out_dim=self.output_dim,
                activation=self.activation,
                mc_dropout=self.mc_dropout,
            )
        else:
            raise ValueError(
                f"Invalid output_mlp_type: {self.output_mlp_type}. "
                "Choose either 'linear' or 'residual'."
            )

    def forward(self, x):
        out = self.layers(x) if len(self.hiddens) > 0 else x
        if self.output_dim is not None:
            out = self.out_mlp(out)
        return out

