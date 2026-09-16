import torch
import torch.nn as nn
from diffpred.models.networks.components.blocks import BaseBlock
from diffpred.models.protocols import DiffNetProtocol
from diffpred.models.networks.factories.nn_factory import create_feature_network
from diffpred.constants.activations import ACTIVATIONS
from diffpred.models.networks.components.embeddings import (
    TargetTimeEmbedding,
    FourierCondEmbedding,
)


class BaselineDiffusionNet(nn.Module, DiffNetProtocol):
    def __init__(self, config, classifier: bool = False):
        super().__init__()
        self.classifier = classifier
        self.target_dim = config.output_dim
        self.self_conditioning = config.self_conditioning
        self.time_embedding = FourierCondEmbedding(
            embedding_dim=config.embedding_dim, feature_dim=config.hidden_dim
        )
        self.embedding = TargetTimeEmbedding(
            embedding_dim=config.embedding_dim,
            feature_dim=config.hidden_dim,
            target_dim=config.output_dim,
        )
        self.target_encoding = nn.Linear(config.output_dim, config.hidden_dim)
        if self.self_conditioning:
            self.self_cond_fusion = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.feature_net = create_feature_network(conf=config)
        self.num_blocks = config.num_hiddens
        self.blocks = nn.ModuleList()
        input_dim = config.feature_hiddens[-1]
        for i in range(self.num_blocks):
            self.blocks.append(
                BaseBlock(
                    embed_dim=config.embedding_dim,
                    input_dim=input_dim,
                    out_dim=(config.hidden_dim // 2**i),
                    activation=ACTIVATIONS[config.activation](),
                    cond_conf=config.cond_config,
                )
            )
            input_dim = config.hidden_dim // 2**i
        self.out_layer = nn.Linear(input_dim, config.output_dim)

    def forward(self, features, target, time, self_cond=None):
        time_emb = self.time_embedding(time)
        target_emb = self._target_encoding(target)

        if self.self_conditioning:
            if self_cond is None:
                self_cond = torch.zeros_like(target)
            self_cond_emb = self._target_encoding(self_cond)
            target_emb = torch.cat([target_emb, self_cond_emb], dim=-1)
            target_emb = self.self_cond_fusion(target_emb)

        embedding = self.embedding(target_emb, time_emb)
        features = self.feature_net(features)

        for block in self.blocks:
            features = block(
                x=features, embedding=embedding, target_emb=target_emb, t=time_emb
            )

        out = self.out_layer(features)

        return out

    def _target_encoding(self, target):
        # nan_imputation shape: (batch, target_dim)
        target = torch.where(torch.isnan(target), torch.zeros_like(target), target)
        target = self.target_encoding(target)
        return target
