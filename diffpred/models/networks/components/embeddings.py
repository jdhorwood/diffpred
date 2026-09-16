from typing import Literal
import torch
import torch.nn as nn
from diffpred.models.encodings import FourierEmbedding

class FourierCondEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feature_dim: int,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.cond_embedding = FourierEmbedding(self.embedding_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, cond: torch.FloatTensor):
        emb = self.cond_mlp(self.cond_embedding(cond.squeeze()))
        return emb


class TargetTimeEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feature_dim: int,
        target_dim: int,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.embedding_dim = embedding_dim
        self.activation = nn.functional.silu

    def forward(
        self, target_emb: torch.FloatTensor, time_emb: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        embedding = time_emb + target_emb
        embedding = self.activation(embedding)
        return embedding