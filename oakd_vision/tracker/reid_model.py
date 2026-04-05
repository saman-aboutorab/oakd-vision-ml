"""ReIDNet — ResNet18 backbone with a custom embedding head.

Architecture:
    ResNet18 (ImageNet pretrained, all layers fine-tuned)
    → Global Average Pooling  (512-dim)
    → FC 512 → 256 + BatchNorm + ReLU + Dropout
    → FC 256 → 128
    → L2 normalize
    → 128-dim unit-sphere embedding

L2 normalization is critical: it forces all embeddings onto a unit hypersphere,
making cosine distance equivalent to L2 distance. This simplifies the tracker's
similarity computation and stabilizes triplet loss training.
"""

import torch
import torch.nn as nn
from torchvision import models


class ReIDNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, dropout: float = 0.4):
        super().__init__()

        # Pretrained ResNet18 — remove the final classification layer
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → [B, 512, 1, 1]

        # Embedding head
        self.head = nn.Sequential(
            nn.Flatten(),                          # → [B, 512]
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] normalized image tensor

        Returns:
            [B, embedding_dim] L2-normalized embedding vectors
        """
        features = self.backbone(x)        # [B, 512, 1, 1]
        embeddings = self.head(features)   # [B, embedding_dim]
        return nn.functional.normalize(embeddings, p=2, dim=1)  # unit sphere
