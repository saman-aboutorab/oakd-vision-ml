"""RGB branch for P3 TraversabilityNet.

Uses a pretrained ResNet18 backbone. First two layer groups are frozen
(low-level edges/textures are universal — no need to retrain them).
Layers 3 and 4 are fine-tuned to learn traversability-relevant texture
and semantic features (carpet vs tile vs obstacle surface).

Output: [B, embedding_dim] feature vector per patch.
"""

import torch
import torch.nn as nn
from torchvision import models


class RGBBranch(nn.Module):
    """ResNet18 backbone → 256-dim feature vector.

    Architecture:
        ResNet18 (pretrained ImageNet)
        → freeze layer1 + layer2 (low-level features)
        → fine-tune layer3 + layer4 (semantic features)
        → Global Average Pooling → [B, 512]
        → FC 512→256 + BN + ReLU + Dropout
        → [B, embedding_dim]

    Args:
        embedding_dim: output feature size (default 256, matches depth branch)
        dropout:       dropout rate before output (default 0.3)
        freeze_layers: number of ResNet layer groups to freeze 0–4 (default 2)
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.3,
                 freeze_layers: int = 2):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Keep everything except the final FC classifier
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool   = backbone.avgpool   # → [B, 512, 1, 1]

        # Freeze early layers
        layers_to_freeze = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers_to_freeze[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False

        # Projection head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] ImageNet-normalized RGB patches

        Returns:
            [B, embedding_dim] feature vectors
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)      # [B, 512, 1, 1]
        return self.head(x)   # [B, embedding_dim]
