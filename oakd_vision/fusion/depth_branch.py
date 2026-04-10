"""Depth branch for P3 TraversabilityNet.

Lightweight custom CNN trained from scratch on normalised depth patches.
No pretrained weights — depth maps look nothing like ImageNet images
(single channel, values encode distance not colour), so pretraining
would not help and could hurt.

Design choices:
  - 4 conv blocks with BatchNorm + ReLU + MaxPool (standard feature extractor)
  - Starts narrow (16 filters) and doubles each block → 128 at the end
  - Keeps it lightweight: ~200K parameters vs ResNet18's 11M
  - Output matches RGB branch: [B, embedding_dim] for easy fusion

Output: [B, embedding_dim] feature vector per patch.
"""

import torch
import torch.nn as nn


class DepthBranch(nn.Module):
    """4-block CNN for single-channel depth patches → 256-dim feature vector.

    Architecture:
        Input [B, 1, H, W]
        → Conv block 1: 1→16,  3×3, BN, ReLU, MaxPool  → H/2
        → Conv block 2: 16→32, 3×3, BN, ReLU, MaxPool  → H/4
        → Conv block 3: 32→64, 3×3, BN, ReLU, MaxPool  → H/8
        → Conv block 4: 64→128,3×3, BN, ReLU, MaxPool  → H/16
        → Global Average Pooling → [B, 128]
        → FC 128→256 + BN + ReLU + Dropout
        → [B, embedding_dim]

    Args:
        embedding_dim: output feature size (default 256, matches RGB branch)
        dropout:       dropout rate before output (default 0.3)
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1,   16),
            conv_block(16,  32),
            conv_block(32,  64),
            conv_block(64, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)   # → [B, 128, 1, 1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] normalised depth patches (values in [0, 1])

        Returns:
            [B, embedding_dim] feature vectors
        """
        x = self.features(x)   # [B, 128, H/16, W/16]
        x = self.pool(x)       # [B, 128, 1, 1]
        return self.head(x)    # [B, embedding_dim]
