"""TraversabilityNet — dual-branch fusion model for P3.

Combines RGB and depth features using one of three fusion strategies:

  concat    — concatenate [rgb | depth] → 512-dim → FC head
              Baseline. Simple, always works, but treats both modalities equally.

  attention — a small network predicts a scalar weight per modality.
              In dark rooms it up-weights depth; on reflective surfaces it
              up-weights RGB. One weight for the whole patch.

  gated     — like attention but element-wise: each of the 256 feature
              dimensions independently decides how much RGB vs depth to use.
              Most expressive, most parameters.

Select strategy via fusion_strategy argument or fusion_config.yaml.

Usage:
    model = TraversabilityNet(fusion_strategy="concat")
    logits = model(rgb_patches, depth_patches)   # [B, 4]
    probs  = logits.softmax(dim=1)               # [B, 4]
"""

import torch
import torch.nn as nn

from oakd_vision.fusion.rgb_branch   import RGBBranch
from oakd_vision.fusion.depth_branch import DepthBranch

FUSION_STRATEGIES = ("concat", "attention", "gated")
NUM_CLASSES = 4   # free / caution / obstacle / unknown


class TraversabilityNet(nn.Module):
    """Dual-branch traversability classifier.

    Args:
        embedding_dim:    feature size output by each branch (default 256)
        num_classes:      number of output classes (default 4)
        fusion_strategy:  "concat" | "attention" | "gated"
        dropout:          dropout in branch heads and classifier (default 0.3)
        freeze_rgb_layers: number of ResNet layer groups to freeze (default 2)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = NUM_CLASSES,
        fusion_strategy: str = "concat",
        dropout: float = 0.3,
        freeze_rgb_layers: int = 2,
    ):
        super().__init__()
        assert fusion_strategy in FUSION_STRATEGIES, \
            f"fusion_strategy must be one of {FUSION_STRATEGIES}"

        self.fusion_strategy = fusion_strategy
        self.embedding_dim   = embedding_dim

        # --- Branches ---
        self.rgb_branch   = RGBBranch(embedding_dim, dropout, freeze_rgb_layers)
        self.depth_branch = DepthBranch(embedding_dim, dropout)

        # --- Fusion + classifier ---
        if fusion_strategy == "concat":
            # Simply stack both feature vectors → 2×embedding_dim → classify
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, num_classes),
            )

        elif fusion_strategy == "attention":
            # Predict a single scalar weight w ∈ [0,1] for each modality
            # fused = w * rgb + (1-w) * depth
            # The gate network sees both features concatenated
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid(),           # w ∈ [0, 1]
            )
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.BatchNorm1d(embedding_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, num_classes),
            )

        elif fusion_strategy == "gated":
            # Element-wise gate: each feature dim independently weights RGB vs depth
            # gate vector g ∈ [0,1]^embedding_dim
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Sigmoid(),           # g ∈ [0,1]^D element-wise
            )
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.BatchNorm1d(embedding_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, num_classes),
            )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rgb:   [B, 3, H, W] ImageNet-normalized RGB patches
            depth: [B, 1, H, W] normalised depth patches [0, 1]

        Returns:
            logits: [B, num_classes]  (apply softmax for probabilities)
        """
        rgb_feat   = self.rgb_branch(rgb)      # [B, D]
        depth_feat = self.depth_branch(depth)  # [B, D]

        if self.fusion_strategy == "concat":
            fused = torch.cat([rgb_feat, depth_feat], dim=1)   # [B, 2D]

        elif self.fusion_strategy == "attention":
            combined = torch.cat([rgb_feat, depth_feat], dim=1)
            w = self.gate(combined)                             # [B, 1]
            fused = w * rgb_feat + (1.0 - w) * depth_feat      # [B, D]

        elif self.fusion_strategy == "gated":
            combined = torch.cat([rgb_feat, depth_feat], dim=1)
            g = self.gate(combined)                             # [B, D]
            fused = g * rgb_feat + (1.0 - g) * depth_feat      # [B, D]

        return self.classifier(fused)   # [B, num_classes]

    def param_count(self) -> dict:
        """Return trainable and frozen parameter counts per component."""
        def count(module):
            total     = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return trainable, total

        rgb_t, rgb_all   = count(self.rgb_branch)
        dep_t, dep_all   = count(self.depth_branch)
        cls_t, cls_all   = count(self.classifier)
        gate_t, gate_all = count(self.gate) if hasattr(self, "gate") else (0, 0)

        return {
            "rgb_branch":   {"trainable": rgb_t,   "total": rgb_all},
            "depth_branch": {"trainable": dep_t,   "total": dep_all},
            "classifier":   {"trainable": cls_t,   "total": cls_all},
            "gate":         {"trainable": gate_t,  "total": gate_all},
            "model_total":  {"trainable": rgb_t + dep_t + cls_t + gate_t,
                             "total": rgb_all + dep_all + cls_all + gate_all},
        }
