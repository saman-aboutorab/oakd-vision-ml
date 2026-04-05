"""Batch-hard triplet loss implemented from scratch.

For each anchor in the batch:
  - Hardest positive: same-identity crop with LARGEST distance from anchor
  - Hardest negative: different-identity crop with SMALLEST distance from anchor

Loss = mean over all anchors of max(0, d_pos - d_neg + margin)

This focuses training on the most difficult cases — easy triplets (already
well-separated) contribute zero gradient and are ignored.
"""

import torch
import torch.nn as nn


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute all-pairs L2 distances for a batch of embeddings.

    Since embeddings are L2-normalized, we can use the identity:
        ||a - b||² = 2 - 2·(a·b)
    which is numerically stable and fast.

    Args:
        embeddings: [B, D] L2-normalized embedding vectors

    Returns:
        [B, B] distance matrix where dist[i, j] = ||emb[i] - emb[j]||
    """
    dot = embeddings @ embeddings.T          # [B, B]
    sq_dist = 2.0 - 2.0 * dot               # [B, B], guaranteed >= 0
    sq_dist = sq_dist.clamp(min=0.0)        # numerical safety
    return sq_dist.sqrt()


class BatchHardTripletLoss(nn.Module):
    """Triplet loss with batch-hard mining.

    Args:
        margin: minimum gap enforced between positive and negative distances
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: [B, D] L2-normalized embeddings
            labels:     [B]    integer identity labels

        Returns:
            scalar loss, scalar fraction of non-zero (active) triplets
        """
        dist = pairwise_distances(embeddings)   # [B, B]
        B = embeddings.size(0)

        # Boolean masks: same_id[i,j] = True if labels[i] == labels[j]
        same_id = labels.unsqueeze(0) == labels.unsqueeze(1)   # [B, B]
        diff_id = ~same_id

        # Hardest positive per anchor: max distance among same-identity pairs
        # Set diagonal and different-id entries to 0 before taking max
        pos_dist = dist * same_id.float()
        pos_dist[~same_id] = 0.0
        hardest_pos, _ = pos_dist.max(dim=1)   # [B]

        # Hardest negative per anchor: min distance among different-identity pairs
        # Set same-id entries to large value before taking min
        neg_dist = dist.clone()
        neg_dist[same_id] = float("inf")
        hardest_neg, _ = neg_dist.min(dim=1)   # [B]

        # Triplet loss
        triplet_loss = (hardest_pos - hardest_neg + self.margin).clamp(min=0.0)
        loss = triplet_loss.mean()

        # Fraction of active triplets (non-zero loss) — useful for monitoring
        active_frac = (triplet_loss > 0).float().mean().item()

        return loss, active_frac
