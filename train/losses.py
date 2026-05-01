"""
Multi-Task Loss for EvoFormer-NC.

Combines three objectives:
  1. Impact score   — Binary cross-entropy (pathogenic vs. benign)
  2. Tissue labels  — Binary cross-entropy with logits (multi-label)
  3. Mechanism      — Cross-entropy (multi-class)

Weights are configurable from config.yaml.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EvoFormerLoss(nn.Module):
    """
    Weighted multi-task loss.

    Parameters
    ----------
    w_impact    : float — weight for impact BCE loss
    w_tissue    : float — weight for tissue multi-label loss
    w_mechanism : float — weight for mechanism cross-entropy loss
    pos_weight  : float — positive class weight for imbalanced impact labels
    """

    def __init__(
        self,
        w_impact:    float = 1.0,
        w_tissue:    float = 0.5,
        w_mechanism: float = 0.5,
        pos_weight:  float = 5.0,  # pathogenic variants are rare
    ):
        super().__init__()
        self.w_impact    = w_impact
        self.w_tissue    = w_tissue
        self.w_mechanism = w_mechanism
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.float)
        )

    def forward(
        self,
        impact_pred:      torch.Tensor,  # (B,)        sigmoid output
        tissue_logits:    torch.Tensor,  # (B, n_tissues)
        mechanism_logits: torch.Tensor,  # (B, n_classes)
        impact_label:     torch.Tensor,  # (B,)        float 0/1
        tissue_labels:    torch.Tensor,  # (B, n_tissues) float 0/1
        mechanism_label:  torch.Tensor,  # (B,)        long
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        total_loss : scalar tensor
        loss_dict  : {"impact": ..., "tissue": ..., "mechanism": ...}
        """
        # 1. Impact loss — treat as binary classification
        loss_impact = F.binary_cross_entropy(
            impact_pred,
            impact_label,
            reduction="mean",
        )

        # 2. Tissue loss — multi-label binary cross-entropy
        loss_tissue = F.binary_cross_entropy_with_logits(
            tissue_logits,
            tissue_labels,
            reduction="mean",
        )

        # 3. Mechanism loss — multi-class cross-entropy
        loss_mechanism = F.cross_entropy(
            mechanism_logits,
            mechanism_label,
            reduction="mean",
        )

        total = (
            self.w_impact    * loss_impact
            + self.w_tissue    * loss_tissue
            + self.w_mechanism * loss_mechanism
        )

        return total, {
            "impact":    loss_impact.detach(),
            "tissue":    loss_tissue.detach(),
            "mechanism": loss_mechanism.detach(),
        }
