"""
Variant Effect Prediction Head for EvoFormer-NC.

Takes the difference between reference and alternate allele representations
at each scale and predicts:
  1. Regulatory impact score   (continuous 0–1)
  2. Affected tissue type      (multi-label, 200+ cell types)
  3. Disruption mechanism      (enhancer / silencer / splice / other)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ── Tissue labels (subset, full list loaded from data/tissue_labels.json) ─────

MECHANISM_LABELS = [
    "enhancer_disruption",
    "silencer_gain",
    "splice_site_alteration",
    "promoter_disruption",
    "insulator_disruption",
    "other",
]


# ── Variant Difference Module ─────────────────────────────────────────────────

class VariantDifferencePooling(nn.Module):
    """
    Computes a fixed-size representation of the effect of a variant by
    comparing reference and alternate allele encoder outputs.

    Strategy: take the representation at the variant position ± a window,
    compute (alt − ref), and mean-pool.
    """

    def __init__(self, d_model: int, window: int = 32):
        super().__init__()
        self.window = window
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        ref_repr: torch.Tensor,   # (B, L, D)
        alt_repr: torch.Tensor,   # (B, L, D)
        variant_pos: torch.Tensor,  # (B,) — position index in the sequence
    ) -> torch.Tensor:            # (B, D)
        B, L, D = ref_repr.shape
        w = self.window

        outputs = []
        for i in range(B):
            pos = int(variant_pos[i].item())
            start = max(0, pos - w)
            end = min(L, pos + w + 1)

            ref_slice = ref_repr[i, start:end]  # (2w+1, D)
            alt_slice = alt_repr[i, start:end]

            diff = alt_slice - ref_slice         # element-wise difference
            pooled = diff.mean(dim=0)            # (D,)
            outputs.append(pooled)

        out = torch.stack(outputs, dim=0)        # (B, D)
        return self.norm(out)


# ── Prediction Heads ──────────────────────────────────────────────────────────

class ImpactScoreHead(nn.Module):
    """Predicts a single regulatory impact score in [0, 1]."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


class TissueSpecificityHead(nn.Module):
    """
    Multi-label classifier predicting which cell types are affected.

    Returns logits (apply sigmoid + threshold at inference time).
    """

    def __init__(self, d_model: int, n_tissues: int = 200, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_tissues),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, n_tissues) — raw logits


class MechanismHead(nn.Module):
    """
    Multi-class classifier for the type of regulatory disruption.
    Classes: enhancer_disruption | silencer_gain | splice_site_alteration |
             promoter_disruption | insulator_disruption | other
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        n_classes = len(MECHANISM_LABELS)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, n_classes) — raw logits


# ── Full Variant Effect Head ──────────────────────────────────────────────────

class VariantEffectHead(nn.Module):
    """
    Combines outputs from all three encoder scales and runs all three
    prediction heads.

    Parameters
    ----------
    d_model   : int   — encoder hidden dimension
    n_tissues : int   — number of cell-type labels
    window    : int   — nt window around variant position for pooling
    dropout   : float

    Example
    -------
    >>> head = VariantEffectHead(d_model=512, n_tissues=200)
    >>> ref_repr = {"local": torch.randn(2, 512, 512), ...}
    >>> alt_repr = {"local": torch.randn(2, 512, 512), ...}
    >>> pos = torch.tensor([128, 300])
    >>> out = head(ref_repr, alt_repr, pos)
    >>> out["impact_score"].shape      # (2,)
    >>> out["tissue_logits"].shape     # (2, 200)
    >>> out["mechanism_logits"].shape  # (2, 6)
    """

    SCALES = ["local", "regional", "macro"]

    def __init__(
        self,
        d_model: int = 512,
        n_tissues: int = 200,
        window: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Per-scale difference pooling
        self.diff_pool = nn.ModuleDict({
            s: VariantDifferencePooling(d_model, window) for s in self.SCALES
        })

        # Fuse all scales → single vector
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * len(self.SCALES), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Prediction heads
        self.impact_head    = ImpactScoreHead(d_model, dropout)
        self.tissue_head    = TissueSpecificityHead(d_model, n_tissues, dropout)
        self.mechanism_head = MechanismHead(d_model, dropout)

    def forward(
        self,
        ref_representations: Dict[str, torch.Tensor],
        alt_representations: Dict[str, torch.Tensor],
        variant_positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        ref_representations : dict of (B, L_scale, D) per scale
        alt_representations : same structure
        variant_positions   : (B,) integer positions in the local-scale coords

        Returns
        -------
        dict with keys:
            "impact_score"      : (B,)        float in [0, 1]
            "tissue_logits"     : (B, n_tissues) raw logits
            "mechanism_logits"  : (B, 6)       raw logits
            "fused"             : (B, D)        fused representation
        """
        # Pool per-scale differences
        scale_diffs = []
        for s in self.SCALES:
            diff = self.diff_pool[s](
                ref_representations[s],
                alt_representations[s],
                variant_positions,
            )
            scale_diffs.append(diff)

        # Fuse scales
        fused = self.scale_fusion(torch.cat(scale_diffs, dim=-1))  # (B, D)

        return {
            "impact_score":     self.impact_head(fused),
            "tissue_logits":    self.tissue_head(fused),
            "mechanism_logits": self.mechanism_head(fused),
            "fused":            fused,
        }

    @staticmethod
    def decode_mechanism(logits: torch.Tensor) -> list[str]:
        """Convert mechanism logits to human-readable labels."""
        indices = logits.argmax(dim=-1).tolist()
        return [MECHANISM_LABELS[i] for i in indices]
