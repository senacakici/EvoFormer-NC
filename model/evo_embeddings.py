"""
Evolutionary Conservation Embeddings for EvoFormer-NC.

Injects per-position phylogenetic conservation scores (PhyloP / PhastCons)
derived from the Zoonomia 240-mammal alignment as positional priors into
the encoder representations.

Conservation scores act as a soft "importance mask": positions conserved
across mammals are more likely to be functionally relevant, so the model
can up-weight their representations accordingly.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


class ConservationEmbedding(nn.Module):
    """
    Projects scalar conservation scores to d_model-dimensional vectors
    and adds them to the sequence representations.

    The scores are expected to be in the range [-10, 10] (PhyloP scale)
    or [0, 1] (PhastCons scale). They are normalised internally.

    Parameters
    ----------
    d_model    : int   — hidden dimension of the encoder
    score_type : str   — "phylop" | "phastcons"
    dropout    : float — dropout on the projected embeddings
    """

    SCORE_RANGES = {
        "phylop":    (-10.0, 10.0),
        "phastcons": (0.0,   1.0),
    }

    def __init__(
        self,
        d_model: int = 512,
        score_type: str = "phylop",
        dropout: float = 0.1,
    ):
        super().__init__()
        assert score_type in self.SCORE_RANGES, (
            f"score_type must be one of {list(self.SCORE_RANGES)}"
        )
        self.score_type = score_type
        self.score_min, self.score_max = self.SCORE_RANGES[score_type]

        # Scalar → d_model projection
        self.projection = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Learnable gate: how much conservation to mix in
        self.gate = nn.Parameter(torch.zeros(1))  # starts at 0 (no influence)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        representations: torch.Tensor,
        conservation_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        representations   : (B, L, D) — encoder hidden states
        conservation_scores : (B, L) or (B, L, 1) — per-position scores

        Returns
        -------
        (B, L, D) — representations enriched with conservation context
        """
        if conservation_scores.dim() == 2:
            conservation_scores = conservation_scores.unsqueeze(-1)  # (B, L, 1)

        # Normalise to [0, 1]
        scores_norm = (conservation_scores - self.score_min) / (
            self.score_max - self.score_min + 1e-8
        )
        scores_norm = scores_norm.clamp(0.0, 1.0)

        # Project to d_model
        evo_emb = self.projection(scores_norm)  # (B, L, D)
        evo_emb = self.dropout(evo_emb)

        # Gated residual addition
        gate = torch.sigmoid(self.gate)
        out = self.norm(representations + gate * evo_emb)
        return out


class MultiSpeciesConservationEmbedding(nn.Module):
    """
    Extended version that takes per-species conservation vectors
    (e.g. 240-dimensional Zoonomia alignment rows) and summarises
    them before injection.

    This allows the model to distinguish *which* clades are conserved,
    not just the aggregate score.

    Parameters
    ----------
    d_model    : int — encoder hidden dimension
    n_species  : int — number of species in the alignment (e.g. 240)
    dropout    : float
    """

    def __init__(
        self,
        d_model: int = 512,
        n_species: int = 240,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_species = n_species

        # Compress species vector → d_model
        self.species_encoder = nn.Sequential(
            nn.Linear(n_species, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )

        self.gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        representations: torch.Tensor,
        species_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        representations : (B, L, D)
        species_matrix  : (B, L, n_species) — binary alignment matrix
                          (1 = nucleotide present in that species)

        Returns
        -------
        (B, L, D)
        """
        evo_emb = self.species_encoder(species_matrix.float())  # (B, L, D)
        evo_emb = self.dropout(evo_emb)
        gate = torch.sigmoid(self.gate)
        return self.norm(representations + gate * evo_emb)
