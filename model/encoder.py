"""
Hierarchical Transformer Encoder for EvoFormer-NC.

Three independent transformer towers (local / regional / macro) share
information through a cross-scale attention layer after each encoder block.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ── Positional Encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32_768, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Single Transformer Block ──────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with pre-norm
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = residual + self.dropout(attn_out)
        # Feed-forward with pre-norm
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn_weights


# ── Cross-Scale Attention ─────────────────────────────────────────────────────

class CrossScaleAttention(nn.Module):
    """
    Allows each scale to attend to a summary of the other scales.
    Each scale produces a CLS-style summary token that is broadcast
    to the other two scales via cross-attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Cross-attention: query from current scale, key/value from other scales
        self.cross_attn = nn.ModuleDict({
            "local":    nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            "regional": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
            "macro":    nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
        })
        self.norm = nn.ModuleDict({
            "local":    nn.LayerNorm(d_model),
            "regional": nn.LayerNorm(d_model),
            "macro":    nn.LayerNorm(d_model),
        })
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        scale_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        scale_outputs : dict
            {"local": (B, L_local, D), "regional": (B, L_reg, D), "macro": (B, L_mac, D)}

        Returns
        -------
        Updated dict with the same shapes, enriched with cross-scale context.
        """
        scales = ["local", "regional", "macro"]
        # Compute mean-pool summary for each scale → (B, 1, D)
        summaries = {
            s: scale_outputs[s].mean(dim=1, keepdim=True)
            for s in scales
        }

        updated = {}
        for s in scales:
            # Context = summaries of the OTHER two scales → (B, 2, D)
            others = [summaries[o] for o in scales if o != s]
            context = torch.cat(others, dim=1)
            # Cross-attention: query=current scale, key/value=other scales
            query = self.norm[s](scale_outputs[s])
            attn_out, _ = self.cross_attn[s](query, context, context)
            updated[s] = scale_outputs[s] + self.dropout(attn_out)

        return updated


# ── Scale Tower ───────────────────────────────────────────────────────────────

class ScaleTower(nn.Module):
    """A stack of transformer blocks for one scale."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        x = self.pos_enc(self.embedding(token_ids))
        all_attn_weights = []
        for block in self.blocks:
            x, w = block(x, key_padding_mask=padding_mask)
            all_attn_weights.append(w)
        return x, all_attn_weights


# ── Main Encoder ──────────────────────────────────────────────────────────────

class EvoFormerEncoder(nn.Module):
    """
    Hierarchical transformer encoder.

    Three scale towers process the tokenized DNA in parallel.
    After every encoder layer a CrossScaleAttention module exchanges
    information between the scales.

    Parameters
    ----------
    vocab_size  : int   — shared vocabulary size across scales
    d_model     : int   — hidden dimension (default 512)
    n_heads     : int   — attention heads (default 8)
    n_layers    : int   — transformer layers per scale (default 6)
    ffn_dim     : int   — feed-forward expansion (default 2048)
    dropout     : float — dropout probability (default 0.1)

    Example
    -------
    >>> encoder = EvoFormerEncoder(vocab_size=4**6 + 4)
    >>> tokens = {"local": torch.randint(4, (2, 512)),
    ...           "regional": torch.randint(4, (2, 64)),
    ...           "macro": torch.randint(4, (2, 8))}
    >>> out = encoder(tokens)
    >>> out["local"].shape    # (2, 512, 512)
    """

    SCALES = ["local", "regional", "macro"]

    def __init__(
        self,
        vocab_size: int = 4100,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # One tower per scale
        self.towers = nn.ModuleDict({
            s: ScaleTower(vocab_size, d_model, n_heads, n_layers, ffn_dim, dropout)
            for s in self.SCALES
        })

        # Cross-scale attention after every layer (shared weights across layers)
        self.cross_scale = CrossScaleAttention(d_model, n_heads, dropout)

        # Final layer norm per scale
        self.final_norm = nn.ModuleDict({
            s: nn.LayerNorm(d_model) for s in self.SCALES
        })

    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        padding_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        tokens        : dict  {"local": (B, L_l), "regional": (B, L_r), "macro": (B, L_m)}
        padding_masks : dict  same keys, bool tensors (True = pad position)

        Returns
        -------
        dict of contextualised representations, same shape as input but last
        dim = d_model.
        """
        if padding_masks is None:
            padding_masks = {s: None for s in self.SCALES}

        # Run each scale tower
        representations = {}
        for s in self.SCALES:
            rep, _ = self.towers[s](tokens[s], padding_masks[s])
            representations[s] = rep

        # Exchange cross-scale context
        representations = self.cross_scale(representations)

        # Final norm
        return {s: self.final_norm[s](representations[s]) for s in self.SCALES}
