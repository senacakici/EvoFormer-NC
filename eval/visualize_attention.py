"""
Attention Weight Visualization for EvoFormer-NC.

Visualizes which positions in the genome the model attends to
when predicting the effect of a noncoding variant.

Usage
-----
    python eval/visualize_attention.py \
        --checkpoint checkpoints/best.ckpt \
        --sequence ATGCATGC... \
        --variant_pos 128
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    sequence: str,
    variant_pos: int,
    scale: str = "local",
    layer: int = -1,
    head: int = 0,
    window: int = 50,
    save_path: Optional[str] = None,
):
    """
    Plot attention weights around the variant position.

    Parameters
    ----------
    attn_weights : (n_layers, n_heads, L, L) attention tensor
    sequence     : DNA string
    variant_pos  : position of the variant
    scale        : which scale to visualise ("local" / "regional" / "macro")
    layer        : which layer (-1 = last)
    head         : which attention head
    window       : number of positions to show on each side
    """
    weights = attn_weights[layer, head]  # (L, L)

    start = max(0, variant_pos - window)
    end   = min(weights.shape[0], variant_pos + window)

    weights_crop = weights[start:end, start:end]
    seq_crop = list(sequence[start:end])
    rel_pos  = variant_pos - start

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), 
                              gridspec_kw={"width_ratios": [3, 1]})

    # ── Heatmap ───────────────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(
        weights_crop,
        cmap="YlOrRd",
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Attention weight")

    # Mark variant position
    ax.axvline(x=rel_pos, color="dodgerblue", linewidth=2, linestyle="--", alpha=0.8)
    ax.axhline(y=rel_pos, color="dodgerblue", linewidth=2, linestyle="--", alpha=0.8)

    ax.set_title(
        f"Attention weights — {scale} scale | Layer {layer} | Head {head}\n"
        f"Variant at position {variant_pos} (blue dashed)",
        fontsize=11,
    )
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    # Tick labels (every 10 positions)
    tick_step = max(1, (end - start) // 10)
    ticks = range(0, end - start, tick_step)
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels([str(start + t) for t in ticks], rotation=45, fontsize=7)
    ax.set_yticklabels([str(start + t) for t in ticks], fontsize=7)

    # ── Per-position attention from variant ───────────────────────────────────
    ax2 = axes[1]
    variant_attn = weights_crop[rel_pos]  # attention FROM the variant position
    positions = np.arange(start, end)

    # Color by nucleotide
    nt_colors = {"A": "#2ecc71", "T": "#e74c3c", "G": "#f39c12", "C": "#3498db", "N": "#95a5a6"}
    colors = [nt_colors.get(nt, "#95a5a6") for nt in seq_crop]

    ax2.barh(
        np.arange(len(positions)),
        variant_attn,
        color=colors,
        edgecolor="none",
        height=0.8,
    )
    ax2.axhline(y=rel_pos, color="dodgerblue", linewidth=2, linestyle="--", alpha=0.8)
    ax2.set_title(f"Attention from pos {variant_pos}", fontsize=10)
    ax2.set_xlabel("Attention weight")
    ax2.set_yticks(np.arange(len(positions))[::tick_step])
    ax2.set_yticklabels([str(p) for p in positions[::tick_step]], fontsize=7)

    # Legend for nucleotides
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=nt) for nt, c in nt_colors.items() if nt != "N"]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig


def plot_cross_scale_summary(
    local_attn:    np.ndarray,  # (L_local,)
    regional_attn: np.ndarray,  # (L_regional,)
    macro_attn:    np.ndarray,  # (L_macro,)
    variant_pos: int,
    save_path: Optional[str] = None,
):
    """
    Side-by-side view of per-position attention weight across all three scales.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    scale_data = [
        ("Local (~10 bp/token)",    local_attn,    "bp"),
        ("Regional (~200 bp/token)", regional_attn, "200-bp bins"),
        ("Macro (~1.5 kb/token)",   macro_attn,    "1.5-kb bins"),
    ]

    for ax, (title, attn, unit) in zip(axes, scale_data):
        x = np.arange(len(attn))
        ax.fill_between(x, attn, alpha=0.6, color="#e74c3c")
        ax.plot(x, attn, color="#c0392b", linewidth=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Attention", fontsize=9)
        ax.set_xlabel(f"Position ({unit})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        f"Cross-scale attention summary — variant at position {variant_pos}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--sequence",     type=str, required=True,
                        help="DNA string to analyse")
    parser.add_argument("--variant_pos",  type=int, required=True)
    parser.add_argument("--scale",        type=str, default="local",
                        choices=["local", "regional", "macro"])
    parser.add_argument("--layer",        type=int, default=-1)
    parser.add_argument("--head",         type=int, default=0)
    parser.add_argument("--output_dir",   type=str, default="results/attention/")
    args = parser.parse_args()

    print("NOTE: Load your trained model here and extract attention weights.")
    print("      Plotting with random weights for demonstration.\n")

    seq_len = len(args.sequence)

    # ── Placeholder: replace with real model attention extraction ──────────────
    n_layers, n_heads = 6, 8
    fake_attn = np.random.rand(n_layers, n_heads, seq_len, seq_len)
    fake_attn = fake_attn / fake_attn.sum(axis=-1, keepdims=True)  # row-normalise

    plot_attention_heatmap(
        attn_weights = fake_attn,
        sequence     = args.sequence,
        variant_pos  = args.variant_pos,
        scale        = args.scale,
        layer        = args.layer,
        head         = args.head,
        save_path    = f"{args.output_dir}/heatmap_{args.scale}_l{args.layer}_h{args.head}.png",
    )


if __name__ == "__main__":
    main()
