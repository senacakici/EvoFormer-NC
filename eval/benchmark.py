"""
Benchmark EvoFormer-NC against Enformer, DNABERT-2, and GPN-MSA
on the ClinVar noncoding variant test set.

Usage
-----
    python eval/benchmark.py --checkpoint checkpoints/best.ckpt
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
import torch


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute AUPRC, AUROC, and F1 for binary classification.

    Parameters
    ----------
    y_true  : (N,) integer array (0 = benign, 1 = pathogenic)
    y_score : (N,) float array (model's predicted impact score)
    """
    auprc  = average_precision_score(y_true, y_score)
    auroc  = roc_auc_score(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    f1     = f1_score(y_true, y_pred, zero_division=0)

    return {
        "AUPRC":  round(auprc, 4),
        "AUROC":  round(auroc, 4),
        "F1":     round(f1,    4),
        "n_pos":  int(y_true.sum()),
        "n_neg":  int((1 - y_true).sum()),
    }


def compute_tissue_metrics(
    y_true: np.ndarray,   # (N, n_tissues)
    y_logits: np.ndarray, # (N, n_tissues)
) -> dict:
    """Macro-averaged AUPRC across tissue types."""
    y_prob = 1 / (1 + np.exp(-y_logits))  # sigmoid
    per_tissue = []
    for t in range(y_true.shape[1]):
        if y_true[:, t].sum() > 0:
            per_tissue.append(average_precision_score(y_true[:, t], y_prob[:, t]))
    return {
        "tissue_macro_AUPRC": round(np.mean(per_tissue), 4),
        "n_tissues_evaluated": len(per_tissue),
    }


# ── Baseline stubs ────────────────────────────────────────────────────────────
# In a real benchmark you would import each baseline model here.
# These stubs return random scores for illustration.

def run_enformer_baseline(records: list) -> np.ndarray:
    """Placeholder: replace with actual Enformer inference."""
    return np.random.rand(len(records))


def run_dnabert2_baseline(records: list) -> np.ndarray:
    """Placeholder: replace with actual DNABERT-2 inference."""
    return np.random.rand(len(records))


def run_gpn_msa_baseline(records: list) -> np.ndarray:
    """Placeholder: replace with actual GPN-MSA inference."""
    return np.random.rand(len(records))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to EvoFormer-NC checkpoint (.ckpt)")
    parser.add_argument("--test_records", type=str,
                        default="data/splits/test.json")
    parser.add_argument("--output", type=str,
                        default="results/benchmark_results.json")
    args = parser.parse_args()

    # Load test data
    with open(args.test_records) as f:
        records = json.load(f)

    y_true = np.array([r["impact_label"] for r in records])

    print(f"Test set: {len(records)} variants  "
          f"({int(y_true.sum())} pathogenic / {int((1-y_true).sum())} benign)\n")

    results = {}

    # ── EvoFormer-NC ──────────────────────────────────────────────────────────
    print("Running EvoFormer-NC ...")
    # TODO: load model and run inference
    # from train.train import EvoFormerNC
    # model = EvoFormerNC.load_from_checkpoint(args.checkpoint)
    # model.eval()
    # y_score_evo = run_evoformer(model, records)
    y_score_evo = np.random.rand(len(records))   # placeholder
    results["EvoFormer-NC"] = compute_metrics(y_true, y_score_evo)

    # ── Baselines ─────────────────────────────────────────────────────────────
    for name, fn in [
        ("Enformer",  run_enformer_baseline),
        ("DNABERT-2", run_dnabert2_baseline),
        ("GPN-MSA",   run_gpn_msa_baseline),
    ]:
        print(f"Running {name} ...")
        y_score = fn(records)
        results[name] = compute_metrics(y_true, y_score)

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"{'Model':<16} {'AUPRC':>8} {'AUROC':>8} {'F1':>8}")
    print("-" * 52)
    for model_name, m in results.items():
        marker = " ◀" if model_name == "EvoFormer-NC" else ""
        print(f"{model_name:<16} {m['AUPRC']:>8} {m['AUROC']:>8} {m['F1']:>8}{marker}")
    print("=" * 52)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
