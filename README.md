# 🧬 EvoFormer-NC

### A Multi-Scale Transformer Architecture for Noncoding Variant Effect Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research Proposal](https://img.shields.io/badge/status-research%20proposal-orange.svg)]()

---

## 🌑 The Dark Genome Problem

Nearly **98% of the human genome** does not encode proteins. Once dismissed as *"junk DNA"*, these noncoding regions are now understood to form the genome's regulatory architecture — controlling when, where, and how much each gene is expressed.

Variants in these regions are strongly associated with complex diseases such as **cancer, diabetes, and autism**, yet predicting their functional consequences remains an open and largely unsolved problem.

> *The question is not whether noncoding DNA matters. The question is: can we learn to read it?*

---

## 💡 Why Existing Models Fall Short

| Model | Architecture | Context Window | Evolutionary Signal | Multi-Scale |
|---|---|---|---|---|
| Enformer | CNN + Transformer | 196 kb | ✗ | ✗ |
| DNABERT-2 | BERT | 512 bp | ✗ | ✗ |
| GPN-MSA | Transformer | 512 bp | ✓ | ✗ |
| **EvoFormer-NC** | **Hierarchical Transformer** | **~1.5 Mb** | **✓** | **✓** |

Existing models treat the genome as a flat sequence with a fixed context window and uniform attention. This design misses a fundamental biological reality: **regulatory effects operate across multiple scales simultaneously** — from local binding motifs (10–50 bp) to long-range enhancer-promoter interactions (up to 1 Mb).

---

## 🏗️ Architecture

```
Raw DNA Sequence  [A T G C A T G C ...]
        │
        ▼
┌───────────────────────────────────────────────┐
│           Multi-Scale Tokenizer               │
│                                               │
│  Local tokens    (6-mer,  ~10 bp)             │  ← TF binding motifs
│  Regional tokens (64-mer, ~200 bp)            │  ← Nucleosome / CpG islands
│  Macro tokens    (512-mer, ~1.5 kb)           │  ← Enhancer-promoter loops
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│       Hierarchical Transformer Encoder        │
│                                               │
│  [Local Attn] → [Regional Attn] → [Macro Attn]│
│              ↕ Cross-Scale Attention ↕        │
└───────────────────────────────────────────────┘
        │                    │
        │    Evolutionary     │
        │    Conservation     │
        │    Embeddings ──────┘
        │    (Zoonomia 240-mammal)
        ▼
┌───────────────────────────────────────────────┐
│           Variant Effect Head                 │
│                                               │
│  • Regulatory impact score  (0–1)             │
│  • Affected tissue type     (200+ cell types) │
│  • Mechanism prediction     (enhancer / splice│
│                              / silencer)      │
└───────────────────────────────────────────────┘
```

### Key Innovations

1. **Multi-Scale Tokenization** — DNA is tokenized at three resolutions simultaneously. Cross-scale attention allows local motifs to be interpreted in the context of long-range regulatory interactions.

2. **Evolutionary Priors** — Per-position conservation scores from 240-mammal alignments (Zoonomia) are injected as positional priors. A noncoding position conserved across 100 mammals is more likely to be functionally important.

3. **Multi-Output Variant Head** — Unlike binary pathogenicity classifiers, EvoFormer-NC predicts *what kind* of disruption a variant causes and *in which tissue* — enabling mechanistic interpretation.

---

## 📁 Repository Structure

```
EvoFormer-NC/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── data/
│   ├── preprocessing/
│   │   ├── encode_pipeline.py       # Download and process ENCODE data
│   │   ├── conservation_scores.py   # Compute Zoonomia conservation embeddings
│   │   └── variant_formatter.py     # Format ClinVar / GTEx variants
│   └── splits/
│       └── chromosome_splits.json   # Train/val/test split by chromosome
│
├── model/
│   ├── tokenizer.py                 # Multi-scale DNA tokenizer
│   ├── encoder.py                   # Hierarchical transformer blocks
│   ├── cross_scale_attn.py          # Cross-scale attention mechanism
│   ├── evo_embeddings.py            # Evolutionary conservation embeddings
│   └── variant_head.py              # Variant effect prediction head
│
├── train/
│   ├── train.py                     # Main training loop (PyTorch Lightning)
│   ├── config.yaml                  # Hyperparameters and data paths
│   └── losses.py                    # Multi-task loss function
│
├── eval/
│   ├── benchmark.py                 # Comparison against baselines
│   ├── clinvar_eval.py              # ClinVar pathogenic variant evaluation
│   └── visualize_attention.py       # Attention weight visualization
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Understanding the input data
│   ├── 02_model_walkthrough.ipynb   # Step-by-step model explanation
│   └── 03_variant_interpretation.ipynb  # Interpreting model predictions
│
└── docs/
    ├── proposal.md                  # Full research proposal
    └── architecture.md              # Detailed architecture notes
```

---

## 🗃️ Training Data

| Dataset | Description | Size |
|---|---|---|
| ENCODE Phase 4 | Chromatin accessibility, histone marks | ~3,000 experiments |
| GTEx v9 | eQTL data: variant → gene expression | ~50M variant-gene pairs |
| ClinVar | Clinically annotated noncoding variants | ~200K variants |
| Zoonomia | 240-mammal genome alignment | 240 genomes |
| 1000 Genomes | Human population variation | ~84M variants |

---

## 📊 Evaluation

We benchmark EvoFormer-NC against three strong baselines:

- **Enformer** (Avsec et al., 2021)
- **DNABERT-2** (Zhou et al., 2023)
- **GPN-MSA** (Benegas et al., 2023)

**Primary metric:** AUPRC on pathogenic vs. benign noncoding variant classification (ClinVar held-out set).

**Additional benchmarks:**
- eQTL prioritization on GTEx held-out chromosomes
- Saturation mutagenesis prediction (CRISPR perturbation datasets)

---

## 🗓️ Roadmap (PhD Year 1)

- [ ] Literature review & data pipeline setup *(Months 1–2)*
- [ ] Baseline reproduction (Enformer, DNABERT-2) *(Months 3–4)*
- [ ] EvoFormer-NC architecture implementation *(Months 5–7)*
- [ ] HPC training (SLURM), hyperparameter search *(Months 8–10)*
- [ ] Evaluation, benchmarking, paper draft *(Months 11–12)*

---

## 🚀 Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/EvoFormer-NC.git
cd EvoFormer-NC
pip install -r requirements.txt
```

### Quick model instantiation

```python
from model.tokenizer import MultiScaleTokenizer
from model.encoder import EvoFormerEncoder
from model.variant_head import VariantEffectHead

tokenizer = MultiScaleTokenizer(scales=["local", "regional", "macro"])
encoder = EvoFormerEncoder(d_model=512, n_heads=8, n_layers=6)
head = VariantEffectHead(d_model=512, n_tissues=200)

tokens = tokenizer.encode("ATGCATGCATGC...")
features = encoder(tokens)
predictions = head(features)
# → {"impact_score": 0.87, "tissue": "liver", "mechanism": "enhancer_disruption"}
```

---

## 📖 Research Proposal

Full research proposal is available in [`docs/proposal.md`](docs/proposal.md).

---

## 📚 References

- Avsec et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods*.
- Zhou et al. (2023). DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. *arXiv*.
- Nguyen et al. (2024). HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. *ICML*.
- Zoonomia Consortium (2020). A comparative genomics multitool for scientific discovery. *Nature*.
- Benegas et al. (2023). DNA language models are powerful predictors of genome-wide variant effects. *PNAS*.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*This repository represents a doctoral research proposal submitted to the CGR Lab, Chalmers University of Technology.*
