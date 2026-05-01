# EvoFormer-NC

**A deep learning model that predicts whether a DNA change can cause disease focusing on the 98% of the genome that most tools ignore.**

---

## What problem does this solve?

The human genome is made up of ~3 billion letters (A, T, G, C).

Only **2% of it** produces proteins the molecules that do most of the work in our cells. Scientists understand this 2% fairly well.

The other **98%** was once called "junk DNA." We now know that's wrong. This region acts like a **control panel**: it decides when, where, and how much each gene is switched on or off. Errors in this control panel are linked to cancer, diabetes, autism, and many other diseases.

The problem? **We don't know how to read it yet.**

When a single letter changes in this 98% region (a "variant"), we usually can't tell whether it's harmless or disease-causing. Existing AI models struggle here because they only look at a small window of DNA at a time — but the control panel works across very large distances.

---

## What does EvoFormer-NC do?

EvoFormer-NC is a **transformer-based deep learning model** (the same technology behind ChatGPT, but applied to DNA) that:

1. **Reads DNA at three zoom levels at once**
   - Close-up (~10 letters) → detects small binding sites
   - Medium (~200 letters) → detects structural signals
   - Wide (~1,500 letters) → detects long-range gene switches

2. **Uses evolutionary history as a clue**
   If a DNA position has stayed the same across 240 mammal species over millions of years, it's probably important. EvoFormer-NC uses this signal (from the [Zoonomia project](https://zoonomiaproject.org/)) to focus attention on what matters.

3. **Predicts three things about any DNA change**
   - How likely it is to be harmful (a score from 0 to 1)
   - Which tissues or cell types are affected (e.g. liver, brain, heart)
   - What kind of disruption it causes (e.g. a gene switch being broken)

---

## How is this different from existing models?

| Model | Zoom levels | Uses evolution | Predicts tissue + mechanism |
|---|---|---|---|
| Enformer | 1 (wide only) | No | Partial |
| DNABERT-2 | 1 (close only) | No | No |
| GPN-MSA | 1 (close only) | Yes | No |
| **EvoFormer-NC** | **3 (all at once)** | **Yes** | **Yes** |

---

## Project structure

```
EvoFormer-NC/
│
├── model/
│   ├── tokenizer.py        ← Converts DNA letters into numbers the model can read
│   ├── encoder.py          ← The main transformer that reads DNA at 3 zoom levels
│   ├── evo_embeddings.py   ← Adds evolutionary conservation as extra context
│   └── variant_head.py     ← Outputs the final predictions
│
├── train/
│   ├── train.py            ← Runs the training process
│   ├── losses.py           ← Measures how wrong the model's predictions are
│   └── config.yaml         ← All settings (learning rate, batch size, etc.)
│
├── eval/
│   ├── benchmark.py        ← Compares EvoFormer-NC against other models
│   └── visualize_attention.py ← Shows which DNA positions the model focused on
│
└── docs/
    └── proposal.md         ← Full research proposal
```

---

## Data sources

The model is trained on publicly available genomic datasets:

| Dataset | What it contains |
|---|---|
| [ENCODE Phase 4](https://www.encodeproject.org/) | Which parts of the genome are "active" in different cell types |
| [GTEx v9](https://gtexportal.org/) | How DNA changes affect gene activity across human tissues |
| [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) | Known disease-causing and harmless DNA variants |
| [Zoonomia](https://zoonomiaproject.org/) | Genome alignments across 240 mammal species |
| [1000 Genomes](https://www.internationalgenome.org/) | Natural DNA variation across human populations |

---

## Quick start

```bash
git clone https://github.com/senacakici/EvoFormer-NC.git
cd EvoFormer-NC
pip install -r requirements.txt
```

```python
from model.tokenizer import MultiScaleTokenizer
from model.encoder import EvoFormerEncoder
from model.variant_head import VariantEffectHead

# Load components
tokenizer = MultiScaleTokenizer()
encoder   = EvoFormerEncoder(d_model=512)
head      = VariantEffectHead(d_model=512, n_tissues=200)

# Encode a DNA sequence
tokens = tokenizer.encode("ATGCATGCATGCATGCATGC...")

# Run the model
features    = encoder(tokens)
predictions = head(features, features, variant_positions)

# predictions["impact_score"]     → 0.87  (high = likely harmful)
# predictions["tissue_logits"]    → which cell types are affected
# predictions["mechanism_logits"] → what kind of disruption
```

---

## Roadmap

-  Data pipeline (ENCODE + GTEx + ClinVar preprocessing)
-  Baseline reproduction (Enformer, DNABERT-2, GPN-MSA)
-  EvoFormer-NC architecture implementation
-  Training on HPC cluster
-  Benchmarking and paper draft



