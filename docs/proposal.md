# Research Proposal

## Decoding the Dark Genome: A Multi-Scale Transformer Architecture for Noncoding Variant Effect Prediction

**Target Lab:** Computational Genomics Research (CGR) Lab, Chalmers University of Technology
**Division:** Data Science and AI (DSAI)
**Supervisor:** Asst. Prof. Sina Majidian

---

## 1. Motivation

Nearly 98% of the human genome does not encode proteins. Once dismissed as "junk DNA," these noncoding regions are now understood to form the genome's regulatory architecture — controlling when, where, and how much each gene is expressed. Crucially, variants in these regions are strongly associated with complex diseases such as cancer, diabetes, and autism, yet predicting their functional consequences remains an open and largely unsolved problem.

Existing models such as Enformer and DNABERT treat the genome as a flat sequence, processing DNA with fixed-length context windows and uniform attention. This design misses a fundamental biological reality: regulatory effects operate across **multiple scales simultaneously** — from local motifs (10–50 bp) to long-range enhancer-promoter interactions (up to 1 Mb). A model blind to this hierarchy cannot fully decode the regulatory grammar of noncoding DNA.

---

## 2. Research Question

> *Can a multi-scale, hierarchical transformer architecture trained on evolutionary conservation signals across a pangenome better predict the functional effects of noncoding variants than current single-scale models?*

---

## 3. Proposed Method: EvoFormer-NC

We propose **EvoFormer-NC** (*Evolutionary Multi-Scale Transformer for Noncoding Variants*), a novel deep learning architecture with three key innovations:

### 3.1 Multi-Scale Tokenization

Rather than tokenizing DNA at a fixed k-mer size, EvoFormer-NC uses a **hierarchical tokenization scheme** operating at three resolutions:

- **Local tokens** (6-mer, stride 1, ~10 bp) — capture transcription factor binding motifs
- **Regional tokens** (6-mer, stride 8, ~200 bp) — capture nucleosome positioning and CpG islands
- **Macro tokens** (6-mer, stride 64, ~1.5 kb) — capture enhancer-promoter loop interactions

Each level is encoded by an independent transformer tower. A cross-scale attention module then allows information to flow between levels — macro-scale context can modulate the interpretation of a local motif.

### 3.2 Evolutionary Signal as a Structural Prior

Instead of training only on functional assay labels (e.g., chromatin accessibility from ENCODE), EvoFormer-NC incorporates **cross-species conservation scores** as an auxiliary training signal.

The intuition: a noncoding position conserved across 100 mammals is more likely to be functionally important than one that is not. We leverage multi-species alignment data from the Zoonomia 240-mammal project to compute per-position PhyloP conservation embeddings, which are injected into the local-scale representation via a learnable gated module.

### 3.3 Multi-Output Variant Effect Head

A dedicated prediction head takes the representation difference between reference and alternate alleles across all three scales and produces:

- **Regulatory impact score** — continuous value in [0, 1]
- **Affected tissue type** — multi-label classification across 200+ cell types
- **Disruption mechanism** — enhancer disruption / silencer gain / splice site alteration / other

---

## 4. Training Data

| Dataset | Description | Size |
|---|---|---|
| ENCODE Phase 4 | Chromatin accessibility, histone marks | ~3,000 experiments |
| GTEx v9 | eQTL data linking variants to gene expression | ~50M variant-gene pairs |
| ClinVar | Clinically annotated noncoding variants | ~200K variants |
| Zoonomia | 240-mammal genome alignment | 240 genomes |
| 1000 Genomes | Human population variation | ~84M variants |

---

## 5. Evaluation

We evaluate against three strong baselines:

| Model | Architecture | Context | Evo. Signal |
|---|---|---|---|
| Enformer | CNN + Transformer | 196 kb | ✗ |
| DNABERT-2 | BERT | 512 bp | ✗ |
| GPN-MSA | Transformer | 512 bp | ✓ |
| **EvoFormer-NC** | **Hierarchical Transformer** | **~1.5 Mb** | **✓** |

**Primary metric:** AUPRC on pathogenic vs. benign noncoding variant classification (ClinVar held-out).

**Secondary benchmarks:**
- eQTL prioritisation on GTEx held-out chromosomes
- Saturation mutagenesis prediction on CRISPR perturbation datasets

---

## 6. Timeline (PhD Year 1)

| Period | Milestone |
|---|---|
| Months 1–2 | Literature review, data pipeline setup |
| Months 3–4 | Baseline model reproduction |
| Months 5–7 | EvoFormer-NC architecture development |
| Months 8–10 | HPC training (SLURM), hyperparameter search |
| Months 11–12 | Evaluation, benchmarking, first paper draft |

---

## 7. Why This is Novel

Current models choose one scale or one signal. EvoFormer-NC is the first architecture to jointly model:

1. **Hierarchical sequence context** from local motifs to long-range loops
2. **Evolutionary conservation** from 240-mammal alignment as a structural prior
3. **Multi-output variant interpretation** — score + tissue + mechanism simultaneously

This mirrors how biology actually works: a disease-causing noncoding variant disrupts a local binding motif, within a conserved enhancer, that regulates a gene in a specific tissue. A model must see all three levels at once.

---

## 8. References

- Avsec et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods*.
- Zhou et al. (2023). DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. *arXiv*.
- Nguyen et al. (2024). HyenaDNA: Long-Range Genomic Sequence Modeling. *ICML*.
- Zoonomia Consortium (2020). A comparative genomics multitool for scientific discovery. *Nature*.
- Benegas et al. (2023). DNA language models are powerful predictors of genome-wide variant effects. *PNAS*.
