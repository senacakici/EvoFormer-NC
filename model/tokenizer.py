"""
Multi-Scale DNA Tokenizer for EvoFormer-NC.

Tokenizes DNA sequences at three resolutions simultaneously:
  - Local  (6-mer,  ~10 bp)  : transcription factor binding motifs
  - Regional (64-mer, ~200 bp): nucleosome positioning, CpG islands
  - Macro  (512-mer, ~1.5 kb): enhancer-promoter interactions
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


# ── Vocabulary ────────────────────────────────────────────────────────────────

DNA_CHARS = ["A", "T", "G", "C", "N"]  # N = unknown/masked
SPECIAL_TOKENS = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3}


def build_kmer_vocab(k: int) -> Dict[str, int]:
    """Build a vocabulary of all k-mers over {A, T, G, C, N}."""
    from itertools import product
    vocab = {**SPECIAL_TOKENS}
    idx = len(vocab)
    for kmer in product(DNA_CHARS, repeat=k):
        vocab["".join(kmer)] = idx
        idx += 1
    return vocab


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class MultiScaleTokenizer:
    """
    Tokenizes a raw DNA string into three resolutions.

    Example
    -------
    >>> tokenizer = MultiScaleTokenizer()
    >>> tokens = tokenizer.encode("ATGCATGCATGCATGC")
    >>> tokens["local"].shape       # (seq_len_local,)
    >>> tokens["regional"].shape    # (seq_len_regional,)
    >>> tokens["macro"].shape       # (seq_len_macro,)
    """

    SCALE_CONFIG = {
        "local":    {"k": 6,   "stride": 1,   "context_bp": 10},
        "regional": {"k": 6,   "stride": 8,   "context_bp": 200},
        "macro":    {"k": 6,   "stride": 64,  "context_bp": 1500},
    }

    def __init__(self, scales: List[str] = None, max_len: int = 131_072):
        self.scales = scales or ["local", "regional", "macro"]
        self.max_len = max_len
        self.vocab = build_kmer_vocab(k=6)
        self.vocab_size = len(self.vocab)
        self.pad_id = SPECIAL_TOKENS["[PAD]"]
        self.mask_id = SPECIAL_TOKENS["[MASK]"]

    # ── public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        sequence: str,
        return_tensors: bool = True,
    ) -> Dict[str, torch.Tensor | List[int]]:
        """
        Encode a DNA string into multi-scale token ids.

        Parameters
        ----------
        sequence : str
            Raw DNA sequence (upper-case A/T/G/C/N).
        return_tensors : bool
            If True, return torch.LongTensor; else return lists.

        Returns
        -------
        dict with keys "local", "regional", "macro".
        """
        sequence = sequence.upper().strip()
        output = {}
        for scale in self.scales:
            cfg = self.SCALE_CONFIG[scale]
            ids = self._kmer_tokenize(sequence, k=cfg["k"], stride=cfg["stride"])
            if return_tensors:
                ids = torch.tensor(ids, dtype=torch.long)
            output[scale] = ids
        return output

    def encode_batch(
        self,
        sequences: List[str],
        padding: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of sequences, optionally padding to equal length."""
        batch = [self.encode(seq) for seq in sequences]
        if not padding:
            return {scale: [b[scale] for b in batch] for scale in self.scales}

        padded = {}
        for scale in self.scales:
            tensors = [b[scale] for b in batch]
            max_len = max(t.size(0) for t in tensors)
            padded[scale] = torch.stack(
                [self._pad(t, max_len) for t in tensors]
            )
        return padded

    def decode(self, ids: torch.Tensor, scale: str = "local") -> str:
        """Decode token ids back to an approximate DNA string."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        kmers = [inv_vocab.get(int(i), "N" * 6) for i in ids]
        # Stitch k-mers (overlapping by k-1 for stride=1)
        cfg = self.SCALE_CONFIG[scale]
        stride = cfg["stride"]
        if stride == 1:
            seq = kmers[0] + "".join(km[-1] for km in kmers[1:])
        else:
            seq = "".join(km[:stride] for km in kmers)
        return seq

    # ── internals ─────────────────────────────────────────────────────────────

    def _kmer_tokenize(self, seq: str, k: int, stride: int) -> List[int]:
        ids = []
        for i in range(0, len(seq) - k + 1, stride):
            kmer = seq[i: i + k]
            ids.append(self.vocab.get(kmer, self.vocab.get("N" * k, 1)))
        return ids

    def _pad(self, tensor: torch.Tensor, length: int) -> torch.Tensor:
        pad_len = length - tensor.size(0)
        if pad_len <= 0:
            return tensor[:length]
        return torch.cat([tensor, torch.full((pad_len,), self.pad_id)])

    # ── properties ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MultiScaleTokenizer("
            f"scales={self.scales}, "
            f"vocab_size={self.vocab_size:,})"
        )
