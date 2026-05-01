"""
EvoFormer-NC — Main Training Script

Usage
-----
    python train/train.py --config train/config.yaml

Requires PyTorch Lightning. Set DATA_DIR and OUTPUT_DIR in config.yaml.
"""

from __future__ import annotations
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from model.tokenizer import MultiScaleTokenizer
from model.encoder import EvoFormerEncoder
from model.evo_embeddings import ConservationEmbedding
from model.variant_head import VariantEffectHead
from train.losses import EvoFormerLoss


# ── Placeholder Dataset ───────────────────────────────────────────────────────
# Replace with your actual data loading logic.

class VariantDataset(Dataset):
    """
    Minimal dataset stub. Each item contains:
      - ref_seq   : reference DNA sequence (string)
      - alt_seq   : alternate DNA sequence (string)
      - var_pos   : variant position (int)
      - conservation : (L,) float array of PhyloP scores
      - impact_label  : float in [0, 1] (0 = benign, 1 = pathogenic)
      - tissue_labels : (n_tissues,) binary vector
      - mechanism_label : int (class index)
    """

    def __init__(self, records: list, tokenizer: MultiScaleTokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        ref_tokens = self.tokenizer.encode(r["ref_seq"])
        alt_tokens = self.tokenizer.encode(r["alt_seq"])
        return {
            "ref_tokens":       ref_tokens,
            "alt_tokens":       alt_tokens,
            "variant_pos":      torch.tensor(r["var_pos"], dtype=torch.long),
            "conservation":     torch.tensor(r["conservation"], dtype=torch.float),
            "impact_label":     torch.tensor(r["impact_label"], dtype=torch.float),
            "tissue_labels":    torch.tensor(r["tissue_labels"], dtype=torch.float),
            "mechanism_label":  torch.tensor(r["mechanism_label"], dtype=torch.long),
        }


# ── Lightning Module ──────────────────────────────────────────────────────────

class EvoFormerNC(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        c = cfg["model"]

        self.tokenizer  = MultiScaleTokenizer()
        self.encoder    = EvoFormerEncoder(
            vocab_size = self.tokenizer.vocab_size,
            d_model    = c["d_model"],
            n_heads    = c["n_heads"],
            n_layers   = c["n_layers"],
            ffn_dim    = c["ffn_dim"],
            dropout    = c["dropout"],
        )
        self.evo_emb    = ConservationEmbedding(d_model=c["d_model"])
        self.head       = VariantEffectHead(
            d_model   = c["d_model"],
            n_tissues = c["n_tissues"],
        )
        self.loss_fn    = EvoFormerLoss(
            w_impact    = cfg["loss"]["w_impact"],
            w_tissue    = cfg["loss"]["w_tissue"],
            w_mechanism = cfg["loss"]["w_mechanism"],
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def _encode(self, tokens, conservation):
        """Run encoder + conservation injection for one allele."""
        representations = self.encoder(tokens)
        # Inject conservation into the local-scale representation
        representations["local"] = self.evo_emb(
            representations["local"], conservation
        )
        return representations

    def forward(self, batch):
        ref_repr = self._encode(batch["ref_tokens"], batch["conservation"])
        alt_repr = self._encode(batch["alt_tokens"], batch["conservation"])
        return self.head(ref_repr, alt_repr, batch["variant_pos"])

    # ── steps ─────────────────────────────────────────────────────────────────

    def _shared_step(self, batch, stage: str):
        preds = self(batch)
        loss, loss_dict = self.loss_fn(
            impact_pred     = preds["impact_score"],
            tissue_logits   = preds["tissue_logits"],
            mechanism_logits= preds["mechanism_logits"],
            impact_label    = batch["impact_label"],
            tissue_labels   = batch["tissue_labels"],
            mechanism_label = batch["mechanism_label"],
        )
        self.log(f"{stage}/loss",      loss,                  prog_bar=True)
        self.log(f"{stage}/loss_impact",    loss_dict["impact"])
        self.log(f"{stage}/loss_tissue",    loss_dict["tissue"])
        self.log(f"{stage}/loss_mechanism", loss_dict["mechanism"])
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    # ── optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr           = self.hparams["training"]["lr"],
            weight_decay = self.hparams["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max = self.hparams["training"]["max_epochs"],
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Data (replace with real loaders) ──────────────────────────────────────
    tokenizer = MultiScaleTokenizer()
    train_ds = VariantDataset(records=[], tokenizer=tokenizer)  # TODO: load records
    val_ds   = VariantDataset(records=[], tokenizer=tokenizer)

    train_dl = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["training"]["num_workers"],
    )
    val_dl = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["training"]["num_workers"],
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = EvoFormerNC(cfg)

    # ── Callbacks ──────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath   = cfg["output"]["checkpoint_dir"],
            filename  = "evoformer-nc-{epoch:02d}-{val/loss:.4f}",
            monitor   = "val/loss",
            save_top_k = 3,
        ),
        EarlyStopping(monitor="val/loss", patience=10),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── Logger ─────────────────────────────────────────────────────────────────
    logger = None
    if cfg.get("wandb", {}).get("enabled", False):
        logger = WandbLogger(
            project = cfg["wandb"]["project"],
            name    = cfg["wandb"]["run_name"],
        )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs      = cfg["training"]["max_epochs"],
        accelerator     = "auto",
        devices         = "auto",
        precision       = cfg["training"].get("precision", "16-mixed"),
        callbacks       = callbacks,
        logger          = logger,
        gradient_clip_val = cfg["training"].get("grad_clip", 1.0),
        log_every_n_steps = 10,
    )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
