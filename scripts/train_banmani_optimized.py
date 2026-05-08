"""BanMANI optimized trainer (8GB VRAM friendly).

Applies the requested VRAM optimizations:
1) Base variants: xlm-roberta-base + google/mt5-base
2) Gradient checkpointing on both encoders
3) AMP mixed precision + GradScaler
4) batch_size=4 + gradient_accumulation_steps=8 (effective 32)
5) AdamW 8-bit via bitsandbytes if available, else torch.optim.AdamW
6) torch.cuda.empty_cache() between validation steps
7) max_seq_length=128 (cls) and 256 (span)
8) Freeze bottom 6 layers of each encoder (fine-tune top layers only)
9) VRAM usage monitor each epoch (torch.cuda.memory_allocated)
10) Keep the same BanMANIModel class structure and forward() signature

Run:
  python scripts/train_banmani_optimized.py

Notes:
- This script keeps the original multi-task heads, but span BIO supervision is a
  placeholder (all-O tags with PAD ignored), matching the notebook's approach.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# -------------------------
# Reproducibility / Device
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(use_gpu: bool = True) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def log_vram(prefix: str = "") -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    p = f"{prefix} " if prefix else ""
    print(f"{p}VRAM allocated: {allocated:.1f} MiB | reserved: {reserved:.1f} MiB | peak: {peak:.1f} MiB")


# -------------------------
# Text normalization
# -------------------------

def normalize_bangla_text(text: str) -> str:
    import re
    import unicodedata

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200B\u200C\u200D\u180E\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------
# Dataset loading / labels
# -------------------------

def load_banmani_csv(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        encoding="utf-8",
        engine="python",  # required for this dataset
        on_bad_lines="warn",
    )

    if "mani_news" in df.columns and "original_news_article" in df.columns:
        df = df.rename(columns={"mani_news": "social_item", "original_news_article": "reference_article"})

    required_cols = {"data_type", "mani_status", "reference_article", "social_item"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    train_df = df.query("data_type == 'TRAIN'").reset_index(drop=True)
    test_df = df.query("data_type == 'TEST'").reset_index(drop=True)

    return train_df, test_df


def create_labels(row: pd.Series) -> Dict[str, object]:
    is_manipulated = 1 if row.get("mani_status") == "MANI" else 0

    altered = row.get("altered_excerpt")
    original = row.get("original_excerpt")

    altered = altered if pd.notna(altered) else "none"
    original = original if pd.notna(original) else "none"

    return {
        "label": is_manipulated,
        "altered_excerpt": str(altered).strip(),
        "original_excerpt": str(original).strip(),
    }


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["reference_article", "social_item", "altered_excerpt", "original_excerpt"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_bangla_text)
    return df


class BanMANIDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer_cls,
        tokenizer_span,
        max_length_cls: int = 128,
        max_length_span: int = 256,
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer_cls = tokenizer_cls
        self.tokenizer_span = tokenizer_span
        self.max_length_cls = max_length_cls
        self.max_length_span = max_length_span

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.data.iloc[idx]
        reference = row["reference_article"]
        social = row["social_item"]

        # Keep the notebook's convention.
        input_text = f"{reference} [SEP] {social}"

        encoded_cls = self.tokenizer_cls(
            input_text,
            max_length=self.max_length_cls,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoded_span = self.tokenizer_span(
            input_text,
            max_length=self.max_length_span,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids_cls": encoded_cls["input_ids"].squeeze(0),
            "attention_mask_cls": encoded_cls["attention_mask"].squeeze(0),
            "input_ids_span": encoded_span["input_ids"].squeeze(0),
            "attention_mask_span": encoded_span["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "altered_excerpt": row.get("altered_excerpt", "none"),
            "original_excerpt": row.get("original_excerpt", "none"),
            "reference_article": reference,
            "social_item": social,
        }


# -------------------------
# Model (keep forward signature)
# -------------------------


class BanMANIModel(nn.Module):
    """Multi-task model for BanMANI with separate encoders.

    Keep the same structure and forward() signature as the notebook.
    """

    def __init__(self, cls_model_name: str, span_model_name: str, num_classes: int = 2):
        super().__init__()

        # Separate encoders
        self.encoder_cls = AutoModel.from_pretrained(cls_model_name)
        self.encoder_span = AutoModel.from_pretrained(span_model_name)

        cls_hidden = getattr(self.encoder_cls.config, "hidden_size", None)
        if cls_hidden is None:
            raise ValueError("Classification encoder config missing hidden_size")

        # T5/MT5 uses d_model (some versions expose hidden_size alias; support both).
        span_hidden = getattr(self.encoder_span.config, "hidden_size", None)
        if span_hidden is None:
            span_hidden = getattr(self.encoder_span.config, "d_model", None)
        if span_hidden is None:
            raise ValueError("Span encoder config missing hidden_size/d_model")

        # Task 1: Classification head (uses first token of encoder_cls)
        self.classification_head = nn.Sequential(
            nn.Linear(cls_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # Task 2: Token-level classification for altered excerpt (BIO tagging)
        self.span_head_altered = nn.Sequential(
            nn.Linear(span_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
        )

        # Task 3: Token-level classification for original excerpt (BIO tagging)
        self.span_head_original = nn.Sequential(
            nn.Linear(span_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),
        )

    def forward(self, input_ids_cls, attention_mask_cls, input_ids_span, attention_mask_span):
        # Classification encoder
        encoder_output_cls = self.encoder_cls(
            input_ids=input_ids_cls,
            attention_mask=attention_mask_cls,
            return_dict=True,
        )
        cls_output = encoder_output_cls.last_hidden_state[:, 0, :]
        classification_logits = self.classification_head(cls_output)

        # Span encoder (use MT5/T5 encoder stack directly to avoid decoder allocation)
        if hasattr(self.encoder_span, "encoder"):
            encoder_output_span = self.encoder_span.encoder(
                input_ids=input_ids_span,
                attention_mask=attention_mask_span,
                return_dict=True,
            )
            sequence_output = encoder_output_span.last_hidden_state
        else:
            encoder_output_span = self.encoder_span(
                input_ids=input_ids_span,
                attention_mask=attention_mask_span,
                return_dict=True,
            )
            sequence_output = encoder_output_span.last_hidden_state

        span_logits_altered = self.span_head_altered(sequence_output)
        span_logits_original = self.span_head_original(sequence_output)

        return {
            "classification_logits": classification_logits,
            "span_logits_altered": span_logits_altered,
            "span_logits_original": span_logits_original,
        }


# -------------------------
# VRAM optimizations
# -------------------------


def enable_gradient_checkpointing(model: BanMANIModel) -> None:
    # Encoder 1
    if hasattr(model.encoder_cls, "gradient_checkpointing_enable"):
        model.encoder_cls.gradient_checkpointing_enable()

    # Encoder 2
    if hasattr(model.encoder_span, "gradient_checkpointing_enable"):
        model.encoder_span.gradient_checkpointing_enable()

    # Required for checkpointing in T5/MT5-style models.
    if hasattr(model.encoder_span, "config") and hasattr(model.encoder_span.config, "use_cache"):
        model.encoder_span.config.use_cache = False


def freeze_bottom_layers(model: BanMANIModel, n_freeze: int = 6) -> None:
    # XLM-R (encoder layers)
    if hasattr(model.encoder_cls, "embeddings"):
        for p in model.encoder_cls.embeddings.parameters():
            p.requires_grad = False

    if hasattr(model.encoder_cls, "encoder") and hasattr(model.encoder_cls.encoder, "layer"):
        layers = model.encoder_cls.encoder.layer
        for layer in layers[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False

    # MT5 (encoder blocks)
    if hasattr(model.encoder_span, "shared"):
        for p in model.encoder_span.shared.parameters():
            p.requires_grad = False

    if hasattr(model.encoder_span, "encoder") and hasattr(model.encoder_span.encoder, "block"):
        blocks = model.encoder_span.encoder.block
        for block in blocks[:n_freeze]:
            for p in block.parameters():
                p.requires_grad = False


def build_optimizer(trainable_params, lr: float, weight_decay: float):
    # Prefer 8-bit AdamW if bitsandbytes is installed.
    try:
        import bitsandbytes as bnb  # type: ignore

        print("Using bitsandbytes AdamW8bit optimizer")
        return bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=weight_decay)
    except Exception as e:
        print(f"bitsandbytes not available -> using torch.optim.AdamW ({type(e).__name__}: {e})")
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


# -------------------------
# Training / Evaluation
# -------------------------


@dataclass
class TrainConfig:
    seed: int = 42
    use_gpu: bool = True

    cls_model_name: str = "xlm-roberta-base"
    span_model_name: str = "google/mt5-base"

    max_length_cls: int = 128
    max_length_span: int = 256

    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    span_loss_weight: float = 0.3
    max_grad_norm: float = 1.0


def evaluate(model: BanMANIModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Validating", leave=False)):
            input_ids_cls = batch["input_ids_cls"].to(device)
            attention_mask_cls = batch["attention_mask_cls"].to(device)
            input_ids_span = batch["input_ids_span"].to(device)
            attention_mask_span = batch["attention_mask_span"].to(device)
            labels = batch["label"].to(device)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids_cls, attention_mask_cls, input_ids_span, attention_mask_span)
                logits = outputs["classification_logits"]

            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # (6) Explicit cache clearing between validation steps.
            if device.type == "cuda":
                torch.cuda.empty_cache()

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    # Compute metrics (avoid crashing if a class is missing in preds)
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
    except Exception:
        acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
        return {"accuracy": acc, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def train() -> None:
    cfg = TrainConfig()

    set_seed(cfg.seed)
    device = get_device(cfg.use_gpu)

    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory / 1e9:.2f} GB")

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "Data" / "BanMANI.csv"

    print(f"Loading dataset: {data_path}")
    train_df, test_df = load_banmani_csv(data_path)

    # Labels
    train_labels = [create_labels(r) for _, r in train_df.iterrows()]
    test_labels = [create_labels(r) for _, r in test_df.iterrows()]

    for df, labels in [(train_df, train_labels), (test_df, test_labels)]:
        df["label"] = [l["label"] for l in labels]
        df["altered_excerpt"] = [l["altered_excerpt"] for l in labels]
        df["original_excerpt"] = [l["original_excerpt"] for l in labels]

    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    print("Label distribution (train):")
    print(train_df["label"].value_counts())

    # Tokenizers
    tokenizer_cls = AutoTokenizer.from_pretrained(cfg.cls_model_name, use_fast=True)
    tokenizer_span = AutoTokenizer.from_pretrained(cfg.span_model_name, use_fast=True)

    # Data
    train_dataset = BanMANIDataset(
        train_df,
        tokenizer_cls,
        tokenizer_span,
        max_length_cls=cfg.max_length_cls,
        max_length_span=cfg.max_length_span,
    )
    test_dataset = BanMANIDataset(
        test_df,
        tokenizer_cls,
        tokenizer_span,
        max_length_cls=cfg.max_length_cls,
        max_length_span=cfg.max_length_span,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = BanMANIModel(cfg.cls_model_name, cfg.span_model_name)
    enable_gradient_checkpointing(model)
    freeze_bottom_layers(model, n_freeze=6)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)

    print(f"Model encoders: {cfg.cls_model_name} + {cfg.span_model_name}")
    print(f"Total params: {total_params:,} | Trainable params: {trainable_count:,}")

    # Optimizer + scheduler
    optimizer = build_optimizer(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Optimizer steps (not micro-batches)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Losses
    cls_loss_fn = nn.CrossEntropyLoss()
    bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # AMP scaler
    scaler = GradScaler(enabled=(device.type == "cuda"))

    print("Training config:")
    print(
        f"epochs={cfg.epochs} batch_size={cfg.batch_size} grad_accum={cfg.gradient_accumulation_steps} "
        f"(effective={cfg.batch_size * cfg.gradient_accumulation_steps}) max_len_cls={cfg.max_length_cls} "
        f"max_len_span={cfg.max_length_span}"
    )

    for epoch in range(cfg.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_acc = 0.0
        seen_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for step, batch in enumerate(progress):
            input_ids_cls = batch["input_ids_cls"].to(device)
            attention_mask_cls = batch["attention_mask_cls"].to(device)
            input_ids_span = batch["input_ids_span"].to(device)
            attention_mask_span = batch["attention_mask_span"].to(device)
            labels = batch["label"].to(device)

            # Dummy BIO labels: all 'O' (0) with PAD ignored
            bio_targets = torch.zeros_like(input_ids_span, dtype=torch.long)
            bio_targets = bio_targets.masked_fill(attention_mask_span == 0, -100)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids_cls, attention_mask_cls, input_ids_span, attention_mask_span)

                classification_loss = cls_loss_fn(outputs["classification_logits"], labels)

                span_logits_altered = outputs["span_logits_altered"]
                span_logits_original = outputs["span_logits_original"]

                span_loss_altered = bio_loss_fn(span_logits_altered.reshape(-1, 3), bio_targets.reshape(-1))
                span_loss_original = bio_loss_fn(span_logits_original.reshape(-1, 3), bio_targets.reshape(-1))

                loss = classification_loss + cfg.span_loss_weight * (span_loss_altered + span_loss_original)
                loss = loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            # Metrics (classification only)
            with torch.no_grad():
                preds = torch.argmax(outputs["classification_logits"], dim=-1)
                acc = (preds == labels).float().mean().item()

            running_loss += loss.item() * cfg.gradient_accumulation_steps
            running_acc += acc
            seen_batches += 1

            do_step = ((step + 1) % cfg.gradient_accumulation_steps == 0) or (step + 1 == len(train_loader))
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            progress.set_postfix(loss=running_loss / seen_batches, acc=running_acc / seen_batches, lr=scheduler.get_last_lr()[0])

        # (9) VRAM usage each epoch
        log_vram(prefix=f"Epoch {epoch+1}")

        # Validation
        metrics = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch+1} done | train_loss={running_loss/seen_batches:.4f} train_acc={running_acc/seen_batches:.4f} "
            f"| val_acc={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f}"
        )

        # (6) Cache clearing after validation
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("✓ Training completed")


if __name__ == "__main__":
    train()
