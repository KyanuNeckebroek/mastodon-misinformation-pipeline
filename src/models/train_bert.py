"""
src/models/train_bert.py

Fine-tunet bert-base-uncased voor binaire misinformatie-detectie.
Vereist: GPU (CUDA), PyTorch, Hugging Face Transformers

Gebruik:
    python src/models/train_bert.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import load_config, setup_logging, ensure_dirs, set_seed, get_device
from src.evaluation.metrics import compute_metrics, print_classification_report


class MisinformatieDataset(Dataset):
    """PyTorch Dataset voor getokeniseerde teksten."""

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get("token_type_ids", torch.zeros_like(
                self.encodings["input_ids"]))[idx],
            "labels": self.labels[idx],
        }


def load_data(config: dict) -> tuple:
    """Laad en valideer de gecombineerde dataset."""
    proc_dir = config["preprocessing"]["processed_dir"]

    train_df = pd.read_csv(f"{proc_dir}train_combined.csv").dropna(subset=["clean_text"])
    val_df   = pd.read_csv(f"{proc_dir}val_combined.csv").dropna(subset=["clean_text"])

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    return (
        train_df["clean_text"].tolist(), train_df["label"].tolist(),
        val_df["clean_text"].tolist(),   val_df["label"].tolist(),
    )


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    use_fp16: bool,
) -> float:
    """Eén trainingsepoche. Geeft gemiddeld verlies terug."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        batch = {k: v.to(device) for k, v in batch.items()}

        if use_fp16:
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

        total_loss += outputs.loss.item()

        if (step + 1) % grad_accum_steps == 0:
            if use_fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list, list, list]:
    """Evalueer het model op een dataset. Geeft (labels, pred, probs) terug."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(dataloader, desc="Evaluatie", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        all_labels.extend(batch["labels"].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return all_labels, all_preds, all_probs


def train_bert(config: dict) -> None:
    """Volledige BERT fine-tuning pipeline."""
    set_seed(config["project"]["random_seed"])
    bert_cfg = config["bert"]
    out_dir = bert_cfg["output_dir"]
    model_path = bert_cfg["model_path"]
    ensure_dirs(out_dir, model_path)

    device = get_device()
    use_fp16 = bert_cfg["fp16"] and device.type == "cuda"

    logger.info(f"BERT model: {bert_cfg['model_name']}")
    logger.info(f"Device: {device}, FP16: {use_fp16}")

    # Tokenizer en data
    logger.info("Tokenizer laden...")
    tokenizer = BertTokenizerFast.from_pretrained(bert_cfg["model_name"])

    X_train, y_train, X_val, y_val = load_data(config)

    logger.info("Datasets tokeniseren...")
    train_dataset = MisinformatieDataset(X_train, y_train, tokenizer, bert_cfg["max_length"])
    val_dataset   = MisinformatieDataset(X_val, y_val, tokenizer, bert_cfg["max_length"])

    train_loader = DataLoader(
        train_dataset, batch_size=bert_cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bert_cfg["batch_size"] * 2, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    logger.info("BERT model laden...")
    model = BertForSequenceClassification.from_pretrained(
        bert_cfg["model_name"],
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = model.to(device)

    # Optimizer en scheduler
    total_steps = len(train_loader) * bert_cfg["num_epochs"] // bert_cfg["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * bert_cfg["warmup_ratio"])

    optimizer = AdamW(
        model.parameters(),
        lr=bert_cfg["learning_rate"],
        weight_decay=bert_cfg["weight_decay"],
        eps=1e-8,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=use_fp16)

    # Training loop
    best_f1 = 0.0
    training_history = []

    logger.info(f"Training starten: {bert_cfg['num_epochs']} epochs")
    logger.info(f"Totale stappen: {total_steps}, Warmup: {warmup_steps}")

    for epoch in range(1, bert_cfg["num_epochs"] + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch}/{bert_cfg['num_epochs']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, device, bert_cfg["gradient_accumulation_steps"], use_fp16
        )

        y_true, y_pred, y_prob = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_pred, y_prob)

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val F1: {val_metrics['f1_macro']:.4f} | "
                    f"Precision: {val_metrics['precision']:.4f} | "
                    f"Recall: {val_metrics['recall']:.4f}")

        epoch_record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        training_history.append(epoch_record)

        # Sla het beste model op
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info(f"✓ Nieuw beste model opgeslagen (F1: {best_f1:.4f})")

    # Sla trainingsgeschiedenis op
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(f"{out_dir}bert_training_history.csv", index=False)

    # Eindresultaten
    logger.info(f"\n=== BERT EINDRESULTATEN ===")
    logger.info(f"Beste validatie F1: {best_f1:.4f}")
    print_classification_report(y_true, y_pred, model_name="BERT")

    # Sla beste metrics op
    best_metrics = history_df.loc[history_df["f1_macro"].idxmax()].to_dict()
    with open(f"{out_dir}bert_best_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    logger.info(f"Model opgeslagen: {model_path}")


def main():
    config = load_config()
    setup_logging(config["logging"]["log_dir"])

    try:
        import torch
    except ImportError:
        logger.error("PyTorch niet geïnstalleerd. Voer uit: pip install torch")
        sys.exit(1)

    train_bert(config)


if __name__ == "__main__":
    main()
