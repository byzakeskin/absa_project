"""
src/training/train_absa.py
---------------------------
ABSA modeli eğitim scripti.
Aspect Term Extraction + Aspect Sentiment Classification modellerini eğitir.

Kullanım:
    python src/training/train_absa.py --config configs/model_config.yaml
"""

import argparse
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm


# ── Dataset Sınıfları ─────────────────────────────────────────────────────────

@dataclass
class ABSASample:
    """Tek bir ABSA örneği."""
    text: str
    aspect: str
    sentiment: str  # "POS", "NEG", "NEU"
    language: str   # "tr", "en"


class ABSADataset(Dataset):
    """
    Aspect Sentiment Classification veri seti.
    
    JSON formatı:
    [
        {
            "text": "Pantolonun kumaşı çok kaliteliydi ama beden büyük geldi.",
            "aspect": "kumaş kalitesi",
            "sentiment": "POS",
            "language": "tr"
        },
        ...
    ]
    """

    LABEL_MAP = {"POS": 0, "NEG": 1, "NEU": 2}

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        augment: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples = [ABSASample(**item) for item in raw_data]
        print(f"✓ {len(self.samples)} örnek yüklendi: {data_path}")

        # Sınıf dağılımı
        sentiments = [s.sentiment for s in self.samples]
        for label in ["POS", "NEG", "NEU"]:
            count = sentiments.count(label)
            print(f"  {label}: {count} ({count/len(sentiments)*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # BERT pair encoding: [CLS] text [SEP] aspect [SEP]
        encoding = self.tokenizer(
            sample.text,
            sample.aspect,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label = self.LABEL_MAP[sample.sentiment]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "text": sample.text,
            "aspect": sample.aspect,
        }


# ── Trainer ──────────────────────────────────────────────────────────────────

class ABSATrainer:
    """
    ABSA model eğitim döngüsü.
    Early stopping, learning rate scheduling ve WandB logging destekler.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: Dict,
        output_dir: str = "checkpoints/absa",
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥  Cihaz: {self.device}")
        self.model.to(self.device)

        # DataLoader'lar
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"].get("num_workers", 2),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"] * 2,
            shuffle=False,
            num_workers=config["training"].get("num_workers", 2),
        )

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config["training"]["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(params, lr=config["training"]["learning_rate"])

        # LR Scheduler
        total_steps = len(self.train_loader) * config["training"]["num_epochs"]
        warmup_steps = int(total_steps * config["training"].get("warmup_ratio", 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Sınıf ağırlıkları (dengesiz veri için)
        class_weights = config["training"].get("class_weights")
        if class_weights:
            weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = config["training"].get("early_stopping_patience", 3)

        # WandB (opsiyonel)
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project="absa-ecommerce", config=config)
                self.wandb = wandb
            except ImportError:
                print("⚠️  wandb bulunamadı. pip install wandb")
                self.use_wandb = False

    def train(self) -> Dict:
        """Tam eğitim döngüsü."""
        num_epochs = self.config["training"]["num_epochs"]
        print(f"\n🚀 Eğitim başlıyor — {num_epochs} epoch, {len(self.train_loader)} batch/epoch\n")

        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_accuracy": []}

        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("─" * 50)

            # Eğitim adımı
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)

            # Doğrulama adımı
            val_metrics = self._evaluate(self.val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_metrics["macro_f1"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val F1:     {val_metrics['macro_f1']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")

            if self.use_wandb:
                self.wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                })

            # En iyi model kaydet
            if val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                self._save_model(epoch, val_metrics)
                print(f"  ✅ Yeni en iyi model kaydedildi! F1: {self.best_val_f1:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"  ⏳ İyileşme yok ({self.patience_counter}/{self.early_stopping_patience})")

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⛔ Early stopping: {self.early_stopping_patience} epoch boyunca iyileşme olmadı.")
                break

            print()

        print(f"\n✅ Eğitim tamamlandı. En iyi Val F1: {self.best_val_f1:.4f}")
        return history

    def _train_epoch(self, epoch: int) -> float:
        """Tek bir eğitim epoch'u."""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"  Epoch {epoch} [Train]", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs["loss"] if "loss" in outputs else self.loss_fn(outputs["logits"], labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def _evaluate(self, data_loader: DataLoader) -> Dict:
        """Model değerlendirme."""
        self.model.eval()
        all_labels = []
        all_preds = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="  [Eval]", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs["logits"]
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        label_names = ["POS", "NEG", "NEU"]
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
        accuracy = accuracy_score(all_labels, all_preds)

        return {
            "loss": total_loss / len(data_loader),
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": accuracy,
            "report": classification_report(all_labels, all_preds, target_names=label_names),
        }

    def _save_model(self, epoch: int, metrics: Dict):
        """En iyi modeli kaydet."""
        save_path = self.output_dir / "best_model"
        save_path.mkdir(exist_ok=True)

        torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")

        metadata = {
            "epoch": epoch,
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "config": self.config,
        }
        with open(save_path / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


# ── Ana Script ────────────────────────────────────────────────────────────────

def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("ABSA MODEL EĞİTİMİ")
    print(f"Model: {config['model']['name']}")
    print(f"Dil  : {config['model']['language']}")
    print("=" * 60)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    # Dataset'ler
    train_dataset = ABSADataset(
        data_path=config["data"]["train_path"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        augment=config["training"].get("augment", False),
    )
    val_dataset = ABSADataset(
        data_path=config["data"]["val_path"],
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
    )

    # Model import
    import sys
    sys.path.append(".")
    from src.models.aspect_extractor import AspectSentimentClassifier

    model = AspectSentimentClassifier(
        bert_model_name=config["model"]["name"],
        num_labels=3,
        dropout=config["model"].get("dropout", 0.1),
    )

    # Eğitim
    trainer = ABSATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        output_dir=config["training"]["output_dir"],
    )
    history = trainer.train()

    # Sonuçları kaydet
    with open(Path(config["training"]["output_dir"]) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA Model Eğitimi")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    args = parser.parse_args()
    main(args.config)
