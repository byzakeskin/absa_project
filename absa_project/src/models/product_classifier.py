"""
src/models/product_classifier.py
----------------------------------
Ürün kategorisi tanımlama modeli.
BERT/BERTurk tabanlı metin sınıflandırıcı.

Desteklenen kategoriler:
    0: pantolon_jean
    1: mont_kaban
    2: elbise_etek
    3: tisort_gomlek_bluz
    4: ayakkabi
    5: terlik_sandalet_cizme
    6: canta
"""

import torch
import torch.nn as nn
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from typing import List, Dict, Optional, Union
import yaml
import re
from pathlib import Path

# Label mapping
CATEGORY_LABELS = {
    0: "pantolon_jean",
    1: "mont_kaban",
    2: "elbise_etek",
    3: "tisort_gomlek_bluz",
    4: "ayakkabi",
    5: "terlik_sandalet_cizme",
    6: "canta",
}

LABEL_TO_ID = {v: k for k, v in CATEGORY_LABELS.items()}

# Pretrained model seçenekleri
PRETRAINED_MODELS = {
    "tr": "dbmdz/bert-base-turkish-cased",        # BERTurk
    "en": "bert-base-uncased",                      # BERT base
    "multilingual": "bert-base-multilingual-cased"  # mBERT
}


class ProductClassifier(nn.Module):
    """
    Ürün kategorisi sınıflandırıcı.
    BERT tabanlı sequence classification.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        num_labels: int = 7,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.model_name = model_name

        # BERT encoder
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)


class KeywordBasedClassifier:
    """
    Kural tabanlı ürün kategori sınıflandırıcı.
    Model eğitilmeden önce baseline veya fallback olarak kullanılır.
    """

    def __init__(self, config_path: str = "configs/aspect_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.categories = config["product_categories"]
        self._build_keyword_map()

    def _build_keyword_map(self):
        """Keyword → category eşleme sözlüğü oluştur."""
        self.keyword_map = {}
        for cat in self.categories:
            cat_id = cat["id"]
            for kw in cat.get("keywords_tr", []):
                self.keyword_map[kw.lower()] = cat_id
            for kw in cat.get("keywords_en", []):
                self.keyword_map[kw.lower()] = cat_id

    def classify(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Metni anahtar kelime eşleştirmesiyle sınıflandır.

        Returns:
            {"category": str, "confidence": float, "method": "keyword"}
        """
        text_lower = text.lower()
        scores = {}

        for keyword, category in self.keyword_map.items():
            # Kelime sınırlarına dikkat et
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                scores[category] = scores.get(category, 0) + matches

        if not scores:
            return {"category": "unknown", "confidence": 0.0, "method": "keyword"}

        best_category = max(scores, key=scores.get)
        total_matches = sum(scores.values())
        confidence = scores[best_category] / total_matches

        return {
            "category": best_category,
            "confidence": round(confidence, 3),
            "method": "keyword",
            "all_scores": scores,
        }


class ProductClassifierInference:
    """
    Eğitilmiş modeli kullanarak inference yapan wrapper sınıfı.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "bert-base-multilingual-cased",
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_path else model_name
        )

        if model_path:
            self.model = ProductClassifier(model_name=model_name)
            self.model.load_state_dict(
                torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device)
            )
        else:
            self.model = ProductClassifier(model_name=model_name)

        self.model.to(self.device)
        self.model.eval()

        # Fallback olarak keyword-based classifier
        self.keyword_classifier = None  # config varsa initialize et

    def predict(self, text: str, max_length: int = 128) -> Dict:
        """
        Tek metin için kategori tahmini.

        Args:
            text: Yorum veya ürün açıklaması metni
            max_length: Max token sayısı

        Returns:
            {
                "category": "pantolon_jean",
                "category_id": 0,
                "confidence": 0.95,
                "all_probabilities": {...}
            }
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)[0]

        predicted_id = torch.argmax(probs).item()
        confidence = probs[predicted_id].item()

        return {
            "category": CATEGORY_LABELS[predicted_id],
            "category_id": predicted_id,
            "confidence": round(confidence, 4),
            "all_probabilities": {
                CATEGORY_LABELS[i]: round(p.item(), 4)
                for i, p in enumerate(probs)
            },
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Toplu metin tahmini."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encoding = self.tokenizer(
                batch,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1)

            for j, prob in enumerate(probs):
                pred_id = torch.argmax(prob).item()
                results.append({
                    "text": batch[j],
                    "category": CATEGORY_LABELS[pred_id],
                    "category_id": pred_id,
                    "confidence": round(prob[pred_id].item(), 4),
                })
        return results


# ── Kullanım örneği ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Keyword-based classifier (model eğitime gerek yok)
    clf = KeywordBasedClassifier(config_path="configs/aspect_config.yaml")

    test_texts = [
        "Aldığım pantolonun kumaşı çok güzel, beden uyumu mükemmel",
        "The jacket is very warm and fits perfectly for winter",
        "Ayakkabı dar geldi, numara büyük alınmalı",
        "Elbise çok şık ama biraz şeffaf",
        "Çanta kaliteli görünüyor, fermuar sağlam",
    ]

    print("Keyword-Based Classifier Sonuçları:")
    print("-" * 60)
    for text in test_texts:
        result = clf.classify(text)
        print(f"Metin  : {text[:50]}")
        print(f"Kategori: {result['category']} (Güven: {result['confidence']:.2f})")
        print()
