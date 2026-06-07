"""
src/models/aspect_extractor.py
--------------------------------
Aspect-Based Sentiment Analysis (ABSA) modeli.
İki aşamalı yaklaşım:
  1. Aspect Term Extraction  → BiLSTM + CRF (NER benzeri)
  2. Aspect Sentiment Classification → BERT fine-tuned

Desteklenen sentiment etiketleri: POS (pozitif), NEG (negatif), NEU (nötr)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import numpy as np


# Sentiment label mapping
SENTIMENT_LABELS = {0: "POS", 1: "NEG", 2: "NEU"}
LABEL_TO_SENTIMENT = {"POS": 0, "NEG": 1, "NEU": 2}

# NER etiketleri (BIO şeması)
NER_LABELS = {
    "O": 0,       # Outside (aspect değil)
    "B-ASP": 1,   # Beginning of aspect term
    "I-ASP": 2,   # Inside of aspect term
}


class AspectTermExtractor(nn.Module):
    """
    BERT + BiLSTM + CRF tabanlı aspect term extraction modeli.
    BIO etiketleme şeması ile sequence labeling yapar.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-multilingual-cased",
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_tags: int = 3,  # O, B-ASP, I-ASP
    ):
        super().__init__()
        self.num_tags = num_tags

        # BERT encoder (frozen veya fine-tuned olabilir)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # Genellikle 768

        # BiLSTM katmanı
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        # Projection: BiLSTM çıkışını tag sayısına indir
        self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)

        # CRF katmanı (basit versiyon — production'da torchcrf kullan)
        # pip install pytorch-crf
        try:
            from torchcrf import CRF
            self.crf = CRF(num_tags, batch_first=True)
            self.use_crf = True
        except ImportError:
            print("⚠️  torchcrf bulunamadı. CRF yerine softmax kullanılacak.")
            print("   Kurmak için: pip install pytorch-crf")
            self.use_crf = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        # BERT encoding
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state  # (batch, seq_len, 768)
        sequence_output = self.dropout(sequence_output)

        # BiLSTM
        lengths = attention_mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(sequence_output, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)

        # Emission scores
        emissions = self.hidden2tag(lstm_out)  # (batch, seq_len, num_tags)

        if self.use_crf:
            mask = attention_mask.bool()
            if labels is not None:
                # CRF negatif log-likelihood kaybı
                loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
                return {"loss": loss, "emissions": emissions}
            else:
                # Viterbi decode
                predictions = self.crf.decode(emissions, mask=mask)
                return {"predictions": predictions, "emissions": emissions}
        else:
            # CRF yoksa basit cross-entropy
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(emissions.view(-1, self.num_tags), labels.view(-1))
                return {"loss": loss, "emissions": emissions}
            else:
                predictions = torch.argmax(emissions, dim=-1)
                return {"predictions": predictions.tolist(), "emissions": emissions}

    def extract_aspects(self, tokens: List[str], predictions: List[int]) -> List[str]:
        """BIO etiketlerinden aspect terimlerini çıkar."""
        aspects = []
        current_aspect = []

        for token, label in zip(tokens, predictions):
            if label == NER_LABELS["B-ASP"]:
                if current_aspect:
                    aspects.append(" ".join(current_aspect))
                current_aspect = [token]
            elif label == NER_LABELS["I-ASP"] and current_aspect:
                current_aspect.append(token)
            else:
                if current_aspect:
                    aspects.append(" ".join(current_aspect))
                    current_aspect = []

        if current_aspect:
            aspects.append(" ".join(current_aspect))

        return aspects


class AspectSentimentClassifier(nn.Module):
    """
    BERT tabanlı aspect-level sentiment classification.
    Girdi: [CLS] yorum metni [SEP] aspect terimi [SEP]
    Çıktı: POS / NEG / NEU
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-multilingual-cased",
        num_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

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
        )

        # [CLS] token'ı kullan (global bağlamı temsil eder)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probs}


class ABSAPipeline:
    """
    Tam ABSA pipeline'ı:
    Metin → Aspect Extraction → Sentiment Classification → Sonuç Dict
    """

    def __init__(
        self,
        extractor_path: Optional[str] = None,
        sentiment_path: Optional[str] = None,
        model_name: str = "bert-base-multilingual-cased",
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Aspect extractor
        self.extractor = AspectTermExtractor(bert_model_name=model_name)
        if extractor_path:
            self.extractor.load_state_dict(torch.load(extractor_path, map_location=self.device))
        self.extractor.to(self.device)
        self.extractor.eval()

        # Sentiment classifier
        self.sentiment_clf = AspectSentimentClassifier(bert_model_name=model_name)
        if sentiment_path:
            self.sentiment_clf.load_state_dict(torch.load(sentiment_path, map_location=self.device))
        self.sentiment_clf.to(self.device)
        self.sentiment_clf.eval()

    def analyze(self, text: str, category_aspects: Optional[List[str]] = None) -> Dict:
        """
        Bir yorumu analiz et.

        Args:
            text: Ham yorum metni
            category_aspects: Ürün kategorisine özel aspect listesi (varsa)

        Returns:
            {
                "text": str,
                "aspects": [
                    {
                        "aspect": "beden uyumu",
                        "sentiment": "NEG",
                        "confidence": 0.92,
                        "opinion_words": ["dar", "büyük"]
                    },
                    ...
                ],
                "overall_sentiment": "NEG",
                "summary_data": {...}
            }
        """
        # 1. Aspect extraction
        extracted_aspects = self._extract_aspects(text)

        # Kategori bazlı aspect'leri de ekle (kategori biliniyorsa)
        if category_aspects:
            for asp in category_aspects:
                if asp not in extracted_aspects:
                    # Aspect metinde geçiyor mu kontrol et
                    if any(kw in text.lower() for kw in asp.lower().split("_")):
                        extracted_aspects.append(asp)

        if not extracted_aspects:
            return {
                "text": text,
                "aspects": [],
                "overall_sentiment": "NEU",
                "message": "Aspect bulunamadı",
            }

        # 2. Her aspect için sentiment analizi
        aspect_results = []
        for aspect in extracted_aspects:
            sentiment_result = self._classify_sentiment(text, aspect)
            aspect_results.append({
                "aspect": aspect,
                "sentiment": sentiment_result["sentiment"],
                "confidence": sentiment_result["confidence"],
                "probabilities": sentiment_result["probabilities"],
            })

        # 3. Genel sentiment skoru
        overall = self._compute_overall_sentiment(aspect_results)

        return {
            "text": text,
            "aspects": aspect_results,
            "overall_sentiment": overall,
            "aspect_count": len(aspect_results),
        }

    def _extract_aspects(self, text: str) -> List[str]:
        """Metinden aspect terimlerini çıkar."""
        encoding = self.tokenizer(
            text,
            max_length=128,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.extractor(input_ids=input_ids, attention_mask=attention_mask)

        predictions = output["predictions"]
        if isinstance(predictions, torch.Tensor):
            predictions = predictions[0].tolist()
        else:
            predictions = predictions[0]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        aspects = self.extractor.extract_aspects(tokens, predictions)

        # WordPiece token'larını temizle (## prefix)
        aspects = [a.replace(" ##", "").replace("##", "") for a in aspects]
        return [a for a in aspects if len(a) > 1]

    def _classify_sentiment(self, text: str, aspect: str) -> Dict:
        """Bir aspect için sentiment tahmini yap."""
        # BERT pair encoding: [CLS] text [SEP] aspect [SEP]
        encoding = self.tokenizer(
            text,
            aspect,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            output = self.sentiment_clf(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        probs = output["probabilities"][0]
        pred_id = torch.argmax(probs).item()

        return {
            "sentiment": SENTIMENT_LABELS[pred_id],
            "confidence": round(probs[pred_id].item(), 4),
            "probabilities": {
                label: round(probs[i].item(), 4)
                for i, label in SENTIMENT_LABELS.items()
            },
        }

    def _compute_overall_sentiment(self, aspect_results: List[Dict]) -> str:
        """Tüm aspect sonuçlarından genel bir sentiment üret."""
        if not aspect_results:
            return "NEU"

        scores = {"POS": 0, "NEG": 0, "NEU": 0}
        for result in aspect_results:
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            scores[sentiment] += confidence

        return max(scores, key=scores.get)


# ── Kullanım örneği ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ABSA Pipeline — Demo Modu (Model ağırlıkları yüklenmeden)")
    print("Gerçek kullanım için eğitilmiş model ağırlıkları gereklidir.")
    print()

    # Pipeline oluştur (eğitilmemiş model ile — sadece yapıyı test et)
    pipeline = ABSAPipeline(model_name="bert-base-multilingual-cased")

    sample_text = "Pantolonun kumaşı çok kaliteliydi ama beden büyük geldi. Uzunluk tam."
    print(f"Test metni: {sample_text}")
    print("Not: Anlamlı sonuçlar için modelin fine-tune edilmesi gerekir.")
