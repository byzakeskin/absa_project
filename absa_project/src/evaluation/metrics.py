"""
src/evaluation/metrics.py
---------------------------
ABSA sistemi için değerlendirme metrikleri.
F1, Precision, Recall, Accuracy ve özel ABSA metrikleri.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


SENTIMENT_LABELS = ["POS", "NEG", "NEU"]
LABEL_TO_ID = {"POS": 0, "NEG": 1, "NEU": 2}


class ABSAMetrics:
    """
    ABSA değerlendirme metrikleri hesaplama sınıfı.
    """

    def __init__(self, labels: List[str] = None):
        self.labels = labels or SENTIMENT_LABELS

    def compute_sentiment_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        verbose: bool = True,
    ) -> Dict:
        """
        Aspect sentiment classification metrikleri.

        Args:
            y_true: Gerçek sentiment etiketleri ["POS", "NEG", ...]
            y_pred: Tahmin edilen etiketler

        Returns:
            {
                "accuracy": float,
                "macro_f1": float,
                "weighted_f1": float,
                "per_class": {"POS": {"f1": ..., "precision": ..., "recall": ...}},
                "confusion_matrix": List[List[int]]
            }
        """
        # String → int dönüşümü
        y_true_ids = [LABEL_TO_ID.get(y, 2) for y in y_true]
        y_pred_ids = [LABEL_TO_ID.get(y, 2) for y in y_pred]

        accuracy = accuracy_score(y_true_ids, y_pred_ids)
        macro_f1 = f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_ids, y_pred_ids, average="weighted", zero_division=0)
        macro_precision = precision_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)
        macro_recall = recall_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)

        # Sınıf bazlı metrikler
        per_class = {}
        for i, label in enumerate(self.labels):
            per_class[label] = {
                "f1": round(f1_score(y_true_ids, y_pred_ids, labels=[i], average="micro", zero_division=0), 4),
                "precision": round(precision_score(y_true_ids, y_pred_ids, labels=[i], average="micro", zero_division=0), 4),
                "recall": round(recall_score(y_true_ids, y_pred_ids, labels=[i], average="micro", zero_division=0), 4),
                "count": y_true_ids.count(i),
            }

        cm = confusion_matrix(y_true_ids, y_pred_ids, labels=list(range(len(self.labels)))).tolist()

        metrics = {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "per_class": per_class,
            "confusion_matrix": cm,
        }

        if verbose:
            self._print_metrics(metrics)

        return metrics

    def compute_aspect_extraction_metrics(
        self,
        y_true_aspects: List[List[str]],
        y_pred_aspects: List[List[str]],
        match_type: str = "exact",
    ) -> Dict:
        """
        Aspect term extraction metrikleri.

        Args:
            y_true_aspects: Gerçek aspect listesi (her örnek için)
            y_pred_aspects: Tahmin edilen aspect listesi
            match_type: "exact" (tam eşleşme) veya "partial" (kısmi eşleşme)

        Returns:
            {"precision": float, "recall": float, "f1": float}
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for true_aspects, pred_aspects in zip(y_true_aspects, y_pred_aspects):
            true_set = set(true_aspects)
            pred_set = set(pred_aspects)

            if match_type == "exact":
                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
            else:  # partial match
                tp = sum(
                    1 for p in pred_set
                    if any(self._partial_match(p, t) for t in true_set)
                )
                fp = len(pred_set) - tp
                fn = sum(
                    1 for t in true_set
                    if not any(self._partial_match(p, t) for p in pred_set)
                )

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "match_type": match_type,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        }

    def _partial_match(self, pred: str, true: str, threshold: float = 0.5) -> bool:
        """İki string arasında kısmi eşleşme kontrolü."""
        pred_tokens = set(pred.lower().split())
        true_tokens = set(true.lower().split())
        if not pred_tokens or not true_tokens:
            return False
        overlap = len(pred_tokens & true_tokens)
        return overlap / max(len(pred_tokens), len(true_tokens)) >= threshold

    def compute_category_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        category_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Ürün kategorisi sınıflandırma metrikleri.
        """
        if category_names is None:
            category_names = sorted(set(y_true + y_pred))

        label_to_id = {label: i for i, label in enumerate(category_names)}
        y_true_ids = [label_to_id.get(y, 0) for y in y_true]
        y_pred_ids = [label_to_id.get(y, 0) for y in y_pred]

        report = classification_report(
            y_true_ids, y_pred_ids,
            target_names=category_names,
            output_dict=True,
            zero_division=0,
        )

        return {
            "accuracy": round(accuracy_score(y_true_ids, y_pred_ids), 4),
            "macro_f1": round(f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0), 4),
            "per_category": {
                cat: {k: round(v, 4) for k, v in report[cat].items()}
                for cat in category_names if cat in report
            },
        }

    def _print_metrics(self, metrics: Dict):
        """Metrikleri güzel biçimde yazdır."""
        print("\n📊 Değerlendirme Sonuçları")
        print("=" * 45)
        print(f"  Accuracy     : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Macro F1     : {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1  : {metrics['weighted_f1']:.4f}")
        print(f"  Macro Prec.  : {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall : {metrics['macro_recall']:.4f}")
        print()
        print("  Per-Class Breakdown:")
        print(f"  {'Label':<8} {'F1':>7} {'Prec':>8} {'Rec':>8} {'Count':>7}")
        print("  " + "-" * 42)
        for label, vals in metrics["per_class"].items():
            print(f"  {label:<8} {vals['f1']:>7.4f} {vals['precision']:>8.4f} {vals['recall']:>8.4f} {vals['count']:>7}")
        print()
        print("  Confusion Matrix (satır=gerçek, sütun=tahmin):")
        header = "       " + "".join(f"{l:>7}" for l in SENTIMENT_LABELS)
        print("  " + header)
        for i, row in enumerate(metrics["confusion_matrix"]):
            row_str = f"  {SENTIMENT_LABELS[i]:<6}" + "".join(f"{v:>7}" for v in row)
            print(row_str)
        print()


class BaselineComparison:
    """
    Sistem sonuçlarını baseline ile karşılaştırma.
    """

    def compare(
        self,
        baseline_metrics: Dict,
        model_metrics: Dict,
        metric_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Baseline ve model metriklerini karşılaştır.
        """
        if metric_names is None:
            metric_names = ["accuracy", "macro_f1", "weighted_f1"]

        comparison = {}
        print("\n📈 Baseline vs Model Karşılaştırması")
        print("=" * 50)
        print(f"{'Metrik':<18} {'Baseline':>10} {'Model':>10} {'İyileşme':>12}")
        print("-" * 50)

        for metric in metric_names:
            b_val = baseline_metrics.get(metric, 0)
            m_val = model_metrics.get(metric, 0)
            improvement = m_val - b_val
            improvement_pct = (improvement / b_val * 100) if b_val > 0 else 0
            emoji = "✅" if improvement > 0 else "❌"

            print(f"{metric:<18} {b_val:>10.4f} {m_val:>10.4f} {emoji} {improvement_pct:>+8.1f}%")
            comparison[metric] = {
                "baseline": b_val,
                "model": m_val,
                "improvement": round(improvement, 4),
                "improvement_pct": round(improvement_pct, 2),
            }

        print()
        return comparison


# ── Kullanım örneği ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics = ABSAMetrics()

    # Simüle edilmiş tahminler
    y_true = ["POS", "NEG", "NEU", "POS", "NEG", "POS", "NEU", "NEG", "POS", "NEG"]
    y_pred = ["POS", "NEG", "NEU", "POS", "POS", "POS", "NEU", "NEG", "NEG", "NEG"]

    result = metrics.compute_sentiment_metrics(y_true, y_pred)

    # Aspect extraction metrikleri
    true_aspects = [["beden uyumu", "kumaş kalitesi"], ["rahatlık"], ["numara uyumu", "taban"]]
    pred_aspects = [["beden uyumu", "kalite"], ["rahatlık", "uzunluk"], ["numara"]]

    ext_metrics = metrics.compute_aspect_extraction_metrics(true_aspects, pred_aspects, match_type="exact")
    print(f"Aspect Extraction (Exact)  — F1: {ext_metrics['f1']:.4f}")

    ext_metrics_p = metrics.compute_aspect_extraction_metrics(true_aspects, pred_aspects, match_type="partial")
    print(f"Aspect Extraction (Partial) — F1: {ext_metrics_p['f1']:.4f}")

    # Baseline karşılaştırması
    baseline = {"accuracy": 0.72, "macro_f1": 0.67, "weighted_f1": 0.70}
    BaselineComparison().compare(baseline, result)
