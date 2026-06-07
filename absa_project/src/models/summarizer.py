"""
src/models/summarizer.py
--------------------------
Aspect analizi sonuçlarından özet paragraf üreten modül.
Her ürün kategorisi için yapılandırılmış şablon tabanlı özetleme.
"""

from typing import List, Dict, Optional
from collections import defaultdict


# Her kategori için özet şablonları (Türkçe)
SUMMARY_TEMPLATES_TR = {
    "pantolon_jean": (
        "Kullanıcı yorumlarına göre {product_name}; {kumas_kalitesi_text} "
        "{beden_uyumu_text} {uzunluk_text} {kalip_text} "
        "Genel değerlendirmede ürün {overall_text}."
    ),
    "mont_kaban": (
        "Bu ürün genel olarak {sicaklik_text} {kalinlik_text} "
        "{su_gecirmezlik_text} {ruzgar_text} "
        "Beden uyumu {beden_uyumu_text} Genel olarak ürün {overall_text}."
    ),
    "elbise_etek": (
        "Ürün yorumlarına göre {kumas_kalitesi_text} {boy_uyumu_text} "
        "{seffaflik_text} {kalip_text} "
        "Genel değerlendirmede {overall_text}."
    ),
    "tisort_gomlek_bluz": (
        "{kumas_kalitesi_text} {seffaflik_text} {dikis_kalitesi_text} "
        "Yıkama sonrası performans {yikama_text} "
        "Genel olarak ürün {overall_text}."
    ),
    "ayakkabi": (
        "Ayakkabı {rahatlik_text} {numara_uyumu_text} "
        "{ayak_vurma_text} {deformasyon_text} "
        "Genel değerlendirmede ürün {overall_text}."
    ),
    "terlik_sandalet_cizme": (
        "{rahatlik_text} {numara_uyumu_text} "
        "{malzeme_kalitesi_text} {taban_text} "
        "Genel olarak {overall_text}."
    ),
    "canta": (
        "{malzeme_kalitesi_text} {fermuar_toka_text} "
        "{kapasite_text} {tasima_text} "
        "Genel değerlendirmede {overall_text}."
    ),
}

# Sentiment → doğal dil çevirisi (Türkçe)
SENTIMENT_TO_TEXT_TR = {
    "beden_uyumu": {
        "POS": "beden uyumu standart ve doğru beden alınması önerilmektedir",
        "NEG": "beden konusunda dikkatli olunmalı; yorumlara göre {direction} beden alınması tavsiye edilmektedir",
        "NEU": "beden uyumu konusunda yeterli bilgi bulunmamaktadır",
    },
    "kumas_kalitesi": {
        "POS": "kumaş kalitesi kullanıcılar tarafından olumlu bulunmuş ve kaliteli hissettirdiği belirtilmiştir",
        "NEG": "kumaş kalitesine ilişkin olumsuz yorumlar bulunmakta; ince veya düşük kaliteli olduğu belirtilmektedir",
        "NEU": "kumaş kalitesi hakkında yeterince yorum yapılmamıştır",
    },
    "rahatlik": {
        "POS": "kullanıcıların büyük çoğunluğu ürünün rahat olduğunu ve günlük kullanıma uygun bulunduğunu belirtmiştir",
        "NEG": "bazı kullanıcılar ürünün uzun süreli kullanımda rahatsızlık yarattığını ifade etmiştir",
        "NEU": "rahatlık konusunda görüşler farklılık göstermektedir",
    },
    "numara_uyumu": {
        "POS": "numara uyumu standart olup doğru numaranın alınması yeterlidir",
        "NEG": "numara konusunda dikkat edilmesi gerektiği; yorumlara göre {direction} numara tercih edilmesi önerilmektedir",
        "NEU": "numara uyumu hakkında yeterli bilgi bulunmamaktadır",
    },
    "sicaklik_korumasi": {
        "POS": "sıcaklık koruması güçlü olup soğuk havalarda yeterli koruma sağlamaktadır",
        "NEG": "sıcaklık koruması bazı kullanıcılara göre yetersiz kalmakta, çok soğuk havalarda ilave giysi gerektirebilmektedir",
        "NEU": "sıcaklık koruması konusunda görüşler farklılık göstermektedir",
    },
    "seffaflik": {
        "POS": "kumaş şeffaf değildir ve içini göstermemektedir",
        "NEG": "kumaşın şeffaf olduğuna dair yorumlar mevcuttur; içlik giyilmesi önerilmektedir",
        "NEU": "şeffaflık konusunda yeterli bilgi bulunmamaktadır",
    },
    "default": {
        "POS": "bu özellik kullanıcılar tarafından olumlu değerlendirilmiştir",
        "NEG": "bu özellik bazı kullanıcılar tarafından olumsuz bulunmuştur",
        "NEU": "bu özellik hakkında nötr değerlendirmeler yapılmıştır",
    },
}

OVERALL_TEXT_TR = {
    "POS": "olumlu yorumlar almakta ve önerilmektedir",
    "NEG": "olumsuz yorumlar almaktadır; satın almadan önce yorumların dikkatlice incelenmesi tavsiye edilir",
    "NEU": "karışık yorumlar almakta olup kişisel beklentilere göre değerlendirilmesi önerilmektedir",
}


class SummaryGenerator:
    """
    ABSA sonuçlarından ürün kategorisine özel özet paragraf üretici.
    """

    def __init__(self, language: str = "tr"):
        self.language = language
        self.templates = SUMMARY_TEMPLATES_TR  # Gelecekte EN şablonları eklenebilir

    def generate(
        self,
        category: str,
        aspect_results: List[Dict],
        review_count: int = 0,
        product_name: str = "ürün",
    ) -> str:
        """
        Özet paragraf üret.

        Args:
            category: Ürün kategorisi (örn: "pantolon_jean")
            aspect_results: ABSAPipeline'dan gelen aspect sonuçları
            review_count: Analiz edilen yorum sayısı
            product_name: Ürün adı

        Returns:
            Okunabilir özet paragraf (str)
        """
        if not aspect_results:
            return self._no_data_summary(product_name, review_count)

        # Aspect → sentiment eşlemesi
        aspect_sentiment_map = {
            result["aspect"]: result["sentiment"]
            for result in aspect_results
        }

        # Genel sentiment (ağırlıklı)
        overall = self._compute_overall(aspect_results)

        # Kategori bazlı metin üret
        if category in self.templates:
            summary = self._template_based_summary(
                category, aspect_sentiment_map, overall, product_name
            )
        else:
            summary = self._generic_summary(aspect_results, overall, product_name)

        # Review count ekle
        if review_count > 0:
            summary = f"({review_count} yorum analiz edildi) " + summary

        return summary.strip()

    def _template_based_summary(
        self,
        category: str,
        aspect_map: Dict[str, str],
        overall: str,
        product_name: str,
    ) -> str:
        """Şablon tabanlı özet üret."""
        template = self.templates[category]
        fill_values = {"product_name": product_name, "overall_text": OVERALL_TEXT_TR[overall]}

        # Her aspect için metin üret
        for aspect_key, sentiment_texts in SENTIMENT_TO_TEXT_TR.items():
            if aspect_key == "default":
                continue
            matched_sentiment = self._find_aspect_sentiment(aspect_key, aspect_map)
            text = sentiment_texts.get(matched_sentiment or "NEU", "")
            text = text.replace("{direction}", self._infer_direction(aspect_key, aspect_map))
            fill_values[f"{aspect_key}_text"] = text + "." if text else ""

        # Eksik placeholder'ları temizle
        import re
        summary = template.format(**fill_values)
        summary = re.sub(r'\{[^}]+\}', '', summary)
        summary = re.sub(r'\s+', ' ', summary)
        return summary.strip()

    def _generic_summary(
        self,
        aspect_results: List[Dict],
        overall: str,
        product_name: str,
    ) -> str:
        """Şablon olmayan kategoriler için genel özet."""
        pos_aspects = [r["aspect"] for r in aspect_results if r["sentiment"] == "POS"]
        neg_aspects = [r["aspect"] for r in aspect_results if r["sentiment"] == "NEG"]

        parts = [f"Kullanıcı yorumlarına göre {product_name};"]

        if pos_aspects:
            parts.append(f"{', '.join(pos_aspects)} konularında olumlu değerlendirmeler yapılmıştır.")
        if neg_aspects:
            parts.append(f"Ancak {', '.join(neg_aspects)} konularında dikkat edilmesi önerilmektedir.")

        parts.append(f"Genel olarak ürün {OVERALL_TEXT_TR[overall]}.")
        return " ".join(parts)

    def _find_aspect_sentiment(
        self,
        aspect_key: str,
        aspect_map: Dict[str, str],
    ) -> Optional[str]:
        """Aspect haritasında en uygun eşleşmeyi bul."""
        # Direkt eşleşme
        for aspect, sentiment in aspect_map.items():
            aspect_normalized = aspect.lower().replace(" ", "_").replace("/", "_")
            if aspect_key in aspect_normalized or aspect_normalized in aspect_key:
                return sentiment
        return None

    def _infer_direction(self, aspect_key: str, aspect_map: Dict[str, str]) -> str:
        """Beden/numara yönünü çıkar ('bir beden büyük' veya 'bir beden küçük')."""
        # Gelecekte yorum metninden çıkarılabilir
        # Şimdilik genel tavsiye
        return "bir beden büyük"

    def _compute_overall(self, aspect_results: List[Dict]) -> str:
        """Ağırlıklı genel sentiment hesapla."""
        scores = defaultdict(float)
        for r in aspect_results:
            scores[r["sentiment"]] += r.get("confidence", 1.0)
        if not scores:
            return "NEU"
        return max(scores, key=scores.get)

    def _no_data_summary(self, product_name: str, review_count: int) -> str:
        count_text = f"{review_count} yorum incelendi ancak " if review_count > 0 else ""
        return (
            f"{count_text}{product_name} için yeterli aspect bilgisi çıkarılamadı. "
            "Yorumları doğrudan incelemeniz önerilir."
        )


# ── Kullanım örneği ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    generator = SummaryGenerator(language="tr")

    # Simüle edilmiş ABSA sonuçları
    aspect_results = [
        {"aspect": "kumaş kalitesi", "sentiment": "POS", "confidence": 0.91},
        {"aspect": "beden uyumu",    "sentiment": "NEG", "confidence": 0.88},
        {"aspect": "uzunluk",        "sentiment": "POS", "confidence": 0.76},
        {"aspect": "şeffaflık",      "sentiment": "NEG", "confidence": 0.83},
    ]

    summary = generator.generate(
        category="elbise_etek",
        aspect_results=aspect_results,
        review_count=147,
        product_name="Midi Elbise",
    )

    print("Üretilen Özet:")
    print("-" * 60)
    print(summary)
    print()

    # Ayakkabı örneği
    shoe_results = [
        {"aspect": "rahatlık",     "sentiment": "POS", "confidence": 0.85},
        {"aspect": "numara uyumu", "sentiment": "NEG", "confidence": 0.79},
        {"aspect": "ayak vurma",   "sentiment": "NEG", "confidence": 0.91},
    ]

    shoe_summary = generator.generate(
        category="ayakkabi",
        aspect_results=shoe_results,
        review_count=63,
        product_name="Spor Ayakkabı",
    )
    print("Ayakkabı Özeti:")
    print("-" * 60)
    print(shoe_summary)
