"""
src/preprocessing/lang_detector.py
------------------------------------
Yorum dilini otomatik algılama modülü.
Türkçe ve İngilizce için optimize edilmiştir.
"""

from langdetect import detect, detect_langs, LangDetectException
import re
from typing import Tuple


# Türkçe'ye özgü karakter ve kelimeler
TURKISH_INDICATORS = {
    'chars': set('ğĞüÜşŞıİöÖçÇ'),
    'common_words': {
        'bir', 've', 'bu', 'ile', 'da', 'de', 'için', 'ama', 'çok',
        'aldım', 'geldi', 'güzel', 'kaliteli', 'beden', 'kumaş',
        'ayakkabı', 'mont', 'pantolon', 'elbise', 'beğendim', 'tavsiye'
    }
}

ENGLISH_INDICATORS = {
    'common_words': {
        'the', 'and', 'is', 'it', 'for', 'this', 'that', 'with',
        'size', 'quality', 'fabric', 'love', 'great', 'bought', 'fits',
        'recommend', 'comfortable', 'returned', 'disappointed'
    }
}


class LanguageDetector:
    """
    Yorum dili algılama sınıfı.
    Kısa metinler için kural tabanlı yöntem,
    uzun metinler için langdetect kütüphanesi kullanılır.
    """

    def __init__(self, min_text_length: int = 20, confidence_threshold: float = 0.85):
        """
        Args:
            min_text_length: Bu uzunluktan kısa metinler için kural tabanlı yöntem kullan
            confidence_threshold: Bu eşiğin altında güvenle algılanamayan metinler için fallback
        """
        self.min_text_length = min_text_length
        self.confidence_threshold = confidence_threshold

    def detect(self, text: str) -> str:
        """
        Metnin dilini algıla.

        Returns:
            "tr", "en" veya "unknown"
        """
        if not text or len(text.strip()) < 3:
            return "unknown"

        text = text.strip()

        # 1. Türkçe karakterlere bak (en hızlı yöntem)
        turkish_char_count = sum(1 for c in text if c in TURKISH_INDICATORS['chars'])
        if turkish_char_count >= 2:
            return "tr"

        # 2. Kısa metin → kural tabanlı
        if len(text) < self.min_text_length:
            return self._rule_based_detect(text)

        # 3. Uzun metin → langdetect
        return self._library_detect(text)

    def detect_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Dil tespiti + güven skoru döndür.

        Returns:
            ("tr", 0.98) gibi (dil, güven) tuple'ı
        """
        if not text or len(text.strip()) < 3:
            return ("unknown", 0.0)

        text = text.strip()

        # Türkçe karakterler varsa çok yüksek güven
        turkish_chars = sum(1 for c in text if c in TURKISH_INDICATORS['chars'])
        if turkish_chars >= 3:
            return ("tr", 0.99)
        elif turkish_chars >= 1:
            return ("tr", 0.85)

        try:
            langs = detect_langs(text)
            for lang in langs:
                if lang.lang in ("tr", "en"):
                    return (lang.lang, lang.prob)
            # İlk tahmine dön
            return (langs[0].lang if langs else "unknown", langs[0].prob if langs else 0.0)
        except LangDetectException:
            return (self._rule_based_detect(text), 0.5)

    def _rule_based_detect(self, text: str) -> str:
        """Kural tabanlı dil tespiti — kısa metinler için."""
        words = set(text.lower().split())

        tr_score = len(words & TURKISH_INDICATORS['common_words'])
        en_score = len(words & ENGLISH_INDICATORS['common_words'])

        if tr_score > en_score:
            return "tr"
        elif en_score > tr_score:
            return "en"
        return "unknown"

    def _library_detect(self, text: str) -> str:
        """langdetect kütüphanesi ile dil tespiti."""
        try:
            lang = detect(text)
            if lang == "tr":
                return "tr"
            elif lang == "en":
                return "en"
            else:
                return "unknown"
        except LangDetectException:
            return "unknown"

    def is_turkish(self, text: str) -> bool:
        return self.detect(text) == "tr"

    def is_english(self, text: str) -> bool:
        return self.detect(text) == "en"


# ── Kullanım örneği ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = LanguageDetector()

    test_cases = [
        "Bu ürün gerçekten çok güzel, beden uyumu mükemmel!",
        "The jacket fits perfectly and the quality is amazing!",
        "Ayakkabı dar geldi biraz",
        "Size runs small, ordered up one",
        "güzel ürün",
        "nice product",
        "KALİTELİ ÜRÜN ÇOK BEĞENDİM",
    ]

    print(f"{'Text':<50} {'Lang':<8} {'Confidence'}")
    print("-" * 70)
    for text in test_cases:
        lang, conf = detector.detect_with_confidence(text)
        print(f"{text[:48]:<50} {lang:<8} {conf:.2f}")
