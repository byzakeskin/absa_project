"""
src/preprocessing/cleaner.py
----------------------------
Türkçe ve İngilizce yorum metinleri için temizleme pipeline'ı.
"""

import re
import unicodedata
from typing import Optional
import nltk
from nltk.corpus import stopwords

# İlk çalıştırmada NLTK verilerini indir
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Türkçe özel stop word listesi (NLTK'ye ek olarak)
TURKISH_EXTRA_STOPWORDS = {
    "bir", "bu", "şu", "o", "ve", "ile", "da", "de", "ki", "mi", "mu",
    "mü", "mı", "ama", "fakat", "lakin", "ancak", "için", "gibi", "kadar",
    "çok", "az", "daha", "en", "her", "hiç", "bile", "yani", "zaten",
    "artık", "hep", "ne", "nasıl", "neden", "niye"
}

TURKISH_STOPWORDS = set(stopwords.words('turkish')) | TURKISH_EXTRA_STOPWORDS
ENGLISH_STOPWORDS = set(stopwords.words('english'))


class TextCleaner:
    """
    E-ticaret yorumları için metin temizleme sınıfı.
    Türkçe ve İngilizce dil desteği sağlar.
    """

    def __init__(self, language: str = "tr", remove_stopwords: bool = False):
        """
        Args:
            language: "tr" (Türkçe) veya "en" (İngilizce)
            remove_stopwords: Stop word'leri kaldır (default False — BERT için gerekli değil)
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.stopwords = TURKISH_STOPWORDS if language == "tr" else ENGLISH_STOPWORDS

    def clean(self, text: str) -> str:
        """Ana temizleme pipeline'ı."""
        if not text or not isinstance(text, str):
            return ""

        text = self._remove_html_tags(text)
        text = self._normalize_unicode(text)
        text = self._remove_urls(text)
        text = self._remove_special_chars(text)
        text = self._normalize_whitespace(text)
        text = self._lowercase(text)
        text = self._fix_repeated_chars(text)

        if self.remove_stopwords:
            text = self._remove_stopwords(text)

        return text.strip()

    def _remove_html_tags(self, text: str) -> str:
        """HTML etiketlerini temizle."""
        return re.sub(r'<[^>]+>', ' ', text)

    def _normalize_unicode(self, text: str) -> str:
        """Unicode karakterleri normalize et, emoji'leri kaldır."""
        # Emoji ve özel sembolleri kaldır
        text = re.sub(r'[^\w\s\u00C0-\u024F\u0100-\u017E.,!?;:\-\'\"()]', ' ', text)
        # Unicode normalizasyonu
        text = unicodedata.normalize('NFC', text)
        return text

    def _remove_urls(self, text: str) -> str:
        """URL'leri kaldır."""
        return re.sub(r'http[s]?://\S+|www\.\S+', '', text)

    def _remove_special_chars(self, text: str) -> str:
        """Gereksiz özel karakterleri temizle (noktalama hariç)."""
        # Sayıları koru, Türkçe karakterleri koru
        text = re.sub(r'[*#@$%^&+=~`|\\/<>{}[\]]', ' ', text)
        # Birden fazla noktalama işaretini teke indir
        text = re.sub(r'([!?,.]){2,}', r'\1', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Fazla boşlukları ve newline'ları tek boşluğa indir."""
        text = re.sub(r'\s+', ' ', text)
        return text

    def _lowercase(self, text: str) -> str:
        """
        Türkçe farkındalıklı küçük harfe çevirme.
        Türkçe'de 'I' → 'ı' (noktalı I → noktalısız i değil!)
        """
        if self.language == "tr":
            # Türkçe büyük-küçük harf dönüşümü
            text = text.replace('İ', 'i').replace('I', 'ı')
            text = text.replace('Ğ', 'ğ').replace('Ü', 'ü')
            text = text.replace('Ş', 'ş').replace('Ö', 'ö')
            text = text.replace('Ç', 'ç')
        return text.lower()

    def _fix_repeated_chars(self, text: str) -> str:
        """Tekrar eden karakterleri normalize et: 'çoookk' → 'çok'"""
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def _remove_stopwords(self, text: str) -> str:
        """Stop word'leri kaldır."""
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords]
        return ' '.join(tokens)


class ReviewPreprocessor:
    """
    Yorum verisi için tam preprocessing pipeline'ı.
    Temizleme + normalizasyon + metadata çıkarımı.
    """

    def __init__(self):
        self.tr_cleaner = TextCleaner(language="tr")
        self.en_cleaner = TextCleaner(language="en")

    def process(self, text: str, language: str = "tr") -> dict:
        """
        Bir yorumu işle ve sonuç dict döndür.

        Returns:
            {
                "original": str,
                "cleaned": str,
                "language": str,
                "char_count": int,
                "word_count": int,
                "has_size_mention": bool,
                "has_quality_mention": bool
            }
        """
        cleaner = self.tr_cleaner if language == "tr" else self.en_cleaner
        cleaned = cleaner.clean(text)

        return {
            "original": text,
            "cleaned": cleaned,
            "language": language,
            "char_count": len(cleaned),
            "word_count": len(cleaned.split()),
            "has_size_mention": self._check_size_mention(cleaned, language),
            "has_quality_mention": self._check_quality_mention(cleaned, language),
        }

    def _check_size_mention(self, text: str, lang: str) -> bool:
        """Metinde beden/numara bilgisi var mı?"""
        patterns = {
            "tr": r'\b(beden|numara|ölçü|xs|s\b|m\b|l\b|xl|xxl|\d{2}\s*beden)\b',
            "en": r'\b(size|xs|small|medium|large|xl|xxl|\d{1,2}\s*inch)\b'
        }
        return bool(re.search(patterns.get(lang, patterns["en"]), text, re.IGNORECASE))

    def _check_quality_mention(self, text: str, lang: str) -> bool:
        """Metinde kalite bilgisi var mı?"""
        patterns = {
            "tr": r'\b(kalite|kumaş|malzeme|dayanıklı|sağlam|bozuldu)\b',
            "en": r'\b(quality|fabric|material|durable|sturdy|broke|fell apart)\b'
        }
        return bool(re.search(patterns.get(lang, patterns["en"]), text, re.IGNORECASE))


# ── Kullanım örneği ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocessor = ReviewPreprocessor()

    samples = [
        ("Pantolonu aldım, kumaşı ÇOK kaliteliydi!!! Ama beden büyük geldi 😕", "tr"),
        ("The jacket is AMAZING but runs really small. Size up for sure!!!", "en"),
        ("   <p>Ayakkabı dar geldi...   </p>\n Topuklar yara açtı  ", "tr"),
    ]

    for text, lang in samples:
        result = preprocessor.process(text, lang)
        print(f"\nOriginal : {result['original'][:60]}...")
        print(f"Cleaned  : {result['cleaned']}")
        print(f"Words    : {result['word_count']} | Size mention: {result['has_size_mention']}")
