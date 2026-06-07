# 🛍️ Yapay Zeka Destekli E-Ticaret Ürün Yorum Analiz Sistemi

> **Aspect-Based Sentiment Analysis (ABSA)** — Giyim ürünleri için kategori bazlı yorum analizi

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Proje Özeti

Bu proje, e-ticaret platformlarındaki (Trendyol, Hepsiburada, Amazon) giyim ürünü yorumlarını otomatik olarak analiz eden, **aspect bazlı duygu analizi** yapan ve kullanıcıya özet bilgi sunan bir yapay zeka sistemidir.

**Desteklenen Diller:** Türkçe 🇹🇷 + İngilizce 🇬🇧

### Sistem Ne Yapar?

```
"Pantolonu aldım, kumaşı çok kaliteli ama beden büyük geldi. 
 Uzunluk tam ama paça dar."

            ⬇️  ABSA Sistemi

Kumaş Kalitesi  : ✅ Pozitif
Beden Uyumu     : ❌ Negatif  (1 beden küçük al önerilir)
Uzunluk         : ✅ Nötr/Pozitif
Kalıp (Paça)    : ❌ Negatif
```

---

## 📁 Proje Yapısı

```
absa_project/
│
├── data/
│   ├── raw/                    # Ham yorumlar (Kaggle + scraping)
│   ├── processed/              # Temizlenmiş ve tokenize edilmiş veri
│   └── annotated/              # El ile etiketlenmiş ABSA verisi
│
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py          # Metin temizleme
│   │   ├── tokenizer.py        # Tokenization pipeline
│   │   ├── lang_detector.py    # Dil algılama
│   │   └── aspect_annotator.py # Yarı-otomatik aspect etiketleme
│   │
│   ├── models/
│   │   ├── product_classifier.py    # Ürün kategori sınıflandırıcı
│   │   ├── aspect_extractor.py      # Aspect çıkarım modeli
│   │   ├── sentiment_classifier.py  # Duygu sınıflandırıcı
│   │   └── summarizer.py            # Özet paragraf üretici
│   │
│   ├── training/
│   │   ├── train_classifier.py      # Kategori modeli eğitimi
│   │   ├── train_absa.py            # ABSA modeli eğitimi
│   │   └── multi_task_trainer.py    # Çok görevli eğitim
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # F1, Accuracy, Precision, Recall
│   │   └── evaluator.py             # Model değerlendirme pipeline
│   │
│   └── api/
│       ├── app.py                   # FastAPI uygulaması
│       └── schemas.py               # Request/Response şemaları
│
├── chrome_extension/
│   ├── manifest.json
│   ├── content.js
│   ├── popup.html
│   └── popup.js
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Keşifsel veri analizi
│   ├── 02_baseline_model.ipynb      # Baseline SVM modeli
│   ├── 03_bert_finetuning.ipynb     # BERT fine-tuning deneyleri
│   └── 04_evaluation.ipynb          # Sonuç analizi
│
├── scripts/
│   ├── scrape_trendyol.py           # Trendyol scraper
│   ├── scrape_hepsiburada.py        # Hepsiburada scraper
│   └── download_kaggle_data.py      # Kaggle veri indirme
│
├── configs/
│   ├── model_config.yaml            # Model hyperparameter'ları
│   └── aspect_config.yaml           # Ürün sınıfı → aspect listesi
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/kullanici/absa-ecommerce.git
cd absa-ecommerce

# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## 📊 Kullanılan Modeller

| Görev | Model | Dil |
|-------|-------|-----|
| Ürün Kategorisi Tanıma | BERT-base fine-tuned | TR + EN |
| Aspect Extraction | BiLSTM + CRF | TR + EN |
| Aspect Sentiment | BERTurk fine-tuned | TR |
| Aspect Sentiment | RoBERTa fine-tuned | EN |
| Özet Üretimi | Template + Rule-based | TR + EN |

---

## 📈 Başarı Hedefleri

| Metrik | Hedef |
|--------|-------|
| Aspect Sentiment Macro-F1 | ≥ 0.80 |
| Kategori Tanıma Accuracy | ≥ %90 |
| Aspect Extraction F1 | ≥ 0.75 |
| İnsan Değerlendirmesi | ≥ 4.0/5.0 |

---

## 🗂️ Veri Setleri

- [Women's Clothing Reviews (Kaggle)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- [Amazon Product Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- [Clothing Fit Dataset](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation)
- Trendyol / Hepsiburada (Web Scraping)

---

## 📄 Lisans

MIT License — Detaylar için [LICENSE](LICENSE) dosyasına bakınız.
