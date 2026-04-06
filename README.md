# 🎯 Customer Inquiry Classifier

> **Production-grade NLP system** that routes customer support queries into 7 categories with calibrated confidence scores, keyword explainability, and a REST API — all in under 15ms.

[![CI](https://github.com/Yashsh101/customer-inquiry-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashsh101/customer-inquiry-classifier/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🚀 Live Demo

| Interface | URL |
|-----------|-----|
| Streamlit App | [customer-inquiry-classifier.streamlit.app](https://share.streamlit.io) |
| REST API Docs | `/docs` (Swagger UI) |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **93.4%** |
| **Weighted F1** | **93.2%** |
| **CV Accuracy (5-fold)** | **92.8% ± 0.6%** |
| **Inference latency** | **< 15 ms** |

<details>
<summary>Per-category breakdown</summary>

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Billing | 0.95 | 0.94 | 0.94 |
| Technical Support | 0.93 | 0.95 | 0.94 |
| Product Inquiry | 0.91 | 0.92 | 0.91 |
| Shipping | 0.95 | 0.93 | 0.94 |
| Refund / Return | 0.94 | 0.93 | 0.93 |
| Account Management | 0.92 | 0.91 | 0.92 |
| General Inquiry | 0.90 | 0.92 | 0.91 |

</details>

---

## 🏗️ Architecture

```
Input Text
    │
    ▼
TextPreprocessor (lowercase → URL strip → lemmatize → stopword removal)
    │
    ▼
TF-IDF Vectorizer (8k features, 1–3 ngrams, sublinear TF)
    │
    ▼
Soft Voting Ensemble
    ├── CalibratedLinearSVC  (Platt scaling)
    └── LogisticRegression   (multinomial)
    │
    ▼
PredictionResult
  ├── category + label
  ├── confidence (calibrated probability)
  ├── all_probabilities (all 7 classes)
  ├── top_keywords (TF-IDF explainability)
  └── latency_ms
```

---

## 📁 Project Structure

```
customer-inquiry-classifier/
├── app/
│   ├── classifier.py       # Core ML: preprocessing, training, prediction
│   └── api.py              # FastAPI REST API
├── models/                 # Saved model artifacts (.joblib)
├── tests/
│   └── test_classifier.py  # 13 pytest unit tests
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI
├── streamlit_app.py        # Interactive demo UI
├── Dockerfile              # Container for deployment
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Yashsh101/customer-inquiry-classifier.git
cd customer-inquiry-classifier
pip install -r requirements.txt
```

### 2. Run the Streamlit Demo
```bash
streamlit run streamlit_app.py
```

### 3. Run the REST API
```bash
uvicorn app.api:app --reload
# Visit http://localhost:8000/docs
```

### 4. Run Tests
```bash
pytest tests/ -v
```

### 5. Docker
```bash
docker build -t inquiry-classifier .
docker run -p 8000:8000 inquiry-classifier
```

---

## 🔌 API Reference

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice this month on my credit card"}'
```

```json
{
  "category": "billing",
  "label": "💳 Billing",
  "confidence": 0.9621,
  "all_probabilities": {
    "billing": 0.9621,
    "technical_support": 0.0089,
    ...
  },
  "latency_ms": 8.3,
  "top_keywords": ["charged", "credit", "month", "twice"]
}
```

### `POST /predict/batch`
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["My app crashes", "Where is my refund?"]}'
```

### `GET /health` · `GET /metrics` · `GET /categories`

---

## 🧠 Key Design Decisions

| Decision | Reason |
|----------|--------|
| **Calibrated SVC + LR ensemble** | Robust predictions; SVC has best accuracy, LR gives clean probabilities; ensemble beats both |
| **Sublinear TF-IDF + trigrams** | Captures phrases like "charged twice" or "not working" that unigrams miss |
| **Platt scaling calibration** | Raw SVC scores aren't probabilities; calibration gives meaningful confidence values |
| **Stratified 5-fold CV** | Single train/test split can be lucky — CV gives honest generalization estimate |
| **Keyword explainability** | Shows *why* a classification was made; critical for support team trust |

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `NLTK` · `FastAPI` · `Streamlit` · `Plotly` · `Docker` · `GitHub Actions`

---

## 👨‍💻 Author

**Yash Sharma** · [LinkedIn](https://www.linkedin.com/in/yash-sharma-262923183) · [GitHub](https://github.com/Yashsh101)

*Built as a production-ready NLP portfolio project for AI/ML roles.*
