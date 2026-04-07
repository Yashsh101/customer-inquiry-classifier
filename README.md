🚀 Customer Inquiry Classifier (Production-Ready NLP System)












⚡ A production-grade NLP system that classifies customer support inquiries into 7 actionable categories with calibrated confidence scores, keyword-level explainability, and sub-15ms latency.

🌐 Live Demo
🎯 Streamlit App: https://customer-inquiry-classifier.streamlit.app
📡 API Docs (Swagger): /docs
🧠 Problem Statement

Customer support teams handle thousands of unstructured queries daily:

Refund requests
Billing issues
Technical problems
Account-related queries

Manual triaging is:

❌ Slow
❌ Inconsistent
❌ Hard to scale

👉 This system automates classification in milliseconds, enabling:

Faster routing
Reduced support load
Improved customer experience
✨ Key Features
🧠 Multi-class NLP classification (7 real-world categories)
⚡ Ultra-fast inference (< 15 ms)
📊 Calibrated confidence scores (probability-aware predictions)
🔍 Keyword-level explainability (TF-IDF insights)
🌐 FastAPI production-ready backend
🎨 Streamlit interactive UI
🧪 18/18 pytest unit tests
🔁 CI pipeline with GitHub Actions
📦 Model persistence with .joblib
🐳 Dockerized deployment
📊 Model Performance
Metric	Score
Test Accuracy	93.4%
Weighted F1	93.2%
CV Accuracy (5-fold)	92.8% ± 0.6%
Inference Latency	< 15 ms
<details> <summary>Per-category breakdown</summary>
Category	Precision	Recall	F1
Billing	0.95	0.94	0.94
Technical Support	0.93	0.95	0.94
Product Inquiry	0.91	0.92	0.91
Shipping	0.95	0.93	0.94
Refund / Return	0.94	0.93	0.93
Account Management	0.92	0.91	0.92
General Inquiry	0.90	0.92	0.91
</details>
🏗️ System Architecture
Input Text
    │
    ▼
Text Preprocessing
(lowercase → clean → lemmatize → stopword removal)
    │
    ▼
TF-IDF Vectorizer
(8k features, 1–3 ngrams, sublinear TF)
    │
    ▼
Soft Voting Ensemble
├── Calibrated Linear SVC (Platt scaling)
└── Logistic Regression (multinomial)
    │
    ▼
Prediction Output
├── category
├── confidence score
├── class probabilities
├── top keywords
└── latency (ms)
🖼️ Demo
🔹 Streamlit Interface

🔹 Prediction Output

🔹 API (Swagger UI)

📁 Project Structure
customer-inquiry-classifier/
├── app/
│   ├── classifier.py       # ML pipeline (training + inference)
│   └── api.py              # FastAPI backend
├── models/                 # Saved model artifacts
├── tests/
│   └── test_classifier.py  # Unit tests
├── .github/workflows/
│   └── ci.yml              # CI pipeline
├── streamlit_app.py        # UI
├── Dockerfile
├── requirements.txt
└── README.md
⚡ Quick Start
1. Clone & Install
git clone https://github.com/Yashsh101/customer-inquiry-classifier.git
cd customer-inquiry-classifier
pip install -r requirements.txt
2. Run Streamlit App
streamlit run streamlit_app.py
3. Run API
uvicorn app.api:app --reload

👉 Visit: http://localhost:8000/docs

4. Run Tests
pytest -v
5. Docker
docker build -t inquiry-classifier .
docker run -p 8000:8000 inquiry-classifier
🔌 API Example
POST /predict
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text": "I was charged twice this month"}'
Response
{
  "category": "billing",
  "confidence": 0.9621,
  "latency_ms": 8.3,
  "top_keywords": ["charged", "credit", "month", "twice"]
}
🧠 Key Design Decisions
Decision	Reason
Ensemble (SVC + LR)	Combines accuracy + calibrated probabilities
TF-IDF with n-grams	Captures phrases like "charged twice"
Platt scaling	Converts SVC scores into meaningful probabilities
Stratified CV	Ensures reliable generalization
Explainability	Builds trust in real-world usage
🛠️ Tech Stack

Python · scikit-learn · FastAPI · Streamlit · Plotly · pytest · Docker · GitHub Actions

🎯 Why This Project Stands Out

This is not a basic ML notebook.

It demonstrates:

End-to-end ML system (model + API + UI)
Production-ready architecture
Deployment & persistence
Testing + CI pipeline

👉 Built with real-world engineering practices used in industry.

🚀 Future Improvements
Transformer-based models (BERT)
Multi-language support
Real-time inference pipeline
Model monitoring & drift detection
Database integration
👨‍💻 Author

Yash Sharma
LinkedIn
 · GitHub

⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
It helps improve visibility and reach.
