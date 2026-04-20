# 🚀 Customer Inquiry Classifier (Production-Grade NLP Routing)

A recruiter-ready AI/ML project for **customer support automation**: classify incoming support text, route high-confidence tickets automatically, and send uncertain/ambiguous cases for human review.

## 🌍 Business Problem
Support teams lose time triaging repetitive tickets (billing, tech, shipping, refunds). This project automates first-line routing with confidence-aware decisions to reduce SLA delays, improve consistency, and allow agents to focus on complex cases.

---

## ✅ What’s New (v3)
- **Confidence-Based Routing (core upgrade):**
  - Uses `predict_proba` for calibrated class confidence.
  - **High confidence** → `auto_route` to business queue.
  - **Low confidence** → `human_review` (or optional LLM fallback route).
- **More realistic synthetic dataset:**
  - typos/noise, short + long queries,
  - tone variation (angry/urgent/polite),
  - ambiguous and multi-intent queries,
  - optional Hinglish-style expressions.
- **Advanced ML layer (Option A):**
  - Optional **OpenAI GPT fallback** for low-confidence cases.
  - ML prediction + optional LLM prediction comparison exposed in API.

---

## 🧠 System Architecture

```text
Incoming Inquiry
   │
   ▼
TextPreprocessor
(lowercase, cleanup, tokenize, lemmatize, stopwords)
   │
   ▼
TF-IDF (1-3 grams, max 8k)
   │
   ▼
Soft Voting Ensemble
  ├─ Calibrated LinearSVC (sigmoid calibration)
  └─ Logistic Regression
   │
   ▼
Confidence Router
  ├─ confidence >= threshold → auto_route to queue
  └─ confidence < threshold  → human_review
                             └─ optional GPT fallback route
```

---

## 🧪 ML Approach
- **Baseline/primary model:** TF-IDF + calibrated SVC + LR soft voting.
- **Feature engineering:** word n-grams, sublinear TF-IDF, preprocessing.
- **Evaluation:** holdout accuracy/F1 + 5-fold stratified CV.
- **Explainability:** top TF-IDF keywords surfaced for each prediction.

---

## 📊 Results Snapshot
(Results vary slightly run-to-run because of synthetic data generation.)

- Strong weighted F1 and accuracy for support-ticket category routing.
- Fast inference suitable for real-time API use.
- Confidence routing adds operational safety by preventing blind automation.

> Dataset note: training data is **synthetically generated for privacy** and demo reproducibility; no customer PII is used.

---

## 🔀 Confidence Routing Logic
- `ROUTING_CONF_THRESHOLD` (default: `0.75`) controls automation aggressiveness.
- API returns:
  - `confidence`
  - `routing_decision` (`auto_route`, `human_review`, `llm_fallback_route`)
  - `routed_team`
  - `requires_human_review`
  - optional `llm_prediction`, `llm_explanation`

---

## 🔌 API Usage

### Local run
```bash
pip install -r requirements.txt
uvicorn app.api:app --reload
```

### Predict example
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice this month and want a refund"}'
```

### Example response
```json
{
  "category": "billing",
  "final_category": "billing",
  "label": "💳 Billing",
  "confidence": 0.8421,
  "confidence_threshold": 0.75,
  "routing_decision": "auto_route",
  "routed_team": "billing_queue",
  "requires_human_review": false,
  "llm_fallback_used": false
}
```

---

## 🤖 Optional GPT Fallback Setup

Install OpenAI SDK only if you plan to enable fallback:

```bash
pip install openai
```

Set environment variables before starting API:

```bash
export ENABLE_LLM_FALLBACK=true
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-4.1-mini
# optional
export COMPARE_WITH_LLM=true
export ROUTING_CONF_THRESHOLD=0.75
```

Fallback behavior:
- If ML confidence is low, API can query GPT and return `llm_fallback_route` with `final_category` from LLM.
- If `COMPARE_WITH_LLM=true`, it can compare ML vs LLM even for high-confidence traffic.

---

## 🛠️ Deployment
- **Streamlit demo:** https://customer-inquiry-classifier-1.streamlit.app/
- **API docs (local/deployed):** `/docs`
- Production deployment template is included in `render.yaml` (API + Streamlit services).
- Streamlit runtime config is included in `.streamlit/config.toml`.

For Render/Railway deployment:
- ensure `requirements.txt` installed,
- use `uvicorn app.api:app --host 0.0.0.0 --port $PORT` for API,
- for Streamlit use `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`,
- persist `/models` or allow clean retrain on startup.

---

## 📁 Project Structure

```text
app/
  classifier.py   # model, confidence router, realistic data generation, GPT fallback adapter
  api.py          # FastAPI endpoints + startup/retrain behavior
tests/
  test_classifier.py
streamlit_app.py
requirements.txt
Dockerfile
```

---

## 💼 Recruiter Pitch (30 seconds)
- Solves a real support-ops bottleneck (ticket triage automation).
- Demonstrates ML + backend + explainability + confidence safety + LLM integration.
- Production-minded design: startup resilience, API contracts, tests, deployability.
