# Customer Inquiry Classifier

Production-ready customer support inquiry classifier deployed on Vercel with a FastAPI backend and a minimal responsive frontend.

## What It Does

- Classifies customer messages into billing, technical support, product inquiry, shipping, refund/return, account management, or general inquiry.
- Returns calibrated confidence, routing decision, destination queue, top text signals, and full probability breakdown.
- Auto-routes high-confidence inquiries and flags uncertain cases for human review.
- Uses a pre-trained TF-IDF + calibrated SVC + logistic regression ensemble stored in `models/classifier_v2.joblib`.

## Project Structure

```text
api/
  index.py            # Vercel Python serverless entrypoint
app/
  api.py              # FastAPI app and API contracts
  classifier.py       # ML preprocessing, model loading, inference, routing
models/
  classifier_v2.joblib
public/
  index.html          # Static UI
  styles.css
  app.js
tests/
  test_classifier.py
requirements.txt
vercel.json
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api:app --reload
```

Open `http://localhost:8000/docs` for API docs.

For the full Vercel-style local app:

```bash
npx vercel dev
```

## API

```bash
curl -X POST http://localhost:8000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"I was charged twice this month and need a refund\"}"
```

Main endpoints:

- `GET /api/health`
- `GET /api/categories`
- `POST /api/predict`
- `POST /api/predict/batch`

## Deployment

This repo is ready for Vercel with no manual edits:

```bash
vercel --prod
```

Vercel serves the frontend from `public/` and rewrites `/api/*` requests to the FastAPI serverless function in `api/index.py`.

## Configuration

Optional environment variables:

- `ROUTING_CONF_THRESHOLD` defaults to `0.75`
- `ENABLE_LLM_FALLBACK` defaults to `false`
- `COMPARE_WITH_LLM` defaults to `false`
- `OPENAI_API_KEY` required only if LLM fallback is enabled
- `OPENAI_MODEL` defaults to `gpt-4.1-mini`

## Validation

```bash
pip install pytest ruff
ruff check app api tests --ignore E501
pytest tests/ -v
```
