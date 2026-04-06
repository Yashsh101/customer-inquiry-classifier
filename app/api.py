"""
Customer Inquiry Classifier — FastAPI REST API
Endpoints: /predict  /predict/batch  /health  /metrics  /categories
"""

from __future__ import annotations
import os, time, logging
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.classifier import CustomerInquiryClassifier, DataGenerator, CATEGORY_LABELS, MODEL_PATH

logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
clf: Optional[CustomerInquiryClassifier] = None
_request_count = 0
_start_time = time.time()


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global clf
    if MODEL_PATH.exists():
        clf = CustomerInquiryClassifier.load(MODEL_PATH)
        logger.info("Loaded pre-trained model.")
    else:
        logger.info("No saved model found — training from scratch …")
        clf = CustomerInquiryClassifier()
        df = DataGenerator().generate(n_samples=4200)
        clf.train(df)
        clf.save(MODEL_PATH)
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Inquiry Classifier API",
    description="Production-grade NLP classification API — Yash Sharma",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000,
                      example="I was charged twice on my credit card this month")

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text must not be blank")
        return v.strip()


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50,
                              example=["My app keeps crashing", "Where is my refund?"])


class PredictResponse(BaseModel):
    category: str
    label: str
    confidence: float
    all_probabilities: dict[str, float]
    latency_ms: float
    top_keywords: list[str]


# ── Middleware: count requests ────────────────────────────────────────────────
@app.middleware("http")
async def count_requests(request: Request, call_next):
    global _request_count
    _request_count += 1
    return await call_next(request)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "model_loaded": clf is not None and clf.is_trained,
        "uptime_s": round(time.time() - _start_time, 1),
    }


@app.get("/metrics", tags=["System"])
async def metrics():
    if clf is None:
        raise HTTPException(503, "Model not ready")
    return {
        "total_requests": _request_count,
        "uptime_s": round(time.time() - _start_time, 1),
        "model_metrics": clf.train_metrics,
    }


@app.get("/categories", tags=["Info"])
async def categories():
    return {"categories": CATEGORY_LABELS}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(req: PredictRequest):
    if clf is None or not clf.is_trained:
        raise HTTPException(503, "Model not ready — please retry in a moment")
    result = clf.predict(req.text)
    return PredictResponse(
        category=result.category,
        label=result.label,
        confidence=round(result.confidence, 4),
        all_probabilities={k: round(v, 4) for k, v in result.all_probabilities.items()},
        latency_ms=result.latency_ms,
        top_keywords=result.top_keywords,
    )


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(req: BatchRequest):
    if clf is None or not clf.is_trained:
        raise HTTPException(503, "Model not ready")
    results = clf.predict_batch(req.texts)
    return {
        "results": [
            {
                "text": r.original_text,
                "category": r.category,
                "label": r.label,
                "confidence": round(r.confidence, 4),
                "latency_ms": r.latency_ms,
            }
            for r in results
        ]
    }
