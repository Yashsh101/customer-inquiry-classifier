"""
Customer Inquiry Classifier — FastAPI REST API
Endpoints: /predict  /predict/batch  /health  /metrics  /categories
"""

from __future__ import annotations
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.classifier import (
    CustomerInquiryClassifier,
    DataGenerator,
    CATEGORY_LABELS,
    MODEL_PATH,
    OpenAILLMFallback,
)

logger = logging.getLogger(__name__)

clf: Optional[CustomerInquiryClassifier] = None
llm_fallback: Optional[OpenAILLMFallback] = None
_request_count = 0
_start_time = time.time()

def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%s. Falling back to %s", name, raw, default)
        value = default
    return max(0.0, min(1.0, value))


ROUTING_CONF_THRESHOLD = _read_float_env("ROUTING_CONF_THRESHOLD", 0.75)
ENABLE_LLM_FALLBACK = os.getenv("ENABLE_LLM_FALLBACK", "false").lower() in {"1", "true", "yes"}
COMPARE_WITH_LLM = os.getenv("COMPARE_WITH_LLM", "false").lower() in {"1", "true", "yes"}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clf, llm_fallback

    if MODEL_PATH.exists():
        clf = CustomerInquiryClassifier.load(MODEL_PATH)
        logger.info("Loaded pre-trained model.")
    else:
        logger.info("No saved model found — training from scratch …")
        clf = CustomerInquiryClassifier()
        df = DataGenerator().generate(n_samples=4200)
        clf.train(df)
        clf.save(MODEL_PATH)

    if ENABLE_LLM_FALLBACK:
        llm_fallback = OpenAILLMFallback(model=OPENAI_MODEL)
        logger.info("LLM fallback enabled=%s model=%s available=%s", ENABLE_LLM_FALLBACK, OPENAI_MODEL, llm_fallback.is_available)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Customer Inquiry Classifier API",
    description="Production-grade NLP routing API with confidence-based decisioning and optional LLM fallback",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000, example="I was charged twice on my credit card this month")

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text must not be blank")
        return v.strip()


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50, example=["My app keeps crashing", "Where is my refund?"])


class PredictResponse(BaseModel):
    category: str
    final_category: str
    label: str
    confidence: float
    confidence_threshold: float
    routing_decision: str
    routed_team: str
    requires_human_review: bool
    all_probabilities: dict[str, float]
    latency_ms: float
    top_keywords: list[str]
    llm_fallback_used: bool
    llm_prediction: Optional[str] = None
    llm_explanation: Optional[str] = None


@app.middleware("http")
async def count_requests(request: Request, call_next):
    global _request_count
    _request_count += 1
    return await call_next(request)


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "model_loaded": clf is not None and clf.is_trained,
        "uptime_s": round(time.time() - _start_time, 1),
        "routing_conf_threshold": ROUTING_CONF_THRESHOLD,
        "llm_fallback_enabled": ENABLE_LLM_FALLBACK,
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

    result = clf.predict(
        req.text,
        confidence_threshold=ROUTING_CONF_THRESHOLD,
        llm_fallback=llm_fallback,
        compare_with_llm=COMPARE_WITH_LLM,
    )
    return PredictResponse(
        category=result.category,
        final_category=result.final_category,
        label=result.label,
        confidence=round(result.confidence, 4),
        confidence_threshold=result.confidence_threshold,
        routing_decision=result.routing_decision,
        routed_team=result.routed_team,
        requires_human_review=result.requires_human_review,
        all_probabilities={k: round(v, 4) for k, v in result.all_probabilities.items()},
        latency_ms=result.latency_ms,
        top_keywords=result.top_keywords,
        llm_fallback_used=result.llm_fallback_used,
        llm_prediction=result.llm_prediction,
        llm_explanation=result.llm_explanation,
    )


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(req: BatchRequest):
    if clf is None or not clf.is_trained:
        raise HTTPException(503, "Model not ready")

    results = clf.predict_batch(
        req.texts,
        confidence_threshold=ROUTING_CONF_THRESHOLD,
        llm_fallback=llm_fallback,
        compare_with_llm=COMPARE_WITH_LLM,
    )
    return {
        "results": [
            {
                "text": r.original_text,
                "category": r.category,
                "final_category": r.final_category,
                "label": r.label,
                "confidence": round(r.confidence, 4),
                "routing_decision": r.routing_decision,
                "routed_team": r.routed_team,
                "requires_human_review": r.requires_human_review,
                "llm_fallback_used": r.llm_fallback_used,
                "llm_prediction": r.llm_prediction,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ]
    }
