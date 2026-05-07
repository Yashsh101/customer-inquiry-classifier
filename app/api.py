"""FastAPI REST API for the Customer Inquiry Classifier."""

from __future__ import annotations
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.classifier import (
    CustomerInquiryClassifier,
    DataGenerator,
    CATEGORY_LABELS,
    MODEL_PATH,
    OpenAILLMFallback,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)

FRONTEND_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Customer Classifier Ops</title>
    <meta
      name="description"
      content="A production-ready customer inquiry classifier with FastAPI, Vercel, and confidence-based ML routing."
    />
    <link rel="canonical" href="https://customer-inquiry-classifier.vercel.app/" />
    <link rel="stylesheet" href="/styles.css" />
  </head>
  <body>
    <div class="ambient"></div>
    <main class="shell">
      <nav class="topbar" aria-label="Project">
        <div class="brand">
          <span class="brand-mark">CI</span>
          <span>Classifier Ops</span>
        </div>
        <a class="live-link" href="/api/health">API status</a>
      </nav>

      <section class="intro">
        <div>
          <p class="eyebrow">FastAPI + Vercel + ML routing</p>
          <h1>Customer Inquiry Classifier</h1>
          <p class="lede">
            A production-style triage console for classifying support messages, reading model confidence,
            and routing high-certainty tickets to the right team.
          </p>
        </div>
        <div class="status" id="healthStatus">Checking API...</div>
      </section>

      <section class="stats" aria-label="Model highlights">
        <div><span>7</span><p>support categories</p></div>
        <div><span>&lt;1s</span><p>typical inference</p></div>
        <div><span>ML</span><p>confidence routing</p></div>
        <div><span>Vercel</span><p>serverless deploy</p></div>
      </section>

      <section class="workspace">
        <form class="panel input-panel" id="singleForm">
          <div class="panel-head">
            <div>
              <h2>Single Inquiry</h2>
              <p>Paste a customer message and classify it in real time.</p>
            </div>
            <button type="submit" id="classifyBtn">Classify</button>
          </div>
          <textarea
            id="inquiryText"
            maxlength="2000"
            placeholder="I was charged twice this month and the app keeps crashing when I try to request a refund."
          ></textarea>
          <div class="samples" aria-label="Sample inquiries">
            <button type="button" class="sample" data-sample="I was charged twice this month and need a refund immediately.">Billing</button>
            <button type="button" class="sample" data-sample="The mobile app crashes every time I try to log in.">Technical</button>
            <button type="button" class="sample" data-sample="My package has not arrived and tracking has not updated.">Shipping</button>
          </div>
          <p class="hint">3-2000 characters. Low-confidence cases are flagged for review.</p>
        </form>

        <section class="panel result-panel" aria-live="polite">
          <div class="panel-head">
            <div>
              <h2>Result</h2>
              <p>Prediction, confidence, routing decision, and top signals.</p>
            </div>
          </div>
          <div id="singleResult" class="empty">
            <span class="empty-icon">→</span>
            <strong>Ready for a ticket.</strong>
            <p>Classification, routing decision, latency, and signals will appear here.</p>
          </div>
        </section>
      </section>

      <section class="panel batch-panel">
        <div class="panel-head">
          <div>
            <h2>Batch Analysis</h2>
            <p>One inquiry per line, up to 50 messages.</p>
          </div>
          <button type="button" id="batchBtn">Classify All</button>
        </div>
        <textarea id="batchText" placeholder="My bill is incorrect&#10;The app crashes on login&#10;Where is my package?"></textarea>
        <div id="batchResult" class="table-wrap"></div>
      </section>
    </main>
    <script src="/app.js"></script>
  </body>
</html>
"""

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


router = APIRouter()


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


async def count_requests(request: Request, call_next):
    global _request_count
    _request_count += 1
    return await call_next(request)


@router.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "model_loaded": clf is not None and clf.is_trained,
        "uptime_s": round(time.time() - _start_time, 1),
        "routing_conf_threshold": ROUTING_CONF_THRESHOLD,
        "llm_fallback_enabled": ENABLE_LLM_FALLBACK,
    }


@router.get("/metrics", tags=["System"])
async def metrics():
    if clf is None:
        raise HTTPException(503, "Model not ready")
    return {
        "total_requests": _request_count,
        "uptime_s": round(time.time() - _start_time, 1),
        "model_metrics": clf.train_metrics,
    }


@router.get("/categories", tags=["Info"])
async def categories():
    return {"categories": CATEGORY_LABELS}


@router.post("/predict", response_model=PredictResponse, tags=["Inference"])
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


@router.post("/predict/batch", tags=["Inference"])
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


def create_app() -> FastAPI:
    api_app = FastAPI(
        title="Customer Inquiry Classifier API",
        description="NLP routing API with confidence-based decisioning and optional LLM fallback",
        version="4.0.0",
        lifespan=lifespan,
    )
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_app.middleware("http")(count_requests)
    api_app.include_router(router, prefix="/api")

    public_dir = PROJECT_ROOT / "public"

    @api_app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def frontend_index():
        index_file = public_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse(FRONTEND_HTML)

    if public_dir.exists():
        api_app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")

    return api_app


app = create_app()
