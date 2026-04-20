"""
Customer Inquiry Classifier — Production ML Core
Author: Yash Sharma
Architecture: TF-IDF Ensemble + Calibrated Probabilities + Confidence Routing + Optional LLM Fallback
"""

import os
import re
import time
import json
import random
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── NLTK with graceful offline fallback ──────────────────────────────────────
_NLTK_AVAILABLE = False
try:
    import nltk
    for _pkg in ("punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"):
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass
    from nltk.corpus import stopwords as _nltk_sw
    from nltk.tokenize import word_tokenize as _nltk_tokenize
    from nltk.stem import WordNetLemmatizer as _WNL
    _nltk_sw.words("english")
    _NLTK_AVAILABLE = True
except Exception:
    pass

_BUILTIN_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "please", "help", "thank", "thanks", "hello", "hi",
}

CATEGORIES = [
    "billing",
    "technical_support",
    "product_inquiry",
    "shipping",
    "refund_return",
    "account_management",
    "general_inquiry",
]

CATEGORY_LABELS = {
    "billing": "💳 Billing",
    "technical_support": "🔧 Technical Support",
    "product_inquiry": "📦 Product Inquiry",
    "shipping": "🚚 Shipping",
    "refund_return": "↩️ Refund / Return",
    "account_management": "👤 Account Management",
    "general_inquiry": "💬 General Inquiry",
}

ROUTE_MAPPING = {
    "billing": "billing_queue",
    "technical_support": "tech_queue",
    "product_inquiry": "sales_queue",
    "shipping": "logistics_queue",
    "refund_return": "returns_queue",
    "account_management": "account_ops_queue",
    "general_inquiry": "general_support_queue",
}

CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2CAT = {v: k for k, v in CAT2ID.items()}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models/classifier_v2.joblib"


@dataclass
class PredictionResult:
    category: str
    label: str
    confidence: float
    all_probabilities: dict
    latency_ms: float
    original_text: str
    processed_text: str
    top_keywords: list = field(default_factory=list)
    routing_decision: str = "human_review"
    routed_team: str = "human_review_queue"
    requires_human_review: bool = True
    confidence_threshold: float = 0.75
    final_category: str = ""
    llm_fallback_used: bool = False
    llm_prediction: Optional[str] = None
    llm_explanation: Optional[str] = None


class OpenAILLMFallback:
    """Optional LLM fallback/explanation layer using OpenAI Responses API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4.1-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if not self.api_key:
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except Exception as exc:
            logger.warning("OpenAI client unavailable: %s", exc)
            self.client = None

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def classify(self, text: str) -> Optional[dict]:
        if not self.is_available:
            return None

        prompt = (
            "You are a customer support router. "
            f"Choose exactly one category from: {', '.join(CATEGORIES)}. "
            "Return JSON with keys: category, confidence, explanation."
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_output_tokens=120,
            )
            raw = (response.output_text or "").strip()
            if not raw:
                return None
            data = json.loads(raw)
            if data.get("category") not in CATEGORIES:
                return None
            return {
                "category": data.get("category"),
                "confidence": float(data.get("confidence", 0.5)),
                "explanation": str(data.get("explanation", ""))[:220],
            }
        except Exception as exc:
            logger.warning("LLM fallback failed: %s", exc)
            return None


class TextPreprocessor:
    def __init__(self):
        if _NLTK_AVAILABLE:
            self._lemmatizer = _WNL()
            self._stop_words = set(_nltk_sw.words("english"))
        else:
            self._lemmatizer = None
            self._stop_words = _BUILTIN_STOPWORDS

    def _lemmatize(self, token: str) -> str:
        if self._lemmatizer:
            return self._lemmatizer.lemmatize(token)
        for suffix in ("ing", "tion", "ness", "ment", "ed", "er", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) > 3:
                return token[: -len(suffix)]
        return token

    def _tokenize(self, text: str) -> list:
        if _NLTK_AVAILABLE:
            try:
                return _nltk_tokenize(text)
            except Exception:
                pass
        return text.split()

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = self._tokenize(text)
        tokens = [
            self._lemmatize(t)
            for t in tokens
            if t not in self._stop_words and len(t) > 2
        ]
        return " ".join(tokens)


class DataGenerator:
    TEMPLATES = {
        "billing": [
            "I have a question about my bill for this month",
            "Why was I charged twice for the same service",
            "Can you explain the charges on my account",
            "I need help understanding my invoice",
            "There is an error in my billing statement",
            "How can I update my payment information",
            "I want to dispute a charge on my bill",
            "My payment did not go through what should I do",
        ],
        "technical_support": [
            "My device is not working properly and I need help",
            "I am having trouble logging into my account",
            "The application keeps crashing on my phone",
            "I need help setting up my new device",
            "The website is not loading correctly",
            "I forgot my password and cannot reset it",
            "My internet connection is extremely slow",
            "I cannot install the latest software update",
        ],
        "product_inquiry": [
            "What are the features of this product",
            "Do you have this item in different colors or sizes",
            "Can you tell me more about the warranty terms",
            "Is this product compatible with my existing device",
            "What is the difference between these two models",
            "When will new products be available for purchase",
            "Are there any current discounts available",
            "How long does the battery last on this device",
        ],
        "shipping": [
            "When will my order be delivered to my address",
            "I have not received my package and it is overdue",
            "Can I track the status of my shipment online",
            "I need to change my delivery address for my order",
            "My package was damaged during shipping",
            "I received the wrong item in my order",
            "My tracking number does not show any updates",
            "How long does standard shipping usually take",
        ],
        "refund_return": [
            "I want to return this item and get a full refund",
            "How do I initiate a refund for my recent purchase",
            "What is your return policy and time window",
            "I am not satisfied with the quality of my order",
            "Can I exchange this product for a different size",
            "I ordered the wrong item by mistake",
            "How long does the refund process usually take",
            "My return was rejected and I need to escalate",
        ],
        "account_management": [
            "I want to update my profile and contact information",
            "How do I permanently close my account",
            "I need to change the email address on my account",
            "Can I upgrade my current subscription plan",
            "How do I download my personal account data",
            "Can I merge two separate accounts into one",
            "How do I enable two factor authentication",
            "I was locked out of my account and cannot get in",
        ],
        "general_inquiry": [
            "What are your business hours and availability",
            "Do you have a physical store location near me",
            "How can I contact customer service for assistance",
            "What payment methods do you currently accept",
            "Are you hiring for any open positions right now",
            "Can I speak with a supervisor about my issue",
            "Do you offer student or senior discounts",
            "Where can I find your terms and conditions",
        ],
    }

    NOISY_MISSPELLINGS = {
        "refund": "refnd",
        "invoice": "invioce",
        "account": "acount",
        "shipping": "shiping",
        "delivery": "delievery",
        "password": "pasword",
        "charge": "chrage",
        "package": "pakage",
        "application": "aplication",
    }

    TONES = {
        "angry": [
            "This is unacceptable.",
            "I am extremely frustrated right now.",
        ],
        "urgent": [
            "Need this fixed ASAP.",
            "Please respond immediately.",
        ],
        "polite": [
            "Could you please help me with this?",
            "Thank you for your support.",
        ],
        "neutral": [""],
    }

    HINGLISH_SNIPPETS = [
        "pls jaldi help karo",
        "mera order abhi tak nahi aaya",
        "refund kab milega",
        "app baar baar crash ho raha hai",
        "billing me extra charge dikha raha hai",
    ]

    MULTI_INTENT_FRAGMENTS = [
        "Also I need to update my email.",
        "Also tell me where my shipment is.",
        "And I may cancel if this is not resolved.",
        "Also share refund timeline.",
    ]

    SHORT_QUERIES = [
        "refund??",
        "charged twice",
        "app crash",
        "where order",
        "password reset",
        "cancel order now",
    ]

    def _inject_noise(self, text: str, rng: np.random.Generator) -> str:
        noisy = text
        for correct, wrong in self.NOISY_MISSPELLINGS.items():
            if correct in noisy.lower() and rng.random() < 0.35:
                noisy = re.sub(correct, wrong, noisy, flags=re.IGNORECASE)

        if rng.random() < 0.2:
            noisy = noisy.replace(" ", "", 1)
        if rng.random() < 0.25:
            noisy = noisy.replace("?", "") + random.choice(["??", "!!!", "..."])
        return noisy

    def generate(self, n_samples: int = 4200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        random.seed(seed)
        records = []
        per_cat = n_samples // len(self.TEMPLATES)

        for cat, templates in self.TEMPLATES.items():
            for _ in range(per_cat):
                base = rng.choice(templates)

                if rng.random() < 0.18:
                    base = f"{base} {rng.choice(self.MULTI_INTENT_FRAGMENTS)}"
                if rng.random() < 0.12:
                    base = f"{base} {rng.choice(self.HINGLISH_SNIPPETS)}"
                if rng.random() < 0.18:
                    base = rng.choice(self.SHORT_QUERIES)

                tone = rng.choice(list(self.TONES.keys()), p=[0.25, 0.25, 0.25, 0.25])
                tone_phrase = rng.choice(self.TONES[tone])

                if rng.random() < 0.5:
                    text = f"{tone_phrase} {base}".strip()
                else:
                    text = f"{base} {tone_phrase}".strip()

                text = self._inject_noise(text, rng)
                records.append({"text": text, "category": cat, "label": CAT2ID[cat]})

        df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info("Generated %s realistic samples across %s categories", len(df), len(self.TEMPLATES))
        return df


class CustomerInquiryClassifier:
    """Production-grade customer inquiry classifier with confidence routing and optional LLM fallback."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.train_metrics: dict = {}

    def _build_pipeline(self) -> Pipeline:
        tfidf = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_df=0.92,
            analyzer="word",
        )
        svc = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=10_000, random_state=self.random_state),
            cv=3, method="sigmoid",
        )
        lr = LogisticRegression(
            C=5.0, max_iter=1000,
            solver="lbfgs",
            random_state=self.random_state,
        )
        ensemble = VotingClassifier(
            estimators=[("svc", svc), ("lr", lr)],
            voting="soft",
            weights=[1, 1],
        )
        return Pipeline([("tfidf", tfidf), ("clf", ensemble)])

    def train(self, df: pd.DataFrame) -> dict:
        logger.info("Preprocessing text …")
        df = df.copy()
        df["clean"] = df["text"].apply(self.preprocessor.clean)

        X = df["clean"].to_numpy(dtype=str)
        y = df["label"].to_numpy(dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        logger.info("Building & fitting pipeline …")
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=CATEGORIES, output_dict=True)

        logger.info("Running 5-fold stratified CV …")
        cv_pipeline = self._build_pipeline()
        cv_res = cross_validate(
            cv_pipeline, X, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring=["accuracy", "f1_weighted"],
            return_train_score=False,
        )

        self.train_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
            "cv_accuracy_mean": cv_res["test_accuracy"].mean(),
            "cv_accuracy_std": cv_res["test_accuracy"].std(),
            "cv_f1_mean": cv_res["test_f1_weighted"].mean(),
            "cv_f1_std": cv_res["test_f1_weighted"].std(),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        self.is_trained = True
        logger.info(
            "Training complete | Acc=%.4f F1=%.4f CV-Acc=%.4f±%.4f",
            self.train_metrics["accuracy"],
            self.train_metrics["f1_weighted"],
            self.train_metrics["cv_accuracy_mean"],
            self.train_metrics["cv_accuracy_std"],
        )
        return self.train_metrics

    def predict(
        self,
        text: str,
        confidence_threshold: float = 0.75,
        llm_fallback: Optional[OpenAILLMFallback] = None,
        compare_with_llm: bool = False,
    ) -> PredictionResult:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")

        t0 = time.perf_counter()
        clean = self.preprocessor.clean(text)
        proba = self.pipeline.predict_proba([clean])[0]
        pred_id = int(np.argmax(proba))
        ml_category = ID2CAT[pred_id]
        ml_conf = float(proba[pred_id])

        final_category = ml_category
        routing_decision = "auto_route" if ml_conf >= confidence_threshold else "human_review"
        requires_human_review = routing_decision == "human_review"
        routed_team = ROUTE_MAPPING.get(final_category, "human_review_queue") if not requires_human_review else "human_review_queue"

        llm_used = False
        llm_prediction = None
        llm_explanation = None

        if llm_fallback and llm_fallback.is_available and (compare_with_llm or requires_human_review):
            llm_out = llm_fallback.classify(text)
            if llm_out:
                llm_used = True
                llm_prediction = llm_out["category"]
                llm_explanation = llm_out.get("explanation")
                if requires_human_review:
                    final_category = llm_prediction
                    routed_team = ROUTE_MAPPING.get(final_category, "human_review_queue")
                    routing_decision = "llm_fallback_route"
                    requires_human_review = False

        latency = (time.perf_counter() - t0) * 1000

        tfidf = self.pipeline.named_steps["tfidf"]
        vec = tfidf.transform([clean]).toarray()[0]
        vocab = tfidf.get_feature_names_out()
        top_idx = vec.argsort()[-6:][::-1]
        keywords = [vocab[i] for i in top_idx if vec[i] > 0]

        return PredictionResult(
            category=ml_category,
            final_category=final_category,
            label=CATEGORY_LABELS[final_category],
            confidence=ml_conf,
            all_probabilities={ID2CAT[i]: float(p) for i, p in enumerate(proba)},
            latency_ms=round(latency, 2),
            original_text=text,
            processed_text=clean,
            top_keywords=keywords,
            routing_decision=routing_decision,
            routed_team=routed_team,
            requires_human_review=requires_human_review,
            confidence_threshold=confidence_threshold,
            llm_fallback_used=llm_used,
            llm_prediction=llm_prediction,
            llm_explanation=llm_explanation,
        )

    def predict_batch(
        self,
        texts: list[str],
        confidence_threshold: float = 0.75,
        llm_fallback: Optional[OpenAILLMFallback] = None,
        compare_with_llm: bool = False,
    ) -> list[PredictionResult]:
        return [
            self.predict(
                t,
                confidence_threshold=confidence_threshold,
                llm_fallback=llm_fallback,
                compare_with_llm=compare_with_llm,
            )
            for t in texts
        ]

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metrics": self.train_metrics}, path, compress=3)
        logger.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CustomerInquiryClassifier":
        data = joblib.load(path)
        obj = cls()
        obj.pipeline = data["pipeline"]
        obj.train_metrics = data["metrics"]
        obj.is_trained = True
        logger.info("Model loaded ← %s", path)
        return obj
