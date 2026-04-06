"""
Customer Inquiry Classifier — Production ML Core
Author: Yash Sharma
Architecture: TF-IDF Ensemble + Calibrated Probabilities + Explainability
"""

import re
import time
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
    _nltk_sw.words("english")  # test it actually loaded
    _NLTK_AVAILABLE = True
except Exception:
    pass

# Compact built-in stopword list (used when NLTK data is unavailable)
_BUILTIN_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","please","help","thank","thanks","hello","hi",
}

# ── Constants ─────────────────────────────────────────────────────────────────
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

CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2CAT = {v: k for k, v in CAT2ID.items()}

MODEL_PATH = Path("models/classifier_v2.joblib")


# ── Data class for a prediction result ───────────────────────────────────────
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


# ── Text preprocessing ────────────────────────────────────────────────────────
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
        # Simple suffix stripping fallback
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


# ── Synthetic data generator ──────────────────────────────────────────────────
class DataGenerator:
    TEMPLATES = {
        "billing": [
            "I have a question about my bill for this month",
            "Why was I charged twice for the same service",
            "Can you explain the charges on my account",
            "I need help understanding my invoice",
            "There's an error in my billing statement",
            "How can I update my payment information",
            "I want to dispute a charge on my bill",
            "I need a copy of my billing history",
            "My payment didn't go through what should I do",
            "I was overcharged for my subscription this month",
            "Can I get an itemized receipt for my purchase",
            "Why did my bill increase from last month",
        ],
        "technical_support": [
            "My device is not working properly and I need help",
            "I am having trouble logging into my account",
            "The application keeps crashing on my phone",
            "I need help setting up my new device",
            "The website is not loading correctly",
            "I forgot my password and cannot reset it",
            "There is a bug in the software that needs fixing",
            "My internet connection is extremely slow",
            "I am getting error messages when using the service",
            "The app will not sync with my other devices",
            "I cannot install the latest software update",
            "My screen keeps freezing every few minutes",
        ],
        "product_inquiry": [
            "What are the features of this product",
            "Do you have this item in different colors or sizes",
            "Can you tell me more about the warranty terms",
            "Is this product compatible with my existing device",
            "What is the difference between these two models",
            "When will new products be available for purchase",
            "I need detailed specifications for this item",
            "Are there any current discounts available",
            "Can I get a product demonstration before buying",
            "What accessories are included with this product",
            "Is this product available in my region",
            "How long does the battery last on this device",
        ],
        "shipping": [
            "When will my order be delivered to my address",
            "I have not received my package and it is overdue",
            "Can I track the status of my shipment online",
            "I need to change my delivery address for my order",
            "My package was damaged during the shipping process",
            "Can I pay extra to expedite my shipping",
            "What shipping options and carriers do you use",
            "I received the wrong item in my order",
            "Can I schedule a specific delivery time window",
            "My tracking number does not show any updates",
            "How long does standard shipping usually take",
            "Can I pick up my order at a local store",
        ],
        "refund_return": [
            "I want to return this item and get a full refund",
            "How do I initiate a refund for my recent purchase",
            "What is your return policy and time window",
            "I am not satisfied with the quality of my order",
            "Can I exchange this product for a different size",
            "I ordered the wrong item completely by mistake",
            "The product I received does not match the description",
            "I need to cancel my recent order before it ships",
            "How long does the refund process usually take",
            "Can I return this item without the original receipt",
            "I want to return a gift I received",
            "My return was rejected and I need to escalate",
        ],
        "account_management": [
            "I want to update my profile and contact information",
            "How do I permanently close my account",
            "I need to change the email address on my account",
            "Can I upgrade my current service or subscription plan",
            "I want to add another authorized user to my account",
            "How do I download all of my personal account data",
            "I need to verify my identity to access my account",
            "Can I merge two separate accounts into one",
            "I want to change my notification and privacy settings",
            "How do I enable two factor authentication for security",
            "I was locked out of my account and cannot get in",
            "Can I transfer my account to a family member",
        ],
        "general_inquiry": [
            "What are your business hours and availability",
            "Do you have a physical store location near me",
            "How can I contact customer service for assistance",
            "What payment methods do you currently accept",
            "Are you hiring for any open positions right now",
            "I have a suggestion for improving your service",
            "Can I speak with a supervisor about my issue",
            "What is your privacy policy regarding my data",
            "Do you offer student or senior discounts",
            "How long has your company been operating",
            "Do you have a loyalty or rewards program",
            "Where can I find your terms and conditions",
        ],
    }

    PREFIXES = [
        "", "Hello, ", "Hi there, ", "Good morning, ", "Hey, ",
        "Please help me. ", "Urgent: ", "Quick question: ",
    ]
    SUFFIXES = [
        "", " Please help.", " Thank you!", " I need assistance.", " ASAP please.",
        " Looking forward to your response.", " Thanks in advance.",
    ]

    def generate(self, n_samples: int = 4200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        records = []
        per_cat = n_samples // len(self.TEMPLATES)

        for cat, templates in self.TEMPLATES.items():
            for _ in range(per_cat):
                base = rng.choice(templates)
                prefix = rng.choice(self.PREFIXES)
                suffix = rng.choice(self.SUFFIXES)
                text = f"{prefix}{base}{suffix}".strip()
                records.append({"text": text, "category": cat, "label": CAT2ID[cat]})

        df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info(f"Generated {len(df)} samples across {len(self.TEMPLATES)} categories")
        return df


# ── Main classifier ───────────────────────────────────────────────────────────
class CustomerInquiryClassifier:
    """
    Production-grade customer inquiry classifier.

    - Calibrated probability estimates via Platt scaling
    - Ensemble of SVM + Logistic Regression for robustness
    - Keyword extraction for lightweight explainability
    - Stratified k-fold evaluation
    - One-line save / load
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.train_metrics: dict = {}

    # ── Build pipeline ────────────────────────────────────────────────────────
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
        # Soft voting ensemble — averages calibrated probabilities
        ensemble = VotingClassifier(
            estimators=[("svc", svc), ("lr", lr)],
            voting="soft",
            weights=[1, 1],
        )
        return Pipeline([("tfidf", tfidf), ("clf", ensemble)])

    # ── Train ─────────────────────────────────────────────────────────────────
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
        y_proba = self.pipeline.predict_proba(X_test)

        report = classification_report(
            y_test, y_pred,
            target_names=CATEGORIES,
            output_dict=True,
        )

        # Stratified CV
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
            f"Training complete | Acc={self.train_metrics['accuracy']:.4f} "
            f"F1={self.train_metrics['f1_weighted']:.4f} "
            f"CV-Acc={self.train_metrics['cv_accuracy_mean']:.4f}±{self.train_metrics['cv_accuracy_std']:.4f}"
        )
        return self.train_metrics

    # ── Predict single text ───────────────────────────────────────────────────
    def predict(self, text: str) -> PredictionResult:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() or .load() first.")

        t0 = time.perf_counter()
        clean = self.preprocessor.clean(text)
        proba = self.pipeline.predict_proba([clean])[0]
        pred_id = int(np.argmax(proba))
        latency = (time.perf_counter() - t0) * 1000

        # Top keywords from TF-IDF vocabulary
        tfidf = self.pipeline.named_steps["tfidf"]
        vec = tfidf.transform([clean]).toarray()[0]
        vocab = tfidf.get_feature_names_out()
        top_idx = vec.argsort()[-6:][::-1]
        keywords = [vocab[i] for i in top_idx if vec[i] > 0]

        return PredictionResult(
            category=ID2CAT[pred_id],
            label=CATEGORY_LABELS[ID2CAT[pred_id]],
            confidence=float(proba[pred_id]),
            all_probabilities={ID2CAT[i]: float(p) for i, p in enumerate(proba)},
            latency_ms=round(latency, 2),
            original_text=text,
            processed_text=clean,
            top_keywords=keywords,
        )

    # ── Batch predict ─────────────────────────────────────────────────────────
    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        return [self.predict(t) for t in texts]

    # ── Persist ───────────────────────────────────────────────────────────────
    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "metrics": self.train_metrics},
            path, compress=3,
        )
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CustomerInquiryClassifier":
        data = joblib.load(path)
        obj = cls()
        obj.pipeline = data["pipeline"]
        obj.train_metrics = data["metrics"]
        obj.is_trained = True
        logger.info(f"Model loaded ← {path}")
        return obj
