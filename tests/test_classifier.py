"""
Tests for Customer Inquiry Classifier
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from app.classifier import (
    CustomerInquiryClassifier, DataGenerator, TextPreprocessor,
    CATEGORIES, CAT2ID, ID2CAT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def trained_clf():
    clf = CustomerInquiryClassifier(random_state=42)
    df = DataGenerator().generate(n_samples=1400)
    clf.train(df)
    return clf


# ── Preprocessor tests ────────────────────────────────────────────────────────
class TestPreprocessor:
    def test_lowercasing(self):
        p = TextPreprocessor()
        assert p.clean("HELLO WORLD") == p.clean("hello world")

    def test_removes_urls(self):
        p = TextPreprocessor()
        result = p.clean("visit https://example.com for more info")
        assert "http" not in result and "example" not in result

    def test_returns_string(self):
        p = TextPreprocessor()
        assert isinstance(p.clean("some text"), str)

    def test_empty_string(self):
        p = TextPreprocessor()
        assert p.clean("") == ""


# ── Data generator tests ──────────────────────────────────────────────────────
class TestDataGenerator:
    def test_shape(self):
        df = DataGenerator().generate(n_samples=700)
        assert len(df) >= 700 * 0.9  # allow rounding

    def test_columns(self):
        df = DataGenerator().generate(n_samples=700)
        assert {"text", "category", "label"}.issubset(df.columns)

    def test_all_categories_present(self):
        df = DataGenerator().generate(n_samples=1400)
        assert set(df["category"].unique()) == set(CATEGORIES)

    def test_label_consistency(self):
        df = DataGenerator().generate(n_samples=700)
        for _, row in df.iterrows():
            assert row["label"] == CAT2ID[row["category"]]


# ── Classifier tests ──────────────────────────────────────────────────────────
class TestClassifier:
    def test_train_returns_metrics(self, trained_clf):
        m = trained_clf.train_metrics
        assert "accuracy" in m
        assert "f1_weighted" in m
        assert "cv_accuracy_mean" in m

    def test_accuracy_above_baseline(self, trained_clf):
        # Random baseline = 1/7 ≈ 0.14; well-trained model should beat 0.80
        assert trained_clf.train_metrics["accuracy"] > 0.70

    def test_predict_returns_result(self, trained_clf):
        r = trained_clf.predict("My bill is incorrect this month")
        assert r.category in CATEGORIES
        assert 0.0 <= r.confidence <= 1.0
        assert r.latency_ms > 0


    def test_confidence_routing_fields(self, trained_clf):
        r = trained_clf.predict("I was charged twice on my credit card", confidence_threshold=0.70)
        assert r.routing_decision in {"auto_route", "human_review", "llm_fallback_route"}
        assert isinstance(r.requires_human_review, bool)
        assert r.routed_team

    def test_predict_probabilities_sum_to_one(self, trained_clf):
        r = trained_clf.predict("I want a refund for my order")
        total = sum(r.all_probabilities.values())
        assert abs(total - 1.0) < 1e-4

    def test_batch_predict(self, trained_clf):
        texts = ["my app crashes", "where is my package", "update my email"]
        results = trained_clf.predict_batch(texts)
        assert len(results) == 3

    def test_predict_billing_inquiry(self, trained_clf):
        r = trained_clf.predict("I was charged twice on my credit card this month")
        assert r.category == "billing"

    def test_predict_technical_support(self, trained_clf):
        r = trained_clf.predict("The application keeps crashing on my phone")
        assert r.category == "technical_support"

    def test_predict_shipping(self, trained_clf):
        r = trained_clf.predict("My package has not arrived yet and it is two weeks late")
        assert r.category == "shipping"

    def test_untrained_raises(self):
        clf = CustomerInquiryClassifier()
        with pytest.raises(RuntimeError):
            clf.predict("test")

    def test_save_and_load(self, trained_clf, tmp_path):
        p = tmp_path / "test_model.joblib"
        trained_clf.save(p)
        loaded = CustomerInquiryClassifier.load(p)
        r1 = trained_clf.predict("I need a refund")
        r2 = loaded.predict("I need a refund")
        assert r1.category == r2.category
