"""
Customer Inquiry Classifier — Streamlit Demo
Run: streamlit run streamlit_app.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from app.classifier import CATEGORY_LABELS, CustomerInquiryClassifier, DataGenerator, MODEL_PATH

st.set_page_config(
    page_title="Customer Inquiry Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #6b7280; font-size: 1rem; margin-top: 0; }
    .result-box {
        background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
        border: 1.5px solid #86efac;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
    }
    .warn-box {
        background: #fefce8;
        border: 1.5px solid #fde047;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
    }
    .tag {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model() -> CustomerInquiryClassifier:
    if MODEL_PATH.exists():
        return CustomerInquiryClassifier.load(MODEL_PATH)
    clf = CustomerInquiryClassifier()
    df = DataGenerator().generate(n_samples=4200)
    clf.train(df)
    clf.save(MODEL_PATH)
    return clf


def call_api_single(base_url: str, text: str) -> dict:
    resp = requests.post(
        f"{base_url.rstrip('/')}/predict",
        json={"text": text},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def call_api_batch(base_url: str, texts: list[str]) -> list[dict]:
    resp = requests.post(
        f"{base_url.rstrip('/')}/predict/batch",
        json={"texts": texts},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    confidence_threshold = st.slider(
        "Confidence threshold",
        0.0,
        1.0,
        0.75,
        0.05,
        help="Predictions below this are considered uncertain.",
    )
    mode = st.radio("Mode", ["Single Query", "Batch Analysis"], index=0)

    backend = st.radio("Inference backend", ["Local Model", "API"], index=0)
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    if backend == "API":
        api_base_url = st.text_input("API base URL", value=api_base_url)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    info_placeholder = st.empty()

st.markdown('<p class="main-title">🎯 Customer Inquiry Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Production-grade NLP · confidence routing · optional API mode</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

clf = None
if backend == "Local Model":
    with st.spinner("🔄 Loading local model …"):
        clf = load_model()

    m = clf.train_metrics
    info_placeholder.markdown(
        f"""
| Metric | Value |
|--------|-------|
| Accuracy | **{m.get('accuracy', 0):.1%}** |
| F1 (weighted) | **{m.get('f1_weighted', 0):.1%}** |
| CV Accuracy | **{m.get('cv_accuracy_mean', 0):.1%} ± {m.get('cv_accuracy_std', 0):.1%}** |
| Categories | **7** |
"""
    )
else:
    info_placeholder.markdown(
        """
| Metric | Value |
|--------|-------|
| Backend | **API mode** |
| API URL | dynamic |
| Categories | **7** |
"""
    )


def render_prediction(result: dict, threshold: float):
    confidence = result.get("confidence", 0.0)
    is_confident = confidence >= threshold
    box_class = "result-box" if is_confident else "warn-box"
    icon = "✅" if is_confident else "⚠️"

    st.markdown(
        f'<div class="{box_class}">'
        f'<h3 style="margin:0">{icon} {result.get("label", "Unknown")}</h3>'
        f'<p style="color:#374151;margin:4px 0 0 0">Confidence: <strong>{confidence:.1%}</strong> · '
        f'Latency: <strong>{result.get("latency_ms", 0):.1f} ms</strong></p>'
        f'<p style="color:#374151;margin:4px 0 0 0">Decision: <strong>{result.get("routing_decision", "n/a")}</strong> '
        f'· Team: <strong>{result.get("routed_team", "n/a")}</strong></p>'
        "</div>",
        unsafe_allow_html=True,
    )

    probs = result.get("all_probabilities", {})
    if probs:
        prob_df = pd.DataFrame(
            [
                {"Category": CATEGORY_LABELS.get(k, k), "Probability": v}
                for k, v in sorted(probs.items(), key=lambda item: -item[1])
            ]
        )
        fig = px.bar(
            prob_df,
            x="Probability",
            y="Category",
            orientation="h",
            color="Probability",
            color_continuous_scale="Purples",
            range_x=[0, 1],
            height=300,
            title="Confidence Across All Categories",
        )
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    keywords = result.get("top_keywords", [])
    if keywords:
        st.markdown("**🔍 Key signals detected:**")
        tags_html = " ".join(f'<span class="tag">{kw}</span>' for kw in keywords)
        st.markdown(tags_html, unsafe_allow_html=True)


if mode == "Single Query":
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("### 📝 Enter a customer inquiry")
        user_text = st.text_area(
            "Inquiry text",
            placeholder="I was charged twice and app is crashing, please resolve ASAP",
            height=130,
        )
        classify_btn = st.button("🚀 Classify", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📈 Result")
        if classify_btn and user_text.strip():
            with st.spinner("Classifying …"):
                try:
                    if backend == "Local Model":
                        pred = clf.predict(user_text, confidence_threshold=confidence_threshold)
                        result = {
                            "label": pred.label,
                            "confidence": pred.confidence,
                            "latency_ms": pred.latency_ms,
                            "routing_decision": pred.routing_decision,
                            "routed_team": pred.routed_team,
                            "all_probabilities": pred.all_probabilities,
                            "top_keywords": pred.top_keywords,
                        }
                    else:
                        result = call_api_single(api_base_url, user_text)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                else:
                    render_prediction(result, confidence_threshold)
        elif classify_btn:
            st.info("Please enter some text first.")
        else:
            st.markdown("*Results will appear here after classification.*")

else:
    st.markdown("### 📋 Batch Analysis")
    batch_text = st.text_area(
        "Inquiries (one per line)",
        height=200,
        placeholder="My bill is incorrect\nApp keeps crashing\nWhere is my order?",
    )

    if st.button("🚀 Classify All", type="primary"):
        lines = [line.strip() for line in batch_text.strip().splitlines() if line.strip()]
        if not lines:
            st.warning("Please enter at least one inquiry.")
        elif len(lines) > 50:
            st.error("Maximum 50 inquiries per batch.")
        else:
            with st.spinner(f"Classifying {len(lines)} inquiries …"):
                try:
                    if backend == "Local Model":
                        preds = clf.predict_batch(lines, confidence_threshold=confidence_threshold)
                        results = [
                            {
                                "text": p.original_text,
                                "label": p.label,
                                "confidence": p.confidence,
                                "routing_decision": p.routing_decision,
                                "routed_team": p.routed_team,
                                "latency_ms": p.latency_ms,
                            }
                            for p in preds
                        ]
                    else:
                        results = call_api_batch(api_base_url, lines)
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")
                    results = []

            if results:
                df_out = pd.DataFrame(
                    [
                        {
                            "Inquiry": row.get("text", "")[:80],
                            "Category": row.get("label", "Unknown"),
                            "Confidence": f"{row.get('confidence', 0):.1%}",
                            "Decision": row.get("routing_decision", "n/a"),
                            "Team": row.get("routed_team", "n/a"),
                            "Latency (ms)": row.get("latency_ms", 0),
                        }
                        for row in results
                    ]
                )
                st.dataframe(df_out, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:0.85rem'>"
    "Customer Inquiry Classifier · FastAPI + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
