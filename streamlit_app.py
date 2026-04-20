"""
Customer Inquiry Classifier — Streamlit Demo
Run: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from app.classifier import CustomerInquiryClassifier, DataGenerator, CATEGORY_LABELS, MODEL_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Inquiry Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
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
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
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
</style>
""", unsafe_allow_html=True)


# ── Load / train model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if MODEL_PATH.exists():
        return CustomerInquiryClassifier.load(MODEL_PATH)
    clf = CustomerInquiryClassifier()
    df = DataGenerator().generate(n_samples=4200)
    clf.train(df)
    clf.save(MODEL_PATH)
    return clf


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.05,
                                     help="Predictions below this are flagged as uncertain.")
    mode = st.radio("Mode", ["Single Query", "Batch Analysis"], index=0)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    info_placeholder = st.empty()

    st.markdown("---")
    st.markdown("**Built by [Yash Sharma](https://github.com/Yashsh101)**")
    st.markdown("*Stack: scikit-learn · FastAPI · Streamlit*")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎯 Customer Inquiry Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Production-grade NLP · 7 categories · Calibrated confidence · Keyword explainability</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading model …"):
    clf = load_model()

# Update sidebar info
m = clf.train_metrics
info_placeholder.markdown(f"""
| Metric | Value |
|--------|-------|
| Accuracy | **{m.get('accuracy', 0):.1%}** |
| F1 (weighted) | **{m.get('f1_weighted', 0):.1%}** |
| CV Accuracy | **{m.get('cv_accuracy_mean', 0):.1%} ± {m.get('cv_accuracy_std', 0):.1%}** |
| Categories | **7** |
""")


# ── Helper: render a single prediction ───────────────────────────────────────
def render_prediction(result, threshold):
    is_confident = result.confidence >= threshold
    box_class = "result-box" if is_confident else "warn-box"
    icon = "✅" if is_confident else "⚠️"

    st.markdown(
        f'<div class="{box_class}">'
        f'<h3 style="margin:0">{icon} {result.label}</h3>'
        f'<p style="color:#374151;margin:4px 0 0 0">Confidence: <strong>{result.confidence:.1%}</strong> · '
        f'Latency: <strong>{result.latency_ms:.1f} ms</strong></p>'
        f'<p style="color:#374151;margin:4px 0 0 0">Decision: <strong>{result.routing_decision}</strong> · Team: <strong>{result.routed_team}</strong></p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not is_confident:
        st.warning(f"Confidence is below your threshold ({threshold:.0%}). Consider reviewing manually.")

    # Probability bar chart
    prob_df = pd.DataFrame([
        {"Category": CATEGORY_LABELS[k], "Probability": v}
        for k, v in sorted(result.all_probabilities.items(), key=lambda x: -x[1])
    ])
    fig = px.bar(
        prob_df, x="Probability", y="Category", orientation="h",
        color="Probability", color_continuous_scale="Purples",
        range_x=[0, 1], height=300,
        title="Confidence Across All Categories",
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
                      coloraxis_showscale=False, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    # Keywords
    if result.top_keywords:
        st.markdown("**🔍 Key signals detected:**")
        tags_html = " ".join(f'<span class="tag">{kw}</span>' for kw in result.top_keywords)
        st.markdown(tags_html, unsafe_allow_html=True)


# ── Single mode ───────────────────────────────────────────────────────────────
if mode == "Single Query":
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("### 📝 Enter a customer inquiry")

        examples = {
            "Select an example…": "",
            "💳 Billing": "I was charged twice for my subscription this month. Please help.",
            "🔧 Tech Support": "The app keeps crashing every time I try to open it on my iPhone.",
            "📦 Product": "What's the difference between the Pro and Standard versions?",
            "🚚 Shipping": "My order was supposed to arrive 3 days ago. Where is it?",
            "↩️ Return": "I want to return this item. It doesn't match what was advertised.",
            "👤 Account": "How do I enable two-factor authentication on my account?",
            "💬 General": "What are your customer service hours on weekends?",
        }
        chosen = st.selectbox("Quick examples", list(examples.keys()))
        default_text = examples[chosen]

        user_text = st.text_area(
            "Inquiry text", value=default_text,
            placeholder="Type or paste a customer message here…",
            height=130,
        )

        classify_btn = st.button("🚀 Classify", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📈 Result")
        if classify_btn and user_text.strip():
            with st.spinner("Classifying …"):
                result = clf.predict(user_text, confidence_threshold=confidence_threshold)
            render_prediction(result, confidence_threshold)
        elif classify_btn:
            st.info("Please enter some text first.")
        else:
            st.markdown("*Results will appear here after classification.*")

# ── Batch mode ────────────────────────────────────────────────────────────────
else:
    st.markdown("### 📋 Batch Analysis")
    st.markdown("Enter one inquiry per line — up to 50 at a time.")

    batch_text = st.text_area(
        "Inquiries (one per line)",
        height=200,
        placeholder="My bill is incorrect\nApp keeps crashing\nWhere is my order?",
    )

    if st.button("🚀 Classify All", type="primary"):
        lines = [l.strip() for l in batch_text.strip().splitlines() if l.strip()]
        if not lines:
            st.warning("Please enter at least one inquiry.")
        elif len(lines) > 50:
            st.error("Maximum 50 inquiries per batch.")
        else:
            with st.spinner(f"Classifying {len(lines)} inquiries …"):
                results = clf.predict_batch(lines, confidence_threshold=confidence_threshold)

            rows = [
                {
                    "Inquiry": r.original_text[:80] + ("…" if len(r.original_text) > 80 else ""),
                    "Category": r.label,
                    "Confidence": f"{r.confidence:.1%}",
                    "Confident?": "✅" if r.confidence >= confidence_threshold else "⚠️",
                    "Latency (ms)": r.latency_ms,
                }
                for r in results
            ]
            df_out = pd.DataFrame(rows)
            st.dataframe(df_out, use_container_width=True, hide_index=True)

            # Summary pie
            cats = [r.label for r in results]
            fig = px.pie(names=cats, title="Category Distribution",
                         color_discrete_sequence=px.colors.sequential.Purples_r)
            st.plotly_chart(fig, use_container_width=True)

            # Download
            csv = df_out.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv, "classifications.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:0.85rem'>"
    "Customer Inquiry Classifier v2.0 · Built by Yash Sharma · "
    "<a href='https://github.com/Yashsh101/customer-inquiry-classifier' style='color:#8b5cf6'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True,
)
