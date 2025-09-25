from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data.loader import load_resume_file
from src.models.infer import predict, load_artifacts
from src.explain.shap_explain import get_shap_values
from src.fairness.metrics import compute_fairness_metrics

st.set_page_config(page_title="AI Resume Screening with Bias Detection", layout="wide")

# YouTube-themed styles (dark, red accents)
st.markdown(
    """
    <style>
    :root {
        --yt-red: #FF0000;
        --yt-dark: #0f0f0f;
        --yt-mid: #181818;
        --yt-text: #f1f1f1;
        --yt-muted: #aaaaaa;
    }
    .stApp { background-color: var(--yt-dark); color: var(--yt-text); }
    header, .css-18ni7ap, .css-1avcm0n { background-color: var(--yt-dark) !important; }
    section[data-testid="stSidebar"] { background-color: var(--yt-mid) !important; color: var(--yt-text); }
    .stButton>button { background-color: var(--yt-red); color: white; border: none; }
    .stButton>button:hover { filter: brightness(0.9); }
    .stSlider > div[data-baseweb="slider"] > div { background: linear-gradient(90deg, var(--yt-red), #ff6a6a) !important; }
    .stTextInput>div>div>input, .stFileUploader>div>div { background-color: #202020; color: var(--yt-text); border: 1px solid #303030; }
    .stDataFrame, .stTable { background-color: #121212; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3, h4 { color: var(--yt-text); }
    .yt-badge { display:inline-block; padding:2px 8px; border-radius:12px; background:#222; color:#fff; font-size:12px; margin-left:8px; border:1px solid #333; }
    .yt-title { display:flex; align-items:center; gap:10px; }
    .yt-dot { width:10px; height:10px; background:var(--yt-red); border-radius:50%; display:inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="yt-title">
      <span class="yt-dot"></span>
      <h1 style="margin:0;">AI-Powered Resume Screening</h1>
      <span class="yt-badge">Live</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuration")
    model_dir = st.text_input("Model directory", value="artifacts")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**Theme:** YouTube (dark + red)")

st.markdown("Upload one or more resumes (.pdf, .docx, .txt) to classify.")

uploaded_files = st.file_uploader("Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files and model_dir:
    # Save uploaded to temp dir in memory list
    items: List[Tuple[str, str]] = []
    for f in uploaded_files:
        # Save to temp BytesIO then parse by extension
        suffix = Path(f.name).suffix.lower()
        tmp = io.BytesIO(f.read())
        # Write temp file because pdfplumber/docx2txt expect path-like
        tmp_dir = Path(".tmp_uploads")
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / f.name
        with open(tmp_path, "wb") as out:
            out.write(tmp.getvalue())
        text = load_resume_file(tmp_path)
        items.append((f.name, text))

    try:
        # Try real model prediction
        have_model = False
        try:
            from src.models.infer import predict as _predict
            results = _predict(model_dir, items)
            have_model = True
        except Exception:
            # Fall back to heuristic scoring if model artifacts or deps are missing
            def heuristic_score(txt: str) -> float:
                t = txt or ""
                words = len(t.split())
                keywords = [
                    "python", "machine learning", "data", "sql", "nlp", "pandas",
                    "tensorflow", "pytorch", "aws", "docker", "react", "lead"
                ]
                kcount = sum(1 for k in keywords if k in t.lower())
                # Normalize: length contributes up to 0.6, keywords up to 0.4
                length_score = min(words / 600.0, 1.0)
                kw_score = min(kcount / 10.0, 1.0)
                return float(0.6 * length_score + 0.4 * kw_score)

            results = []
            for name, text in items:
                s = heuristic_score(text)
                results.append({"filename": name, "score": s, "pred": int(s >= threshold)})

        df = pd.DataFrame(results)
        df["decision"] = (df["score"] >= threshold).astype(int)
        st.subheader("Predictions")
        # Display resume score and decision
        score_table = df[["filename", "score", "pred", "decision"]]
        st.dataframe(score_table)

        st.subheader("Score Distribution")
        fig = px.histogram(df, x="score", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

        # SHAP explanations for the first resume
        st.subheader("Explainability (SHAP)")
        texts = [t for _, t in items]
        if 'have_model' in locals() and have_model:
            try:
                try:
                    from src.explain.shap_explain import get_shap_values
                except Exception as e:
                    raise RuntimeError("SHAP dependencies not available. Try: pip install shap") from e

                shap_info = get_shap_values(model_dir, [texts[0]])
                shap_vals = shap_info["shap_values"]
                feature_names = shap_info["feature_names"]
                vals = np.array(shap_vals)
                vals = vals[0] if vals.ndim > 2 else vals
                abs_vals = np.abs(vals[0])
                top_idx = np.argsort(-abs_vals)[:20]
                top_df = pd.DataFrame({
                    "feature": [feature_names[i] for i in top_idx],
                    "impact": vals[0][top_idx],
                    "abs_impact": abs_vals[top_idx],
                }).sort_values("abs_impact", ascending=False)
                st.write(top_df[["feature", "impact"]])
            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")
        else:
            st.info("SHAP not available in heuristic mode.")

        # Suggestions based on quick heuristic analysis of text
        st.subheader("Suggestions")
        suggestions = []
        for name, text in items:
            s = []
            length = len(text.split())
            if length < 150:
                s.append("Resume is quite short; consider adding more detail on projects and impact.")
            if "objective" not in text.lower() and "summary" not in text.lower():
                s.append("Add a concise summary/objective at the top.")
            if "experience" not in text.lower():
                s.append("Include a dedicated Experience section with measurable outcomes.")
            if "education" not in text.lower():
                s.append("Include an Education section with degree and year.")
            if "skills" not in text.lower():
                s.append("List relevant technical and soft skills.")
            if not s:
                s.append("Looks good. Consider tailoring keywords to the target role.")
            suggestions.append({"filename": name, "notes": "\n- ".join(s)})

        sug_df = pd.DataFrame(suggestions)
        st.dataframe(sug_df)

        # Fairness metrics if user provides sensitive attribute per file
        st.subheader("Bias Detection")
        st.markdown("Optionally enter a sensitive attribute for each file (e.g., gender) to compute metrics.")
        sens_values = []
        for name in df["filename"].tolist():
            val = st.text_input(f"Sensitive attr for {name}", key=f"sens_{name}")
            sens_values.append((val or "").strip() or "unknown")

        if st.button("Compute fairness metrics"):
            # Validate lengths
            if len(sens_values) != len(df):
                st.warning("Sensitive attribute entries do not match number of predictions.")
            else:
                # When ground truth not available in uploads, use model predictions as proxy for y_true
                # and thresholded decisions as y_pred for fairness rate comparisons.
                y_true = df["pred"].values
                y_pred = df["decision"].values
                # Align sensitive features to the same order and length
                sens = pd.Series(sens_values, index=df.index, name="sensitive")
                try:
                    try:
                        from src.fairness.metrics import compute_fairness_metrics
                    except Exception as e:
                        raise RuntimeError("Fairlearn not available. Try: pip install fairlearn") from e

                    metrics = compute_fairness_metrics(y_true, y_pred, sens)
                    st.write(metrics)
                except Exception as e:
                    st.info(f"Unable to compute fairness metrics: {e}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Provide a valid model directory and upload files to begin.")
