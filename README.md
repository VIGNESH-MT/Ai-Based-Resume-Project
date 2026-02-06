AI-Powered Resume Screening with Bias Detection

Explainable • Fairness-Aware • End-to-End Hiring Intelligence Pipeline

<p align="center"> <b>Not just resume parsing.</b><br/> <b>A transparent, auditable, and fairness-aware ML system for resume screening.</b> </p> <p align="center"> <a href="#why-this-project">Why This Project</a> • <a href="#system-overview">System Overview</a> • <a href="#capabilities">Capabilities</a> • <a href="#quickstart">Quickstart</a> • <a href="#architecture">Architecture</a> • <a href="#technology">Technology</a> </p>
Why This Project

Most resume screening tools optimize accuracy alone.

That is not sufficient in real hiring systems.

In practice, resume screening models must be:

explainable to recruiters

auditable by compliance teams

measurable for bias and disparate impact

deployable in real workflows

This project demonstrates a complete, production-oriented ML pipeline that treats fairness and explainability as first-class requirements, not afterthoughts.

Hiring models should not just predict — they should justify.

System Overview

This repository implements an end-to-end resume screening system that:

Ingests resumes in PDF / DOCX / TXT

Extracts and preprocesses text

Featurizes resumes using TF-IDF and BERT embeddings

Trains classical ML classifiers

Explains predictions using SHAP

Evaluates fairness using Fairlearn

Exposes results through a Streamlit UI

The system is designed to be:

modular

reproducible

interpretable

extensible

Capabilities
Multi-Format Resume Ingestion

PDF

DOCX

TXT

Unified loading and preprocessing pipeline.

Feature Engineering

TF-IDF for sparse, interpretable signals

BERT embeddings (sentence-transformers/all-MiniLM-L6-v2) for semantic context

Feature strategies can be compared side-by-side.

Classification Models

Logistic Regression (baseline, interpretable)

Random Forest (non-linear benchmark)

Artifacts are versioned and persisted for reuse.

Explainability with SHAP

Local explanations for individual resumes

Global feature importance

Model-agnostic interpretation layer

Predictions are inspectable, not opaque.

Fairness & Bias Detection

Fairness metrics computed using Fairlearn

Supports sensitive attributes such as:

gender

ethnicity

custom protected attributes

Disparity analysis via MetricFrame

This enables measurable bias analysis, not assumptions.

End-to-End Streamlit App

Upload resumes

View predictions

Inspect SHAP explanations

Review fairness metrics interactively

Built for demonstration, validation, and review.

⚡ Quickstart
1️⃣ Create and activate a virtual environment
python -m venv .venv
. .venv/Scripts/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1

2️⃣ Install dependencies
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt wordnet omw-1.4

3️⃣ (Optional) Prepare sample data

Place resumes in:

sample_data/resumes/


Supported formats:

.pdf

.docx

.txt

Create a labels file:

sample_data/labels.csv


Example:

filename,label,gender
resume1.pdf,1,F
resume2.docx,0,M
resume3.txt,1,F


Sensitive attributes are optional but required for fairness analysis.

4️⃣ Train models
python -m src.models.train \
  --data_dir sample_data/resumes \
  --labels_csv sample_data/labels.csv \
  --output_dir artifacts

5️⃣ Run inference on new resumes
python scripts/load_and_classify.py \
  --model_dir artifacts \
  --input files_to_score

6️⃣ Launch Streamlit app
streamlit run app/streamlit_app.py



Architecture
src/
├── data/
│   └── loader.py            # PDF / DOCX / TXT ingestion
│
├── preprocess.py            # Text cleaning & normalization
│
├── features/
│   ├── tfidf.py             # TF-IDF feature extraction
│   └── bert.py              # BERT embeddings
│
├── models/
│   ├── train.py             # Model training & persistence
│   └── infer.py             # Inference pipeline
│
├── explain/
│   └── shap_explain.py      # SHAP explainability
│
├── fairness/
│   └── metrics.py           # Fairlearn metrics & analysis
│
scripts/
└── load_and_classify.py     # CLI for batch scoring
│
app/
└── streamlit_app.py         # Interactive UI
│
artifacts/                   # Saved models & vectorizers
sample_data/
├── resumes/
└── labels.csv


This structure cleanly separates:

data ingestion

modeling

explainability

fairness analysis

presentation layer

🛠 Technology

Python

scikit-learn — classical ML models

sentence-transformers — BERT embeddings

SHAP — explainable AI

Fairlearn — fairness metrics

NLTK — text preprocessing

Streamlit — interactive UI

All libraries are selected for stability, clarity, and reproducibility.

🌟 Why This Repository Stands Out

This is not:

a toy notebook

a black-box model

an accuracy-only demo

This is:

A complete, explainable, fairness-aware ML system for resume screening — designed the way real hiring systems should be built.

If you care about:

responsible AI

explainable ML

hiring fairness

deployable pipelines

👉 This repository is for you.

👤 Author

Vignesh Murugesan
AI / Data Science Engineer

Focus Areas
Explainable AI • Fair ML • Decision Intelligence • Responsible Hiring Systems
