# AI-Powered Resume Screening with Bias Detection

This project provides a complete pipeline to load resumes (PDF/DOCX/TXT), preprocess and featurize text using TF-IDF and BERT embeddings, train classification models (Logistic Regression, Random Forest), explain predictions with SHAP, and detect fairness issues using Fairlearn. It includes a Streamlit app for an end-to-end demo.

## Quickstart

1) Create and activate a virtual environment

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt wordnet omw-1.4
```

3) (Optional) Prepare sample data

Place sample resumes under `sample_data/resumes/` as `.pdf`, `.docx`, or `.txt`. Include a CSV `sample_data/labels.csv` with columns: `filename`, `label`, and optional `gender`, `ethnicity`, or other sensitive attributes for fairness analysis.

Example `labels.csv`:

```csv
filename,label,gender
resume1.pdf,1,F
resume2.docx,0,M
resume3.txt,1,F
```

4) Train models

```bash
python -m src.models.train --data_dir sample_data/resumes --labels_csv sample_data/labels.csv --output_dir artifacts
```

5) Classify resumes and view explanations

```bash
python scripts\load_and_classify.py --model_dir artifacts --input files_to_score
```

6) Launch Streamlit app (YouTube-themed UI)

```bash
streamlit run app/streamlit_app.py
```

## Deploy to GitHub

```bash
git init
git add .
git commit -m "feat: AI resume screening with fairness + SHAP + Streamlit"
git branch -M main
git remote add origin https://github.com/<YOUR_GH_USERNAME>/ai-resume-screening.git
git push -u origin main
```

Replace `<YOUR_GH_USERNAME>` with your GitHub handle (e.g., `VIGNESH-MT`).

## Project Structure

```
src/
  data/loader.py          # PDF/DOCX/TXT loading
  preprocess.py           # Text cleaning
  features/
    tfidf.py              # TF-IDF extractor
    bert.py               # BERT embeddings via sentence-transformers
  models/
    train.py              # Train LR/RF; save artifacts
    infer.py              # Load models and run inference
  explain/
    shap_explain.py       # SHAP explainers
  fairness/
    metrics.py            # Fairlearn metrics and dashboard helpers
scripts/
  load_and_classify.py    # CLI to classify uploaded files and print results
app/
  streamlit_app.py        # Streamlit UI
artifacts/                # Saved models and vectorizers (created after training)
sample_data/
  resumes/                # Place sample resumes here (PDF/DOCX/TXT)
  labels.csv              # Labels + sensitive attrs for fairness analysis
```

## Notes

- BERT model defaults to `sentence-transformers/all-MiniLM-L6-v2` (downloads on first use).
- SHAP explanations are supported for LR and RF models.
- Fairness metrics computed with `fairlearn.metrics.MetricFrame` and displayed in the app.
