from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features.tfidf import TfidfFeaturizer
from src.features.bert import BertFeaturizer
from src.preprocess import preprocess_text


def _load_artifacts(model_dir: str | Path) -> Dict:
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    meta = joblib.load(model_dir / "meta.joblib")
    feat_type = meta.get("feature_type")
    if feat_type == "tfidf":
        featurizer = TfidfFeaturizer.load(str(model_dir / "tfidf.joblib"))
        background = None
    else:
        featurizer = BertFeaturizer.load(str(model_dir / "bert.joblib"))
        bg_path = model_dir / "background.npy"
        background = np.load(bg_path) if bg_path.exists() else None
    return {"model": model, "featurizer": featurizer, "feature_type": feat_type, "background": background}


def get_shap_values(model_dir: str | Path, texts: List[str]) -> Dict:
    arts = _load_artifacts(model_dir)
    model = arts["model"]
    featurizer = arts["featurizer"]
    feat_type = arts["feature_type"]
    background = arts["background"]

    preproc = [preprocess_text(t) for t in texts]
    X = featurizer.transform(preproc)

    explainer = None
    if isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
    elif isinstance(model, LogisticRegression) and feat_type == "tfidf":
        explainer = shap.LinearExplainer(model, X if background is None else background)
        shap_vals = explainer.shap_values(X)
    else:
        f = model.predict_proba
        bg = background if background is not None else (X[:50] if hasattr(X, "__getitem__") else None)
        explainer = shap.KernelExplainer(f, bg)
        shap_vals = explainer.shap_values(X, nsamples=100)

    feature_names = (
        featurizer.get_feature_names() if hasattr(featurizer, "get_feature_names") else [f"dim_{i}" for i in range(X.shape[1])]
    )
    base_value = explainer.expected_value if hasattr(explainer, "expected_value") else None
    return {"shap_values": shap_vals, "base_value": base_value, "feature_names": feature_names}
