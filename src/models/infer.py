from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from src.features.tfidf import TfidfFeaturizer
from src.features.bert import BertFeaturizer
from src.preprocess import preprocess_text


def load_artifacts(model_dir: str | Path) -> Dict:
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    meta = joblib.load(model_dir / "meta.joblib")
    feat_type = meta.get("feature_type")
    if feat_type == "tfidf":
        featurizer = TfidfFeaturizer.load(str(model_dir / "tfidf.joblib"))
    elif feat_type == "bert":
        featurizer = BertFeaturizer.load(str(model_dir / "bert.joblib"))
    else:
        raise ValueError("Unknown feature_type in meta.joblib")
    label_names = meta.get("label_names", [0, 1])
    return {"model": model, "featurizer": featurizer, "feature_type": feat_type, "labels": label_names}


def preprocess_batch(texts: List[str]) -> List[str]:
    return [preprocess_text(t) for t in texts]


def predict(model_dir: str | Path, items: List[Tuple[str, str]]) -> List[Dict]:
    art = load_artifacts(model_dir)
    model = art["model"]
    featurizer = art["featurizer"]

    names = [n for n, _ in items]
    texts = [t for _, t in items]
    preprocessed = preprocess_batch(texts)

    X = featurizer.transform(preprocessed)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    results: List[Dict] = []
    for name, p, yhat in zip(names, proba, preds):
        results.append({"filename": name, "score": float(p), "pred": int(yhat)})
    return results

