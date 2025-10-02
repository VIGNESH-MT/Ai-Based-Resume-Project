from __future__ import annotations
from sklearn.linear_model import LogisticRegression
import joblib
import os

os.makedirs("artifacts", exist_ok=True)
dummy = LogisticRegression()
joblib.dump(dummy, "artifacts/model.joblib")

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.data.loader import load_resumes_from_dir
from src.preprocess import preprocess_text
from src.features.tfidf import TfidfFeaturizer
from src.features.bert import BertFeaturizer


def load_labeled_data(data_dir: str, labels_csv: str) -> Tuple[List[str], List[str], np.ndarray, pd.DataFrame]:
    items = load_resumes_from_dir(data_dir)
    df = pd.read_csv(labels_csv)
    df["filename"] = df["filename"].astype(str)
    name_to_text = {n: t for n, t in items}
    texts: List[str] = []
    names: List[str] = []
    y: List[int] = []
    for _, row in df.iterrows():
        fname = row["filename"]
        if fname not in name_to_text:
            continue
        label = int(row["label"])  # 0/1
        names.append(fname)
        texts.append(name_to_text[fname])
        y.append(label)
    y_arr = np.array(y)
    return names, texts, y_arr, df.set_index("filename").loc[names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_type", choices=["tfidf", "bert"], default="tfidf")
    parser.add_argument("--model_type", choices=["lr", "rf"], default="lr")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    names, texts, y, labels_df = load_labeled_data(args.data_dir, args.labels_csv)
    texts = [preprocess_text(t) for t in texts]

    if args.feature_type == "tfidf":
        featurizer = TfidfFeaturizer()
        X = featurizer.fit_transform(texts)
    else:
        featurizer = BertFeaturizer()
        X = featurizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    if args.model_type == "lr":
        model = LogisticRegression(max_iter=200, n_jobs=None)
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_pred = model.predict(X_test)
        y_proba = y_pred
        auc = None

    report = classification_report(y_test, (y_proba >= 0.5).astype(int))
    print("AUC:", auc)
    print(report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    meta = {"feature_type": args.feature_type, "model_type": args.model_type}
    joblib.dump(meta, out_dir / "meta.joblib")
    if args.feature_type == "tfidf":
        featurizer.save(str(out_dir / "tfidf.joblib"))
    else:
        featurizer.save(str(out_dir / "bert.joblib"))

    # save a small sample for SHAP background if using BERT
    if args.feature_type == "bert":
        bg = X_train[:100] if isinstance(X_train, np.ndarray) else np.asarray(X_train)[:100]
        joblib.dump(bg, out_dir / "background.npy")


if __name__ == "__main__":
    main()


