from __future__ import annotations

from typing import Iterable, List, Optional
from dataclasses import dataclass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as e:
    # Provide a clearer error if scikit-learn is missing
    raise ImportError(
        "scikit-learn is required for TF-IDF features. Install with: pip install scikit-learn"
    ) from e
import joblib


@dataclass
class TfidfFeaturizer:
    max_features: int = 20000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95

    def __post_init__(self) -> None:
        self.vectorizer: Optional[TfidfVectorizer] = None

    def fit(self, texts: Iterable[str]) -> "TfidfFeaturizer":
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: Iterable[str]):
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted. Call fit() first or load().")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: Iterable[str]):
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str) -> None:
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not initialized")
        joblib.dump({
            "vectorizer": self.vectorizer,
            "config": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
                "max_df": self.max_df,
            },
        }, path)

    @classmethod
    def load(cls, path: str) -> "TfidfFeaturizer":
        data = joblib.load(path)
        cfg = data.get("config", {})
        inst = cls(**cfg)
        inst.vectorizer = data["vectorizer"]
        return inst

    def get_feature_names(self) -> List[str]:
        if self.vectorizer is None:
            return []
        return list(self.vectorizer.get_feature_names_out())
