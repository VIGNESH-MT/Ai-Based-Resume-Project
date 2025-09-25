from __future__ import annotations

from typing import Iterable, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
import joblib


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class BertFeaturizer:
    model_name: str = DEFAULT_MODEL_NAME
    device: Optional[str] = None  # "cpu" or "cuda"

    def __post_init__(self) -> None:
        self.model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def fit(self, texts: Iterable[str]) -> "BertFeaturizer":
        # No fitting necessary; keep for API symmetry
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        self._ensure_model()
        embeddings = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return embeddings

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        return self.transform(texts)

    def save(self, path: str) -> None:
        joblib.dump({
            "model_name": self.model_name,
            "device": self.device,
        }, path)

    @classmethod
    def load(cls, path: str) -> "BertFeaturizer":
        data = joblib.load(path)
        return cls(**data)

