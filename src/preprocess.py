import re
from typing import List

try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore
except Exception:
    nltk = None  # type: ignore
    stopwords = None  # type: ignore
    WordNetLemmatizer = None  # type: ignore


def _load_stopwords() -> set:
    try:
        return set(stopwords.words("english"))  # type: ignore[attr-defined]
    except Exception:
        return {"the", "a", "an", "and", "or", "is", "are", "to", "of", "in", "for", "on", "with", "by", "at"}


def _load_lemmatizer():
    try:
        return WordNetLemmatizer()  # type: ignore[call-arg]
    except Exception:
        return None


_stop_words = _load_stopwords()
_lemmatizer = _load_lemmatizer()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def basic_clean(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = normalize_whitespace(text)
    return text


def tokenize(text: str) -> List[str]:
    if nltk is not None:
        try:
            tokens = nltk.word_tokenize(text)  # type: ignore[attr-defined]
            return [t for t in tokens if t.isalnum()]
        except Exception:
            pass
    return [t for t in text.split() if t.isalnum()]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _stop_words]


def lemmatize(tokens: List[str]) -> List[str]:
    if _lemmatizer is None:
        return tokens
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> str:
    cleaned = basic_clean(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)
