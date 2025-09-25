from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pdfplumber
import docx2txt


SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}


def load_text_from_pdf(path: str | Path) -> str:
    text_parts: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # Extract page text; fall back to empty string if None
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def load_text_from_docx(path: str | Path) -> str:
    text = docx2txt.process(str(path)) or ""
    return text


def load_text_from_txt(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_resume_file(path: str | Path) -> str:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_text_from_pdf(path)
    if ext == ".docx":
        return load_text_from_docx(path)
    if ext == ".txt":
        return load_text_from_txt(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def load_resumes_from_dir(directory: str | Path) -> List[Tuple[str, str]]:
    dir_path = Path(directory)
    files = []
    for file in dir_path.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTS:
            files.append(file)
    files.sort()

    items: List[Tuple[str, str]] = []
    for file in files:
        text = load_resume_file(file)
        items.append((file.name, text))
    return items

