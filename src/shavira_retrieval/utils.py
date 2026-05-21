"""Fungsi utilitas umum."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

ILLEGAL_EXCEL_CHARS = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def normalize_text(text: str) -> str:
    """Normalisasi ringan untuk pencocokan exact match berbasis teks."""
    return " ".join(str(text).split()).strip().lower()


def make_hash(text: str) -> str:
    """Membuat hash dari teks yang sudah dinormalisasi."""
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def clean_excel_text(value):
    """Menghapus karakter ilegal agar aman ditulis ke Excel."""
    if isinstance(value, str):
        return ILLEGAL_EXCEL_CHARS.sub("", value)
    return value


def clean_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan seluruh kolom object/string sebelum ditulis ke Excel."""
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(clean_excel_text)
    return df


def safe_name(value: str) -> str:
    """Membuat nama folder cache yang aman untuk Windows."""
    value = str(value)
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    return value.strip("_")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Membaca file JSONL."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada {path.name}, baris {line_no}: {exc}"
                ) from exc


def install_hint() -> str:
    """Pesan bantuan saat dependency belum terpasang."""
    return (
        "Library LlamaIndex/FAISS belum lengkap. Jalankan:\n"
        "pip install -r requirements.txt\n\n"
        "Jika memakai Windows dan faiss-cpu gagal, coba jalankan di Anaconda/Miniconda atau WSL."
    )


def format_preview(text: str, n: int = 260) -> str:
    """Meringkas teks untuk preview hasil retrieval."""
    text = " ".join(str(text).split())
    return text[:n] + ("..." if len(text) > n else "")
