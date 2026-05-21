"""Command line interface untuk eksperimen retrieval SHAVIRA."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .constants import (
    DEFAULT_BM25_MODE,
    DEFAULT_CANDIDATE_K,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_EMBED_MAX_LENGTH,
    DEFAULT_EVAL_K,
    DEFAULT_INDEX_CACHE_DIR,
    DEFAULT_JSONL_FILES,
    DEFAULT_MODEL_NAME,
    DEFAULT_RRF_K,
    DEFAULT_VALIDATION_FILE,
)
from .experiment import run_experiment


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluasi retrieval SHAVIRA dengan LlamaIndex"
    )

    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="results")

    parser.add_argument("--jsonl-files", nargs="+", default=DEFAULT_JSONL_FILES)
    parser.add_argument("--validation-file", default=DEFAULT_VALIDATION_FILE)

    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--eval-k", type=int, nargs="+", default=DEFAULT_EVAL_K)

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Jumlah kandidat awal BM25/FAISS untuk RRF dan evaluasi.",
    )

    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--embed-max-length", type=int, default=DEFAULT_EMBED_MAX_LENGTH)
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)

    parser.add_argument(
        "--bm25-mode",
        default=DEFAULT_BM25_MODE,
        choices=["default", "none", "no_stopwords", "no-stopwords", "english", "en"],
        help="Mode BM25. Default berarti tanpa bahasa khusus dan tanpa custom stopwords.",
    )

    parser.add_argument(
        "--index-cache-dir",
        default=DEFAULT_INDEX_CACHE_DIR,
        help="Folder utama untuk menyimpan cache FAISS index agar embedding tidak diulang.",
    )

    parser.add_argument(
        "--force-rebuild-index",
        action="store_true",
        help="Paksa bangun ulang FAISS index meskipun cache sudah ada.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Batasi jumlah query untuk uji cepat. 0 = semua.",
    )

    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Jangan hapus duplikasi record JSONL.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
