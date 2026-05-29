"""Penyimpanan konfigurasi dan output eksperimen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .constants import RETRIEVAL_METHODS
from .utils import clean_dataframe_for_excel


def save_experiment_config(
    args: argparse.Namespace,
    output_dir: Path,
    index_cache_path: str,
) -> Path:
    """Menyimpan konfigurasi eksperimen agar hasil mudah dilacak."""
    config = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "jsonl_files": args.jsonl_files,
        "validation_file": args.validation_file,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "eval_k": args.eval_k,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "model_name": args.model_name,
        "embed_max_length": args.embed_max_length,
        "embed_batch_size": args.embed_batch_size,
        "bm25_mode": args.bm25_mode,
        "deduplicate": not args.no_deduplicate,
        "evaluation_level": "record/context hash",
        "primary_metrics": ["Hit@K", "MRR"],
        "supporting_metrics": ["Precision@K", "Recall@K"],
        "retrieval_methods": RETRIEVAL_METHODS,
        "index_cache_dir": args.index_cache_dir,
        "index_cache_path": index_cache_path,
        "force_rebuild_index": args.force_rebuild_index,
    }

    config_path = output_dir / "experiment_config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config_path


def save_result_files(
    output_dir: Path,
    summary_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    detail_df: pd.DataFrame,
):
    """Menyimpan hasil eksperimen ke CSV dan Excel."""
    metric_path = output_dir / "per_query_metrics.csv"
    detail_path = output_dir / "retrieval_details_top10.csv"
    summary_path = output_dir / "summary_metrics.csv"
    xlsx_path = output_dir / "summary_and_details.xlsx"

    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        clean_dataframe_for_excel(summary_df).to_excel(
            writer,
            sheet_name="summary_metrics",
            index=False,
        )
        clean_dataframe_for_excel(metric_df).to_excel(
            writer,
            sheet_name="per_query_metrics",
            index=False,
        )
        clean_dataframe_for_excel(detail_df).to_excel(
            writer,
            sheet_name="retrieval_details_top10",
            index=False,
        )

    return {
        "summary": summary_path,
        "metric": metric_path,
        "detail": detail_path,
        "xlsx": xlsx_path,
    }
