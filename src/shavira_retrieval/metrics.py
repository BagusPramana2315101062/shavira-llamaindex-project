"""Perhitungan metrik evaluasi retrieval."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .retrieval import RetrievedItem, dedupe_by_context_hash


def evaluate_one_query(
    items: Sequence[RetrievedItem],
    gold_hash: str,
    k: int,
) -> Dict[str, float]:
    """Evaluasi satu query berdasarkan satu gold context utama.

    Metrik utama:
    - Hit@K: 1 jika gold context ditemukan dalam top-K, 0 jika tidak.
    - MRR: reciprocal rank dari posisi pertama gold context.

    Precision@K dan Recall@K tetap dihitung sebagai metrik pendukung.
    MAP tidak digunakan karena pada skenario satu gold context utama,
    Average Precision akan setara dengan Reciprocal Rank.
    """
    top = dedupe_by_context_hash(items)[:k]

    rel = [1 if item.text_hash == gold_hash else 0 for item in top]

    hit_at_k = 1 if any(rel) else 0

    # Precision tetap dihitung sebagai metrik pendukung.
    # Pada single gold context, precision maksimum adalah 1/K.
    precision = sum(rel) / k

    # Pada single gold context, recall ekuivalen dengan Hit@K.
    recall = float(hit_at_k)

    first_rank = 0
    for idx, val in enumerate(rel, start=1):
        if val == 1:
            first_rank = idx
            break

    mrr = 1.0 / first_rank if first_rank else 0.0

    return {
        "hit_at_k": hit_at_k,
        "mrr": mrr,
        "precision": precision,
        "recall": recall,
        "first_relevant_rank": first_rank,
        "returned_unique_contexts": len(top),
    }


def make_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Membuat ringkasan metrik per metode dan nilai K.

    Hit@K dan MRR menjadi metrik utama.
    Precision@K dan Recall@K disajikan sebagai metrik pendukung.
    """
    return (
        metric_df.groupby(["method", "k"], as_index=False)
        .agg(
            n_queries=("ID", "count"),
            hit_at_k=("hit_at_k", "mean"),
            mrr=("mrr", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            avg_first_relevant_rank=(
                "first_relevant_rank",
                lambda s: np.mean([x for x in s if x > 0])
                if (s > 0).any()
                else 0,
            ),
        )
        .sort_values(["k", "hit_at_k", "mrr"], ascending=[True, False, False])
    )