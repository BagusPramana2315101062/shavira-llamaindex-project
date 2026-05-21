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
    """Evaluasi satu query dengan satu gold context utama."""
    top = dedupe_by_context_hash(items)[:k]
    rel = [1 if item.text_hash == gold_hash else 0 for item in top]

    hit = 1 if any(rel) else 0
    precision = sum(rel) / k
    recall = float(hit)

    first_rank = 0
    for idx, val in enumerate(rel, start=1):
        if val == 1:
            first_rank = idx
            break

    rr = 1.0 / first_rank if first_rank else 0.0

    # Pada dataset ini setiap query memiliki satu gold context utama.
    # Karena itu AP sama dengan reciprocal rank.
    ap = rr

    return {
        "precision": precision,
        "recall": recall,
        "mrr": rr,
        "map": ap,
        "hit": hit,
        "first_relevant_rank": first_rank,
        "returned_unique_contexts": len(top),
    }


def make_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Membuat ringkasan metrik per metode dan nilai K."""
    return (
        metric_df.groupby(["method", "k"], as_index=False)
        .agg(
            n_queries=("ID", "count"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            mrr=("mrr", "mean"),
            map=("map", "mean"),
            hit_rate=("hit", "mean"),
            avg_first_relevant_rank=(
                "first_relevant_rank",
                lambda s: np.mean([x for x in s if x > 0]) if any(s > 0) else 0,
            ),
        )
        .sort_values(["k", "recall", "mrr"], ascending=[True, False, False])
    )
