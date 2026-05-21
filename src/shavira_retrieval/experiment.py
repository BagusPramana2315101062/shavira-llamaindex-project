"""Orkestrasi utama eksperimen retrieval SHAVIRA."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from .constants import REQUIRED_VALIDATION_COLUMNS
from .data_loader import build_nodes, load_documents
from .indexing import build_retrievers, make_faiss_cache_path
from .metrics import evaluate_one_query, make_summary
from .outputs import save_experiment_config, save_result_files
from .retrieval import dedupe_by_context_hash, retrieve_items, rrf_fusion
from .utils import format_preview, make_hash


def run_experiment(args) -> None:
    """Menjalankan seluruh eksperimen dari load data sampai output hasil."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_cache_path = make_faiss_cache_path(args)
    config_path = save_experiment_config(args, output_dir, index_cache_path)

    jsonl_paths = [data_dir / name for name in args.jsonl_files]
    validation_path = data_dir / args.validation_file

    missing = [str(p) for p in jsonl_paths + [validation_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("File berikut belum ditemukan:\n" + "\n".join(missing))

    print("Memuat korpus JSONL")
    documents, stats = load_documents(
        jsonl_paths,
        deduplicate=not args.no_deduplicate,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"Chunking: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}")
    nodes = build_nodes(documents, args.chunk_size, args.chunk_overlap)
    print(f"Total nodes/chunks: {len(nodes)}")

    max_eval_k = max(args.eval_k)
    candidate_k = max(args.candidate_k, max_eval_k)

    bm25_retriever, vector_retriever = build_retrievers(
        nodes=nodes,
        model_name=args.model_name,
        dense_top_k=candidate_k,
        bm25_top_k=candidate_k,
        embed_max_length=args.embed_max_length,
        embed_batch_size=args.embed_batch_size,
        bm25_mode=args.bm25_mode,
        index_cache_dir=index_cache_path,
        force_rebuild_index=args.force_rebuild_index,
    )

    val_df = pd.read_excel(validation_path)
    missing_cols = REQUIRED_VALIDATION_COLUMNS - set(val_df.columns)
    if missing_cols:
        raise ValueError(
            f"Kolom dataset validasi tidak lengkap. Kolom hilang: {sorted(missing_cols)}"
        )

    if args.limit:
        val_df = val_df.head(args.limit).copy()

    val_df["gold_hash"] = val_df["Context"].astype(str).map(make_hash)

    rows_detail: List[Dict[str, Any]] = []
    rows_metric: List[Dict[str, Any]] = []
    started = time.time()

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluasi query"):
        qid = row["ID"]
        query = str(row["Question"])
        gold_hash = str(row["gold_hash"])

        bm25_items = retrieve_items(bm25_retriever, query)
        vector_items = retrieve_items(vector_retriever, query)
        hybrid_items = rrf_fusion(
            [bm25_items, vector_items],
            rrf_k=args.rrf_k,
            top_k=candidate_k,
        )

        method_results = {
            "BM25": bm25_items,
            "BGE_M3_FAISS": vector_items,
            "HYBRID_RRF": hybrid_items,
        }

        for method, items in method_results.items():
            deduped = dedupe_by_context_hash(items)

            for k in args.eval_k:
                metrics = evaluate_one_query(deduped, gold_hash, k)
                rows_metric.append({"ID": qid, "method": method, "k": k, **metrics})

            top10 = deduped[:10]
            rows_detail.append(
                {
                    "ID": qid,
                    "Question": query,
                    "Answer": row["Answer"],
                    "gold_hash": gold_hash,
                    "method": method,
                    "top_hashes": " | ".join([x.text_hash for x in top10]),
                    "top_node_ids": " | ".join([x.node_id for x in top10]),
                    "top_scores": " | ".join(
                        [f"{x.score:.6f}" if not math.isnan(x.score) else "nan" for x in top10]
                    ),
                    "top_titles": " | ".join(
                        [str(x.metadata.get("title", ""))[:80] for x in top10]
                    ),
                    "top_urls": " | ".join(
                        [str(x.metadata.get("url", ""))[:120] for x in top10]
                    ),
                    "top1_preview": format_preview(top10[0].text) if top10 else "",
                    "top1_is_relevant": int(bool(top10 and top10[0].text_hash == gold_hash)),
                }
            )

    metric_df = pd.DataFrame(rows_metric)
    detail_df = pd.DataFrame(rows_detail)
    summary_df = make_summary(metric_df)

    paths = save_result_files(output_dir, summary_df, metric_df, detail_df)
    elapsed = time.time() - started

    print("\n=== SUMMARY ===")
    print(summary_df.to_string(index=False))

    print("\nFile output:")
    print(f"- {paths['summary']}")
    print(f"- {paths['metric']}")
    print(f"- {paths['detail']}")
    print(f"- {paths['xlsx']}")
    print(f"- {config_path}")
    print(f"Durasi evaluasi query: {elapsed:.2f} detik")
