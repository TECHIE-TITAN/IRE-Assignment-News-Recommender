#!/usr/bin/env python3
"""A2 Q4 -- Serving & Scale Analysis.

Measures the two things Q4 asks for directly, on THIS machine, against the
actual persisted indices/model (not estimated):

  1. Index memory: incremental process RSS (via `resource.getrusage`, NOT
     pickle file size -- a pickle's on-disk size and its live in-memory
     footprint diverge for these objects, e.g. BM25Index's sparse matrix
     compresses well on disk but not in RAM, and SBERTIndex's FAISS index
     is rebuilt fresh from the pickled embeddings on load, see
     retrieval/build_indices.py's load_indices docstring) after loading
     each component, in the SAME load order generate_predictions.py uses
     (reranker.pkl before the FAISS-backed semantic index -- seg-faults
     otherwise on this machine, see that script's module docstring).

  2. p99 single-request latency for "candidate generation + re-ranking"
     (Q4's literal wording): for N sampled val impressions, times the FULL
     per-impression serving path -- Assignment-1 retrieval scores
     (retrieval.reranker_scores.compute_retrieval_scores) + Q1 behavioural
     features (pipeline.features_{mind,ebnerd}, called on a ONE-impression
     chunk, i.e. paying the same per-call DataFrame-construction overhead a
     live request would) + LGBMRanker.predict() -- individually, not
     batched, since batching would understate real single-request latency.

Also prints a back-of-envelope cost/QPS estimate and a 10x scaling
argument DERIVED from the measured numbers (not fabricated) -- Q4 asks for
"a measured local benchmark plus a scaling argument," not a real
multi-node load test.

    python scripts/benchmark_serving.py --dataset mind --n_requests 1000
    python scripts/benchmark_serving.py --dataset ebnerd --n_requests 1000

Single-threaded, single-process on purpose: this measures ONE request's
serving cost in isolation (Q4.2's "single request" latency), which is what
the cost/QPS extrapolation below needs as its input. It does NOT measure
this project's actual candidate-generation throughput under concurrent
load -- scripts/generate_predictions.py's own multiprocess measured
throughput (thousands of impressions/sec across N workers, see its module
docstring) is the closer real analogue for THAT question, and is cited
below instead of re-measured here.
"""

import argparse
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ebnerd import load_articles as load_ebnerd_articles_raw
from pipeline.features_common import compute_first_seen_times
from pipeline.features_ebnerd import build_ebnerd_candidate_features, precompute_session_engagement
from pipeline.features_mind import build_mind_candidate_features
from retrieval.build_indices import load_indices
from retrieval.reranker_data import FEATURE_COLUMNS
from retrieval.reranker_scores import compute_retrieval_scores
from retrieval.text_utils import article_text

# macOS/BSD reports ru_maxrss in BYTES; Linux reports it in KB. Getting this
# wrong silently produces a 1000x-off memory number, so it's handled
# explicitly rather than assumed.
_RSS_UNIT_KB = 1 / 1024 if sys.platform == "darwin" else 1


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_UNIT_KB / 1024.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default=None, help="defaults to data/models/<dataset>")
    ap.add_argument("--n_requests", type=int, default=1000, help="val impressions to sample for latency timing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_p99_ms", type=float, default=100.0, help="SLA target for the cost/QPS estimate")
    ap.add_argument("--ebnerd_bundle", default="ebnerd_small")
    args = ap.parse_args()

    import pickle

    model_dir = args.model_dir or os.path.join("data", "models", args.dataset)
    proc_dir = os.path.join(args.data_dir, "processed", args.dataset)

    mem = {"baseline": rss_mb()}

    # Load order matches generate_predictions.py: reranker.pkl (LightGBM)
    # BEFORE the semantic index (FAISS, if this run used the sbert
    # backend) -- see that script's module docstring for the reproducible
    # segfault this order avoids on this machine.
    with open(os.path.join(model_dir, "reranker.pkl"), "rb") as f:
        reranker = pickle.load(f)
    mem["after_reranker_model"] = rss_mb()

    bm25, semantic, config = load_indices(model_dir)
    mem["after_bm25_and_semantic_index"] = rss_mb()

    recent_n, recency_decay = config["recent_n"], config["recency_decay"]
    feature_cols = FEATURE_COLUMNS[args.dataset]

    articles = pd.read_parquet(os.path.join(proc_dir, "articles.parquet"))
    interactions = pd.read_parquet(os.path.join(proc_dir, "interactions.parquet"))
    article_pop = pd.read_parquet(
        os.path.join(args.data_dir, "feature_store", args.dataset, "article_features.parquet")
    ).set_index("article_id")[["train_ctr", "train_impressions"]]
    text_lookup = dict(zip(articles["article_id"],
                            (article_text(t, a) for t, a in zip(articles["title"], articles["abstract"]))))
    entity_lookup = dict(zip(articles["article_id"], articles["entities"]))
    category_lookup = dict(zip(articles["article_id"], articles["category"]))
    first_seen = compute_first_seen_times(interactions)
    mem["after_lookups_and_feature_store"] = rss_mb()

    if args.dataset == "mind":
        subcategory_lookup = dict(zip(articles["article_id"], articles["subcategory"]))

        def build_features(chunk):
            return build_mind_candidate_features(chunk, category_lookup, subcategory_lookup, article_pop,
                                                   first_seen, semantic, recent_n, recency_decay)
    else:
        ebnerd_dir = os.path.join(args.data_dir, args.ebnerd_bundle)
        raw_articles = load_ebnerd_articles_raw(ebnerd_dir)
        published_time_lookup = dict(zip(raw_articles["article_id"].astype(str),
                                          pd.to_datetime(raw_articles["published_time"], errors="coerce")))
        session_engagement = precompute_session_engagement(ebnerd_dir)

        def build_features(chunk):
            return build_ebnerd_candidate_features(chunk, category_lookup, published_time_lookup, article_pop,
                                                     first_seen, session_engagement, semantic, recent_n, recency_decay)
        mem["after_ebnerd_session_engagement_precompute"] = rss_mb()

    val = interactions[interactions["split"] == "val"].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_requests, len(val))
    sample_idx = rng.choice(len(val), size=n, replace=False)

    print(f"\nDataset={args.dataset}, timing {n:,} single-impression requests "
          f"(candidate generation + re-ranking, one at a time) ...")
    latencies_ms = []
    for idx in sample_idx:
        imp = val.iloc[[idx]].reset_index(drop=True)
        history = imp.iloc[0]["history_article_ids"]
        candidates = imp.iloc[0]["candidate_article_ids"]
        if len(candidates) == 0:
            continue

        t0 = time.perf_counter()
        bm25_scores, semantic_scores, fusion_scores = compute_retrieval_scores(
            history, candidates, bm25, semantic, text_lookup, entity_lookup, recent_n, recency_decay, 0.7)
        feats = build_features(imp)
        if feats.empty:
            continue
        feats = feats.assign(bm25_score=bm25_scores, semantic_score=semantic_scores, fusion_score=fusion_scores)
        _ = reranker.predict(feats[feature_cols])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(latencies_ms)
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    print(f"  {len(lat):,} requests timed; candidates/request: mean={np.mean([len(val.iloc[i]['candidate_article_ids']) for i in sample_idx]):.1f}")
    print(f"  latency (ms): mean={lat.mean():.2f} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} max={lat.max():.2f}")

    print("\n=== Index memory (incremental process RSS, MB) ===")
    prev = mem["baseline"]
    for label, val_mb in mem.items():
        if label == "baseline":
            print(f"  {label:<40}{val_mb:>10.1f} MB (process baseline)")
            continue
        print(f"  {label:<40}{val_mb:>10.1f} MB total  (+{val_mb - prev:>8.1f} MB this step)")
        prev = val_mb

    # Back-of-envelope cost/QPS at a target SLA: if p99 must stay under
    # target_p99_ms on ONE single-threaded worker, that worker can sustain
    # roughly 1000/p99 requests/sec before queueing pushes p99 past the
    # target (a conservative approximation -- ignores queueing-theory
    # effects near saturation, which would push real p99 higher at the
    # same throughput; treat this as an upper bound on safe QPS/worker).
    safe_qps_per_worker = 1000.0 / p99 if p99 > 0 else float("inf")
    print(f"\n=== Cost/QPS back-of-envelope (target p99 < {args.target_p99_ms:.0f}ms) ===")
    print(f"  measured p99 = {p99:.2f}ms -> ~{safe_qps_per_worker:.1f} req/sec/worker before p99 risks exceeding target")
    for target_qps in [100, 1000, 10000]:
        workers = max(1, int(np.ceil(target_qps / safe_qps_per_worker)))
        print(f"  to sustain {target_qps:>6,} QPS: ~{workers:>4} workers "
              f"(illustrative only -- no real cloud pricing applied; see design note)")

    report = {
        "dataset": args.dataset, "n_requests_timed": int(len(lat)),
        "latency_ms": {"mean": float(lat.mean()), "p50": float(p50), "p95": float(p95),
                        "p99": float(p99), "max": float(lat.max())},
        "index_memory_mb": mem,
        "safe_qps_per_worker_at_target_p99": float(safe_qps_per_worker),
        "target_p99_ms": args.target_p99_ms,
    }
    import json
    report_path = os.path.join(args.data_dir, "reports", f"{args.dataset}_serving_benchmark.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
