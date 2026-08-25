#!/usr/bin/env python3
"""Q5 (EB-NeRD): generate RecSys 2024 Challenge Codabench prediction files
from BM25F and the semantic scorer (LSA or SBERT+FAISS, whichever
data/models/ebnerd/config.json says scripts/build_indices.py --dataset
ebnerd was last run with).

Mirrors scripts/generate_predictions.py (MIND) in every respect that
carries over -- restricted-candidate scoring only (never full-corpus
retrieval), streamed reading so the 13.5M-impression test set is never
materialized all at once, hyperparameters read from config.json so this
test-corpus index can't drift from the train+val one. It's a separate
script rather than a heavily-branched version of the MIND one because the
raw format genuinely differs: parquet (not TSV) needing chunked reads via
pyarrow (pandas has no `chunksize` for parquet), and history that must be
joined from a *separate* history.parquet by user_id (EB-NeRD doesn't embed
it inline in behaviors the way MIND does).

    ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]

ASSUMPTION, not verified against an official spec file (none is bundled in
this repo, unlike MIND's mind_submission_guidelines.txt): this format
matches MIND's, since the RecSys 2024 Challenge's own tooling is modeled on
MIND's evaluate.py. Verify against the actual Codabench competition page /
the ebnerd-benchmark starter repo before trusting this for a real
submission.

No entity fusion: EB-NeRD ships no entity *embeddings* (see pipeline/ebnerd.py
and retrieval/entity_embeddings.py's docstrings) -- entity_vectors stays
None throughout, semantic_backend must be LSA or SBERT-without-entity-fusion.

    python scripts/build_indices.py --dataset ebnerd          # once, or whenever hyperparameters change
    python scripts/generate_predictions_ebnerd.py --method bm25
    python scripts/generate_predictions_ebnerd.py --method semantic
    python scripts/generate_predictions_ebnerd.py --method both   # default

Writes data/predictions/ebnerd/bm25/{prediction.txt,prediction.zip} and
data/predictions/ebnerd/<semantic_backend>/{prediction.txt,prediction.zip}.
Expect ~5-7x MIND's Q5 runtime (13.5M impressions vs. 2.37M).
"""

import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import ebnerd as ebnerd_loader
from retrieval.build_indices import fit_indices
from retrieval.lsa import mean_pool_user_vector
from retrieval.text_utils import article_text, recency_weights, tokenize


def ranks_from_scores(scores):
    """1-indexed ranks, 1 = highest score. Ties broken deterministically by
    original candidate order (stable sort), so every candidate gets a
    distinct rank -- required by the submission format."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return []


def build_combined_lookups(data_dir, ebnerd_bundle):
    """Text/token lookups span ebnerd_small (train+val) + ebnerd_testset (a
    test impression's history can reference articles clicked during the
    train/val period). Returns
    (combined_ids, combined_texts, token_lookup, test_articles_unified)."""
    small_articles = ebnerd_loader.load_articles(os.path.join(data_dir, ebnerd_bundle))
    small_unified = ebnerd_loader.articles_from_raw_df(small_articles)
    test_articles_raw = ebnerd_loader.load_articles(os.path.join(data_dir, "ebnerd_testset"))
    test_unified = ebnerd_loader.articles_from_raw_df(test_articles_raw)

    combined = pd.concat([small_unified[["article_id", "title", "abstract"]],
                           test_unified[["article_id", "title", "abstract"]]], ignore_index=True)
    combined = combined.drop_duplicates(subset="article_id", keep="first").reset_index(drop=True)

    texts = [article_text(t, a) for t, a in zip(combined["title"], combined["abstract"])]
    token_lookup = {aid: set(tokenize(text)) for aid, text in zip(combined["article_id"], texts)}
    return combined["article_id"].tolist(), texts, token_lookup, test_unified


def build_query_weights(hist, recent_n, decay, token_lookup):
    """Same recency-weighted {term: weight} construction as
    text_utils.weighted_query_terms, but built from a precomputed
    per-article token set -- required for this to stay fast at
    13.5M-impression scale."""
    if not hist:
        return {}
    recent = hist[-recent_n:]
    n = len(recent)
    weights = {}
    for i, aid in enumerate(recent):
        w = decay ** (n - 1 - i)
        for t in token_lookup.get(aid, ()):
            weights[t] = weights.get(t, 0.0) + w
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ebnerd_bundle", default="ebnerd_small", help="train+val bundle used for combined history lookup")
    ap.add_argument("--test_dir", default="data/ebnerd_testset")
    ap.add_argument("--model_dir", default="data/models/ebnerd", help="for config.json (hyperparameters)")
    ap.add_argument("--method", choices=["bm25", "semantic", "both"], default="both")
    ap.add_argument("--chunk_size", type=int, default=50_000, help="behaviors.parquet rows per streamed batch")
    ap.add_argument("--recent_n", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--recency_decay", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--semantic_backend", choices=["lsa", "sbert"], default=None, help="defaults to config.json")
    ap.add_argument("--lsa_components", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--sbert_model", default=None, help="defaults to config.json")
    ap.add_argument("--bm25_k1", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--bm25_b", type=float, default=None, help="defaults to config.json")
    args = ap.parse_args()
    methods_requested = ["bm25", "semantic"] if args.method == "both" else [args.method]

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    recent_n = args.recent_n if args.recent_n is not None else config["recent_n"]
    recency_decay = args.recency_decay if args.recency_decay is not None else config["recency_decay"]
    semantic_backend = args.semantic_backend if args.semantic_backend is not None else config.get("semantic_backend", "lsa")
    lsa_components = args.lsa_components if args.lsa_components is not None else config["lsa_components"]
    sbert_model = args.sbert_model if args.sbert_model is not None else config.get("sbert_model")
    bm25_k1 = args.bm25_k1 if args.bm25_k1 is not None else config["bm25_k1"]
    bm25_b = args.bm25_b if args.bm25_b is not None else config["bm25_b"]
    print(f"Using hyperparameters from {args.model_dir}/config.json: k1={bm25_k1}, b={bm25_b}, "
          f"field_weights={config['bm25_field_weights']}, semantic_backend={semantic_backend}, "
          f"lsa_components={lsa_components}, sbert_model={sbert_model}, "
          f"recent_n={recent_n}, recency_decay={recency_decay} (no entity fusion: EB-NeRD has no entity embeddings)")

    methods = ["bm25" if m == "bm25" else semantic_backend for m in methods_requested]

    print("Building combined article/text lookup (ebnerd_small + ebnerd_testset) ...")
    combined_ids, combined_texts, token_lookup, test_articles = build_combined_lookups(args.data_dir, args.ebnerd_bundle)
    print(f"  {len(combined_ids):,} unique articles in combined lookup, {len(test_articles):,} in the test corpus")

    print("Loading test-set user history (ebnerd_testset/test/history.parquet) ...")
    t0 = time.time()
    history_lookup = ebnerd_loader._build_history_lookup(
        ebnerd_loader.load_history(os.path.join(args.test_dir, "test")))
    print(f"  {len(history_lookup):,} users with history ({time.time()-t0:.1f}s)")

    bm25 = semantic = None
    history_embeddings = None
    if "bm25" in methods_requested or "semantic" in methods_requested:
        bm25_built, semantic_built, test_doc_ids = fit_indices(
            test_articles, semantic_backend=semantic_backend, lsa_components=lsa_components,
            sbert_model=sbert_model, bm25_k1=bm25_k1, bm25_b=bm25_b,
            bm25_field_weights=config["bm25_field_weights"],
            entity_vectors=None,  # EB-NeRD has no entity embeddings -- see module docstring
        )
        bm25 = bm25_built if "bm25" in methods_requested else None
        semantic = semantic_built if "semantic" in methods_requested else None

    if semantic is not None:
        t0 = time.time()
        emb = semantic.embed(combined_ids, combined_texts)
        history_embeddings = dict(zip(combined_ids, emb))
        print(f"  projected {len(combined_ids):,} combined articles for history lookup ({time.time()-t0:.1f}s)")

    out_paths = {}
    fhandles = {}
    for m in methods:
        d = os.path.join(args.data_dir, "predictions", "ebnerd", m)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "prediction.txt")
        out_paths[m] = p
        fhandles[m] = open(p, "w")
    semantic_key = semantic_backend

    beh_path = os.path.join(args.test_dir, "test", "behaviors.parquet")
    pf = pq.ParquetFile(beh_path)
    total_rows = pf.metadata.num_rows
    print(f"\nStreaming {beh_path} ({total_rows:,} rows) in batches of {args.chunk_size:,} ...")
    n_written = 0
    t_start = time.time()
    for batch in pf.iter_batches(batch_size=args.chunk_size, columns=["impression_id", "user_id", "article_ids_inview"]):
        chunk = batch.to_pandas()
        imp_ids = chunk["impression_id"].tolist()
        user_ids = chunk["user_id"].tolist()
        cand_lists = [[str(a) for a in _to_list(x)] for x in chunk["article_ids_inview"].tolist()]

        for imp_id, uid, cand_ids in zip(imp_ids, user_ids, cand_lists):
            hist = history_lookup.get(uid, [])
            recent = hist[-recent_n:] if hist else []

            if bm25 is not None:
                q_weights = build_query_weights(hist, recent_n, recency_decay, token_lookup)
                scores = bm25.score_candidates(q_weights, cand_ids)
                ranks = ranks_from_scores(scores)
                fhandles["bm25"].write(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]\n")

            if semantic is not None:
                embs = [history_embeddings.get(a) for a in recent]
                w = recency_weights(len(recent), recency_decay) if recent else None
                user_vec = mean_pool_user_vector(embs, weights=w)
                scores = semantic.score_candidates(user_vec, cand_ids)
                ranks = ranks_from_scores(scores)
                fhandles[semantic_key].write(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]\n")

        n_written += len(chunk)
        print(f"  {n_written:,}/{total_rows:,} impressions written ({time.time()-t_start:.1f}s elapsed)")

    for f in fhandles.values():
        f.close()

    print(f"\nDone. {n_written:,} predictions written.")
    for m, p in out_paths.items():
        zip_path = os.path.join(os.path.dirname(p), "prediction.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(p, arcname="prediction.txt")
        size_mb = os.path.getsize(zip_path) / 1e6
        print(f"  {m}: {p}  ->  {zip_path} ({size_mb:.1f} MB)")

    print("\nUpload the zip(s) to https://www.codabench.org/competitions/2469/")


if __name__ == "__main__":
    main()
