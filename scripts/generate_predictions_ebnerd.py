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

    python scripts/build_indices.py --dataset ebnerd            # once, or whenever hyperparameters change
    python scripts/train_reranker.py --dataset ebnerd            # once, if you also want --method reranker
    python scripts/generate_predictions_ebnerd.py --method bm25
    python scripts/generate_predictions_ebnerd.py --method semantic
    python scripts/generate_predictions_ebnerd.py --method both       # default
    python scripts/generate_predictions_ebnerd.py --method reranker   # needs data/models/ebnerd/reranker.pkl

`--method reranker` replicates the trained re-ranker's behavioural
features (retrieval.reranker_data.FEATURE_COLUMNS["ebnerd"]) per candidate
inside the streaming loop -- see `_reranker_feature_matrix_ebnerd` and
`combined_session_engagement`. Unlike MIND, ebnerd_testset's own
behaviors.parquet actually ships session_id/read_time/scroll_percentage
(confirmed against the real file), so the session/engagement features
ARE computed for real, not defaulted to zero -- except
session_prior_click_count, which is genuinely unknowable for test rows
(no article_ids_clicked column exists in the test file at all) and is
therefore 0 there by necessity, not by approximation.

Writes data/predictions/ebnerd/{bm25,<semantic_backend>,reranker}/
{prediction.txt,prediction.zip}. Expect ~5-7x MIND's prediction-generation
runtime (13.5M impressions vs. 2.37M) -- `--method reranker` costs more
still, both for the extra per-candidate feature computation and the
one-time train+val+test session/engagement precompute at startup.
"""

import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pickle

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import ebnerd as ebnerd_loader
from pipeline.features_common import (
    compute_first_seen_times,
    days_since_first_seen_scalar,
    history_embedding_similarity_map,
    position_bias_map,
    weighted_category_match_map,
)
from pipeline.features_ebnerd import _session_engagement_from_frame
from retrieval.build_indices import fit_indices
from retrieval.fusion import fuse_scores
from retrieval.lsa import mean_pool_user_vector
from retrieval.reranker_data import FEATURE_COLUMNS
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
    """Text/token/category lookups span ebnerd_small (train+val) +
    ebnerd_testset (a test impression's history can reference articles
    clicked during the train/val period). Returns (combined_ids,
    combined_texts, token_lookup, test_articles_unified, category_lookup).
    `category_lookup` is only needed for --method reranker (Q1's
    history_category_match feature; EB-NeRD has no usable subcategory
    field for a match feature, see pipeline/features_ebnerd.py)."""
    small_articles = ebnerd_loader.load_articles(os.path.join(data_dir, ebnerd_bundle))
    small_unified = ebnerd_loader.articles_from_raw_df(small_articles)
    test_articles_raw = ebnerd_loader.load_articles(os.path.join(data_dir, "ebnerd_testset"))
    test_unified = ebnerd_loader.articles_from_raw_df(test_articles_raw)

    cols = ["article_id", "title", "abstract", "category"]
    combined = pd.concat([small_unified[cols], test_unified[cols]], ignore_index=True)
    combined = combined.drop_duplicates(subset="article_id", keep="first").reset_index(drop=True)

    texts = [article_text(t, a) for t, a in zip(combined["title"], combined["abstract"])]
    token_lookup = {aid: set(tokenize(text)) for aid, text in zip(combined["article_id"], texts)}
    category_lookup = dict(zip(combined["article_id"], combined["category"]))
    return combined["article_id"].tolist(), texts, token_lookup, test_unified, category_lookup


def load_reranker_assets(data_dir, model_dir, ebnerd_bundle, test_dir):
    """Loads the trained re-ranker plus the lookups its behavioural
    features need. `reference_time_lookup`: published_time (spans
    ebnerd_small + testset's own raw articles.parquet -- both ship a real
    published_time, unlike MIND) where known, else `first_seen_times`
    (train+val only -- EB-NeRD's test set isn't folded into
    processed/ebnerd/interactions.parquet the way MIND's is, see
    build_pipeline.py) -- merged ONCE here (not per-candidate in the hot
    loop), matching pipeline/features_ebnerd.py's own precedence.
    `session_engagement` is pre-converted to a plain dict-of-dicts (not
    left as a DataFrame) for cheap per-impression lookups across 13.5M
    rows -- see `combined_session_engagement`."""
    with open(os.path.join(model_dir, "reranker.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "reranker_config.json")) as f:
        rr_config = json.load(f)

    article_features = pd.read_parquet(os.path.join(data_dir, "feature_store", "ebnerd", "article_features.parquet"))
    ctr_lookup = dict(zip(article_features["article_id"], article_features["train_ctr"].fillna(0.0)))
    impressions_lookup = dict(zip(article_features["article_id"], article_features["train_impressions"].fillna(0)))

    small_raw = ebnerd_loader.load_articles(os.path.join(data_dir, ebnerd_bundle))
    test_raw = ebnerd_loader.load_articles(test_dir)
    published = pd.concat([small_raw[["article_id", "published_time"]], test_raw[["article_id", "published_time"]]])
    published_time_lookup = dict(zip(published["article_id"].astype(str),
                                       pd.to_datetime(published["published_time"], errors="coerce")))

    interactions = pd.read_parquet(os.path.join(data_dir, "processed", "ebnerd", "interactions.parquet"))
    first_seen_times = compute_first_seen_times(interactions).to_dict()
    max_age = float((interactions["impression_time"].max() - interactions["impression_time"].min())
                     .total_seconds() / 86400.0)
    reference_time_lookup = dict(first_seen_times)
    reference_time_lookup.update({k: v for k, v in published_time_lookup.items() if pd.notna(v)})

    session_engagement = combined_session_engagement(data_dir, ebnerd_bundle, test_dir).to_dict("index")
    return model, rr_config, ctr_lookup, impressions_lookup, reference_time_lookup, max_age, session_engagement


def combined_session_engagement(data_dir, ebnerd_bundle, test_dir):
    """Extends pipeline.features_ebnerd's train+val session/engagement
    precompute to also cover ebnerd_testset -- needed since a genuine
    prediction pass has to score test impressions too, and a user's
    testing-period session can legitimately follow on from their train/val
    history.

    One real gap, handled honestly rather than faked: ebnerd_testset's own
    behaviors.parquet has NO `article_ids_clicked` column (confirmed
    against the actual file) -- it's the unlabeled set being predicted, so
    whether a PRIOR-in-session test impression was clicked is genuinely
    unknown at this point, not just inconvenient to compute. Test rows get
    `n_clicked=0` unconditionally for that reason (not a guess -- a
    structurally-unavailable-at-serving-time value, in the same spirit as
    Q9's "report metrics with and without features unavailable at serving
    time"). `read_time`/`scroll_percentage` ARE present for test rows
    (they describe general page-view engagement, not click-specific
    outcomes) and are used normally."""
    frames = []
    for split, dirname in ebnerd_loader.SPLIT_DIRS.items():
        beh = ebnerd_loader.load_behaviors(os.path.join(data_dir, ebnerd_bundle, dirname))
        frames.append(pd.DataFrame({
            "impression_id": f"{split}_" + beh["impression_id"].astype(str),
            "user_id": beh["user_id"], "session_id": beh["session_id"],
            "impression_time": beh["impression_time"], "read_time": beh["read_time"],
            "scroll_percentage": beh["scroll_percentage"],
            "n_clicked": beh["article_ids_clicked"].apply(lambda x: len(_to_list(x))),
        }))
    # Reads ONLY the needed columns directly (not pipeline.ebnerd.load_behaviors,
    # which reads every column including postcode/age/gender/etc.) -- at
    # 13.5M rows, materializing those unused columns would be a real,
    # avoidable memory cost, not a cosmetic one.
    test_beh = pd.read_parquet(
        os.path.join(test_dir, "test", "behaviors.parquet"),
        columns=["impression_id", "user_id", "session_id", "impression_time", "read_time",
                 "scroll_percentage", "is_beyond_accuracy"],
    )
    # ebnerd_testset/test/behaviors.parquet bundles the RecSys Challenge's
    # separate "beyond accuracy" evaluation subset (diversity/serendipity
    # track, not implemented by this project -- see README) in the SAME
    # file as the main click-prediction test rows. Confirmed against the
    # actual file: all 200,000 is_beyond_accuracy=True rows share the same
    # placeholder impression_id=0 (not real per-impression IDs), which is
    # what makes `.set_index("impression_id")` raise "index must be
    # unique" below if they aren't dropped first. They're out of this
    # project's scope regardless of the index collision, so dropping them
    # here is correct, not just a workaround.
    n_before = len(test_beh)
    test_beh = test_beh[~test_beh["is_beyond_accuracy"]].drop(columns=["is_beyond_accuracy"])
    if n_before != len(test_beh):
        print(f"  dropped {n_before - len(test_beh):,} beyond-accuracy test rows "
              f"(out of scope, degenerate impression_id -- see comment)")
    test_beh["impression_time"] = pd.to_datetime(test_beh["impression_time"], errors="coerce")
    frames.append(pd.DataFrame({
        "impression_id": "test_" + test_beh["impression_id"].astype(str),
        "user_id": test_beh["user_id"], "session_id": test_beh["session_id"],
        "impression_time": test_beh["impression_time"], "read_time": test_beh["read_time"],
        "scroll_percentage": test_beh["scroll_percentage"],
        "n_clicked": 0,
    }))
    return _session_engagement_from_frame(pd.concat(frames, ignore_index=True))


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


def _reranker_feature_matrix_ebnerd(imp_id, user_id, hist, cand_ids, imp_time, bm25_scores, semantic_scores,
                                      fusion_scores, semantic, recent_n, recency_decay, rr):
    """EB-NeRD counterpart to generate_predictions.py's
    `_reranker_feature_matrix` -- same idea, different feature set
    (retrieval.reranker_data.FEATURE_COLUMNS["ebnerd"]): no subcategory
    (unusable field, see pipeline/features_ebnerd.py), but real
    session/engagement features MIND has no equivalent of at all."""
    pos_map = position_bias_map(cand_ids)
    cat_map = weighted_category_match_map(hist, rr["category_lookup"], recent_n, recency_decay)
    emb_map = history_embedding_similarity_map(hist, semantic, cand_ids, recent_n, recency_decay)
    hist_len = len(hist)

    se = rr["session_engagement"].get(f"test_{imp_id}", {})
    session_prior_click_count = se.get("session_prior_click_count", 0.0)
    user_avg_past_read_time = se.get("user_avg_past_read_time", 0.0)
    user_avg_past_scroll_pct = se.get("user_avg_past_scroll_pct", 0.0)
    if pd.isna(user_avg_past_read_time):
        user_avg_past_read_time = 0.0
    if pd.isna(user_avg_past_scroll_pct):
        user_avg_past_scroll_pct = 0.0

    rows = []
    for i, cid in enumerate(cand_ids):
        cat = rr["category_lookup"].get(cid)
        rows.append([
            pos_map[cid],
            hist_len,
            cat_map.get(cat, 0.0),
            float(emb_map.get(cid, 0.0)),
            rr["ctr_lookup"].get(cid, 0.0),
            rr["impressions_lookup"].get(cid, 0),
            days_since_first_seen_scalar(cid, imp_time, rr["reference_time_lookup"], rr["max_age"]),
            session_prior_click_count, user_avg_past_read_time, user_avg_past_scroll_pct,
            float(bm25_scores[i]), float(semantic_scores[i]), float(fusion_scores[i]),
        ])
    # DataFrame with the trained column names, not a bare ndarray -- see
    # the identical comment in scripts/generate_predictions.py: avoids an
    # sklearn UserWarning firing on every single one of 13.5M impressions.
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS["ebnerd"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--ebnerd_bundle", default="ebnerd_small", help="train+val bundle used for combined history lookup")
    ap.add_argument("--test_dir", default="data/ebnerd_testset")
    ap.add_argument("--model_dir", default="data/models/ebnerd", help="for config.json (hyperparameters)")
    ap.add_argument("--method", choices=["bm25", "semantic", "both", "reranker"], default="both")
    ap.add_argument("--chunk_size", type=int, default=50_000, help="behaviors.parquet rows per streamed batch")
    ap.add_argument("--recent_n", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--recency_decay", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--semantic_backend", choices=["lsa", "sbert"], default=None, help="defaults to config.json")
    ap.add_argument("--lsa_components", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--sbert_model", default=None, help="defaults to config.json")
    ap.add_argument("--bm25_k1", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--bm25_b", type=float, default=None, help="defaults to config.json")
    args = ap.parse_args()
    want_reranker = args.method == "reranker"
    methods_requested = ["bm25", "semantic"] if args.method in ("both", "reranker") else [args.method]
    # --method reranker always fits both scorers (it needs bm25/semantic/
    # fusion scores as re-ranker input features) but writes ONLY the
    # reranker output file -- methods_requested drives index fitting,
    # methods (below) drives which prediction files get written.
    write_bm25_semantic = args.method != "reranker"

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
    combined_ids, combined_texts, token_lookup, test_articles, category_lookup = \
        build_combined_lookups(args.data_dir, args.ebnerd_bundle)
    print(f"  {len(combined_ids):,} unique articles in combined lookup, {len(test_articles):,} in the test corpus")

    rr = None
    if want_reranker:
        # Deliberately loaded BEFORE fit_indices() below -- see the
        # identical comment in scripts/generate_predictions.py: unpickling
        # reranker.pkl (LightGBM) after a FAISS index already exists in
        # this process segfaults reliably on at least one dev machine.
        # Don't reorder this without retesting.
        print("Loading trained re-ranker + its behavioural-feature lookups (this precomputes session/engagement "
              "history over train+val+test, ~14M rows total -- may take a while) ...")
        (rr_model, rr_config, ctr_lookup, impressions_lookup, reference_time_lookup, max_age,
         session_engagement) = load_reranker_assets(args.data_dir, args.model_dir, args.ebnerd_bundle, args.test_dir)
        if rr_config["feature_columns"] != FEATURE_COLUMNS["ebnerd"]:
            raise SystemExit(
                f"data/models/ebnerd/reranker_config.json's feature_columns don't match "
                f"retrieval.reranker_data.FEATURE_COLUMNS['ebnerd'] -- retrain "
                f"(scripts/train_reranker.py --dataset ebnerd) or reconcile the two first."
            )
        rr = {
            "model": rr_model, "category_lookup": category_lookup, "ctr_lookup": ctr_lookup,
            "impressions_lookup": impressions_lookup, "reference_time_lookup": reference_time_lookup,
            "max_age": max_age, "session_engagement": session_engagement,
            "fusion_alpha": rr_config["fusion_alpha"],
        }
        print(f"  loaded reranker.pkl (trained on {len(rr_config['feature_columns'])} features, "
              f"recent_n={rr_config['recent_n']}, recency_decay={rr_config['recency_decay']})")

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
        bm25, semantic = bm25_built, semantic_built

    if semantic is not None:
        t0 = time.time()
        emb = semantic.embed(combined_ids, combined_texts)
        history_embeddings = dict(zip(combined_ids, emb))
        print(f"  projected {len(combined_ids):,} combined articles for history lookup ({time.time()-t0:.1f}s)")

    out_paths = {}
    fhandles = {}
    write_methods = (methods if write_bm25_semantic else []) + (["reranker"] if want_reranker else [])
    for m in write_methods:
        d = os.path.join(args.data_dir, "predictions", "ebnerd", m)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "prediction.txt")
        out_paths[m] = p
        fhandles[m] = open(p, "w")
    semantic_key = semantic_backend

    beh_path = os.path.join(args.test_dir, "test", "behaviors.parquet")
    pf = pq.ParquetFile(beh_path)
    # ebnerd_testset/test/behaviors.parquet bundles the RecSys Challenge's
    # separate "beyond accuracy" evaluation subset (200,000 rows, all
    # sharing a degenerate placeholder impression_id=0 -- confirmed
    # against the actual file) in the SAME file as the main click-
    # prediction test rows. That track isn't implemented by this project
    # (see README), and writing a prediction line for it wouldn't even be
    # well-formed (200,000 rows can't share one impression_id in a
    # 1-line-per-impression submission format) -- every row is dropped by
    # `is_beyond_accuracy`, not just skipped opportunistically.
    n_beyond_accuracy = int(pd.read_parquet(beh_path, columns=["is_beyond_accuracy"])["is_beyond_accuracy"].sum())
    total_rows = pf.metadata.num_rows - n_beyond_accuracy
    read_cols = (["impression_id", "user_id", "article_ids_inview", "is_beyond_accuracy"]
                 + (["impression_time"] if want_reranker else []))
    print(f"\nStreaming {beh_path} ({pf.metadata.num_rows:,} rows, {total_rows:,} in scope after "
          f"dropping {n_beyond_accuracy:,} beyond-accuracy rows) in batches of {args.chunk_size:,} ...")
    n_written = 0
    t_start = time.time()
    for batch in pf.iter_batches(batch_size=args.chunk_size, columns=read_cols):
        chunk = batch.to_pandas()
        chunk = chunk[~chunk["is_beyond_accuracy"]]
        if chunk.empty:
            continue
        imp_ids = chunk["impression_id"].tolist()
        user_ids = chunk["user_id"].tolist()
        cand_lists = [[str(a) for a in _to_list(x)] for x in chunk["article_ids_inview"].tolist()]
        imp_times = pd.to_datetime(chunk["impression_time"]).tolist() if want_reranker else [None] * len(chunk)

        for imp_id, uid, cand_ids, imp_time in zip(imp_ids, user_ids, cand_lists, imp_times):
            hist = history_lookup.get(uid, [])
            recent = hist[-recent_n:] if hist else []

            bm25_scores = None
            if bm25 is not None:
                q_weights = build_query_weights(hist, recent_n, recency_decay, token_lookup)
                bm25_scores = bm25.score_candidates(q_weights, cand_ids)
                if write_bm25_semantic:
                    ranks = ranks_from_scores(bm25_scores)
                    fhandles["bm25"].write(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]\n")

            semantic_scores = None
            if semantic is not None:
                embs = [history_embeddings.get(a) for a in recent]
                w = recency_weights(len(recent), recency_decay) if recent else None
                user_vec = mean_pool_user_vector(embs, weights=w)
                semantic_scores = semantic.score_candidates(user_vec, cand_ids)
                if write_bm25_semantic:
                    ranks = ranks_from_scores(semantic_scores)
                    fhandles[semantic_key].write(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]\n")

            if want_reranker:
                # alpha=rr["fusion_alpha"] (from reranker_config.json), not a
                # hardcoded/CLI value -- this feature must match what the
                # model was actually trained on, see generate_predictions.py's
                # identical comment on this exact point.
                fusion_scores = fuse_scores(bm25_scores, semantic_scores, alpha=rr["fusion_alpha"])
                X = _reranker_feature_matrix_ebnerd(imp_id, uid, hist, cand_ids, imp_time, bm25_scores,
                                                      semantic_scores, fusion_scores, semantic,
                                                      recent_n, recency_decay, rr)
                reranker_scores = rr["model"].predict(X)
                ranks = ranks_from_scores(reranker_scores)
                fhandles["reranker"].write(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]\n")

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
