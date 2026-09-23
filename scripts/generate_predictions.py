#!/usr/bin/env python3
"""Q5: generate MIND Codabench prediction files from BM25F(+entity), the
semantic scorer (LSA or SBERT+FAISS, whichever data/models/mind/config.json
says scripts/build_indices.py was last run with), and/or their fusion.

For each MINDlarge_test impression, ranks *only that impression's own
candidate list* (not full-corpus retrieval -- Q2/Q3's top-K retrieval is a
separate diagnostic, see scripts/evaluate_retrieval.py) using the user's
recent, recency-weighted click history as the query, and writes one line
per the official format (mind_submission_guidelines.txt):

    ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]

This script necessarily fits its *own* index -- it scores a different
corpus (MINDlarge_test's own articles, not the train+val corpus
scripts/evaluate_retrieval.py / scripts/evaluate_ranking.py load from
data/models/mind/) -- but reads every hyperparameter (semantic_backend,
sbert_model, k1, b, field weights, bm25_entity_boost, lsa_components,
entity_weight, recent_n, recency_decay) from that same
data/models/mind/config.json by default, so this test-corpus index stays
guaranteed-consistent with the train+val one rather than drifting on
separately-specified CLI defaults. Run scripts/build_indices.py first.

Streams data/MINDlarge_test/behaviors.tsv in chunks (never materializes all
2.37M rows at once) since the assignment explicitly calls out memory
efficiency for the large test sets. The per-impression scoring loop is
parallelized across `--n_workers` processes (default: all cores but one) --
it's the actual bottleneck at 2.37M+ impressions, and embarrassingly
parallel (each impression is scored independently), so this is where
multiprocessing actually pays for itself. The (already-fit) bm25/semantic
index objects are sent to each worker once at pool startup, not per task.

    python scripts/build_indices.py                # once, or whenever hyperparameters change
    python scripts/train_reranker.py --dataset mind  # once, if you also want --method reranker
    python scripts/generate_predictions.py --method bm25
    python scripts/generate_predictions.py --method semantic
    python scripts/generate_predictions.py --method fusion    # needs both scorers as input, writes only the fused ranking
    python scripts/generate_predictions.py --method reranker  # needs the trained re-ranker (data/models/mind/reranker.pkl)
    python scripts/generate_predictions.py --method all       # default: writes bm25 + semantic + fusion + reranker

`--method reranker` (A2 Q2) replicates Q1's behavioural features
(pipeline/features_mind.py, via pipeline/features_common.py's shared
helpers) per candidate inside the streaming worker -- see
`_reranker_feature_matrix` -- rather than calling build_features.py, since
that script explicitly refuses the unlabeled test split (features built
there assume a `data/processed/mind/interactions.parquet`-shaped input
this streamed, raw-TSV pass never materializes). It always computes
bm25/semantic/fusion scores as a side effect (they're re-ranker input
features), even if only `--method reranker` was requested.

Writes data/predictions/mind/{bm25,<semantic_backend>,fusion,reranker}/
{prediction.txt,prediction.zip} for whichever methods were requested (e.g.
.../sbert/... or .../lsa/... for the semantic one, named after whichever
backend config.json specifies). Each zip contains only prediction.txt at
its root, per submission guidelines.
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import mind as mind_loader
from pipeline.features_common import (
    compute_first_seen_times,
    days_since_first_seen_scalar,
    history_embedding_similarity_map,
    position_bias_map,
    weighted_category_match_map,
)
from retrieval.build_indices import fit_indices
from retrieval.entity_embeddings import article_entity_embedding, load_entity_vectors
from retrieval.fusion import fuse_scores
from retrieval.lsa import mean_pool_user_vector
from retrieval.reranker_data import FEATURE_COLUMNS
from retrieval.text_utils import article_text, recency_weights, tokenize, weighted_query_entities

BEH_COLS = mind_loader.BEH_COLS


def ranks_from_scores(scores):
    """1-indexed ranks, 1 = highest score. Ties broken deterministically by
    original candidate order (stable sort), so every candidate gets a
    distinct rank -- required by the submission format."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def build_combined_lookups(data_dir):
    """Text/entity/token/category lookups span train+val+test (a test
    impression's history can reference articles clicked during the
    train+val period), built once and reused across the whole streaming
    pass. Returns (combined_ids, combined_texts, combined_entities,
    token_lookup, entity_lookup, test_articles, category_lookup,
    subcategory_lookup). The category/subcategory lookups are only needed
    for --method reranker (Q1's history_category_match/
    history_subcategory_match features), but are cheap to build alongside
    the rest regardless of --method."""
    train_val_articles = pd.read_parquet(os.path.join(data_dir, "processed", "mind", "articles.parquet"))
    test_news = mind_loader.load_news_raw(os.path.join(data_dir, "MINDlarge_test"))
    test_articles = mind_loader.articles_from_news_df(test_news)

    cols = ["article_id", "title", "abstract", "entities", "category", "subcategory"]
    combined = pd.concat([train_val_articles[cols], test_articles[cols]], ignore_index=True)
    combined = combined.drop_duplicates(subset="article_id", keep="first").reset_index(drop=True)

    texts = [article_text(t, a) for t, a in zip(combined["title"], combined["abstract"])]
    # Tokenize once per unique article (not once per impression -- the same
    # history article is referenced by many impressions/users) and dedupe
    # within-article, matching weighted-query semantics. Only used for BM25.
    token_lookup = {aid: set(tokenize(text)) for aid, text in zip(combined["article_id"], texts)}
    entity_lookup = dict(zip(combined["article_id"], combined["entities"]))
    category_lookup = dict(zip(combined["article_id"], combined["category"]))
    subcategory_lookup = dict(zip(combined["article_id"], combined["subcategory"]))
    return (combined["article_id"].tolist(), texts, combined["entities"].tolist(), token_lookup, entity_lookup,
            test_articles, category_lookup, subcategory_lookup)


def load_reranker_assets(data_dir, model_dir):
    """Loads the trained re-ranker (scripts/train_reranker.py) plus the two
    lookups its behavioural features need that generate_predictions.py
    otherwise has no reason to load: train-split article popularity
    (data/feature_store/mind/article_features.parquet, same
    train_ctr/train_impressions Q1 already computed) and per-article
    first-seen times. `compute_first_seen_times` is called over the FULL
    (train+val+test) interactions.parquet -- safe to do without per-split
    gating, per that function's own docstring, and MIND's build_pipeline.py
    already folds MINDlarge_test into that same table, so test articles'
    real first-appearance times are already in there without needing a
    separate pass over the raw test TSV."""
    with open(os.path.join(model_dir, "reranker.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "reranker_config.json")) as f:
        rr_config = json.load(f)
    article_features = pd.read_parquet(os.path.join(data_dir, "feature_store", "mind", "article_features.parquet"))
    ctr_lookup = dict(zip(article_features["article_id"], article_features["train_ctr"].fillna(0.0)))
    impressions_lookup = dict(zip(article_features["article_id"], article_features["train_impressions"].fillna(0)))

    interactions = pd.read_parquet(os.path.join(data_dir, "processed", "mind", "interactions.parquet"))
    first_seen_times = compute_first_seen_times(interactions).to_dict()
    max_age = float((interactions["impression_time"].max() - interactions["impression_time"].min())
                     .total_seconds() / 86400.0)
    return model, rr_config, ctr_lookup, impressions_lookup, first_seen_times, max_age


def build_query_weights(hist, recent_n, decay, token_lookup):
    """Same recency-weighted {term: weight} construction as
    text_utils.weighted_query_terms, but built from a precomputed
    per-article token set (`token_lookup`) instead of re-tokenizing text on
    every call -- required for this to stay fast at 2.37M impressions."""
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


# -- worker-process state and functions (module-level: required for pickling
# under macOS/Windows's default "spawn" multiprocessing start method) -------
_WORKER = {}


def _init_worker(reranker, bm25, semantic, token_lookup, entity_lookup, history_embeddings,
                  recent_n, recency_decay, fusion_alpha, want_bm25, want_semantic, want_fusion, semantic_key):
    """Runs once per worker process at pool startup -- the (already-fit)
    index objects are pickled to each worker here, not re-fit per task and
    not re-sent per task; this is the one-time cost that makes the
    per-impression parallelism a net win. `reranker` (optional, None unless
    --method reranker/all): a dict with model/feature_cols/category_lookup/
    subcategory_lookup/ctr_lookup/impressions_lookup/first_seen_times/
    max_age. Listed FIRST in this signature to match `init_args`'s tuple
    order in main() -- see the comment there for why the order itself,
    not just presence, matters."""
    _WORKER.update(dict(
        bm25=bm25, semantic=semantic, token_lookup=token_lookup, entity_lookup=entity_lookup,
        history_embeddings=history_embeddings, recent_n=recent_n, recency_decay=recency_decay,
        fusion_alpha=fusion_alpha, want_bm25=want_bm25, want_semantic=want_semantic,
        want_fusion=want_fusion, semantic_key=semantic_key, reranker=reranker,
    ))


def _reranker_feature_matrix(hist, cand_ids, imp_time, bm25_scores, semantic_scores, fusion_scores,
                               semantic, recent_n, recency_decay, rr):
    """Builds the (n_candidates, n_features) matrix the trained re-ranker
    scores, in the EXACT column order it was trained on
    (retrieval.reranker_data.FEATURE_COLUMNS["mind"]) -- replicates
    pipeline/features_mind.py's per-impression feature construction, since
    that module operates on an already-built processed/mind/interactions.parquet
    table and can't be called directly against a streamed, unlabeled
    behaviors.tsv row. `semantic` here is fit over the TEST corpus (a
    different embedding space than train+val's persisted index the
    re-ranker was trained against) -- exact for SBERT, since it's a fixed
    pretrained encoder whose embedding space doesn't depend on which corpus
    it's applied to; would NOT be exact for LSA (corpus-specific SVD basis)
    if this project's config ever switched semantic_backend away from
    sbert -- not the case for MIND's actual config, but worth knowing if
    that ever changes."""
    pos_map = position_bias_map(cand_ids)
    cat_map = weighted_category_match_map(hist, rr["category_lookup"], recent_n, recency_decay)
    subcat_map = weighted_category_match_map(hist, rr["subcategory_lookup"], recent_n, recency_decay)
    emb_map = history_embedding_similarity_map(hist, semantic, cand_ids, recent_n, recency_decay)
    hist_len = len(hist)

    rows = []
    for i, cid in enumerate(cand_ids):
        cat = rr["category_lookup"].get(cid)
        subcat = rr["subcategory_lookup"].get(cid)
        rows.append([
            pos_map[cid],
            hist_len,
            cat_map.get(cat, 0.0),
            subcat_map.get(subcat, 0.0),
            float(emb_map.get(cid, 0.0)),
            rr["ctr_lookup"].get(cid, 0.0),
            rr["impressions_lookup"].get(cid, 0),
            days_since_first_seen_scalar(cid, imp_time, rr["first_seen_times"], rr["max_age"]),
            float(bm25_scores[i]), float(semantic_scores[i]), float(fusion_scores[i]),
        ])
    # A DataFrame with the trained column names, not a bare ndarray: the
    # model was fit on named columns (train_reranker.py's train_df[feature_cols]),
    # and LightGBM predicts by column POSITION either way -- correctness
    # doesn't depend on this -- but a nameless array triggers a real
    # sklearn UserWarning on every single call, which at 2.37M impressions
    # is 2.37M warnings flooding stderr, not just one cosmetic notice.
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS["mind"])


def _process_batch(batch):
    """batch: list of (impression_id, history, candidate_ids, impression_time).
    Returns {method: [output lines]} for whichever methods were requested."""
    w = _WORKER
    bm25, semantic = w["bm25"], w["semantic"]
    token_lookup, entity_lookup = w["token_lookup"], w["entity_lookup"]
    history_embeddings = w["history_embeddings"]
    recent_n, recency_decay, fusion_alpha = w["recent_n"], w["recency_decay"], w["fusion_alpha"]
    want_bm25, want_semantic, want_fusion = w["want_bm25"], w["want_semantic"], w["want_fusion"]
    semantic_key = w["semantic_key"]
    rr = w["reranker"]
    want_reranker = rr is not None
    need_bm25 = want_bm25 or want_fusion or want_reranker
    need_semantic = want_semantic or want_fusion or want_reranker

    out = {"bm25": [], semantic_key: [], "fusion": [], "reranker": []}
    for imp_id, hist, cand_ids, imp_time in batch:
        recent = hist[-recent_n:] if hist else []

        bm25_scores = None
        if need_bm25:
            q_weights = build_query_weights(hist, recent_n, recency_decay, token_lookup)
            q_entities = weighted_query_entities(hist, entity_lookup, recent_n=recent_n, decay=recency_decay)
            bm25_scores = bm25.score_candidates(q_weights, cand_ids, query_entity_weights=q_entities)
            if want_bm25:
                ranks = ranks_from_scores(bm25_scores)
                out["bm25"].append(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]")

        semantic_scores = None
        if need_semantic:
            embs = [history_embeddings.get(a) for a in recent]
            wts = recency_weights(len(recent), recency_decay) if recent else None
            user_vec = mean_pool_user_vector(embs, weights=wts)
            semantic_scores = semantic.score_candidates(user_vec, cand_ids)
            if want_semantic:
                ranks = ranks_from_scores(semantic_scores)
                out[semantic_key].append(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]")

        if want_fusion:
            fusion_scores = fuse_scores(bm25_scores, semantic_scores, alpha=fusion_alpha)
            ranks = ranks_from_scores(fusion_scores)
            out["fusion"].append(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]")

        if want_reranker:
            # Deliberately NOT reusing the `fusion_scores` above (computed
            # at the CLI's --fusion_alpha, which a user might legitimately
            # set differently for --method all's own fusion output): the
            # re-ranker's fusion_score FEATURE must match the alpha it was
            # actually TRAINED with (rr["fusion_alpha"], from
            # reranker_config.json), or this would silently feed the model
            # a feature computed differently than during training.
            reranker_fusion_scores = fuse_scores(bm25_scores, semantic_scores, alpha=rr["fusion_alpha"])
            X = _reranker_feature_matrix(hist, cand_ids, imp_time, bm25_scores, semantic_scores,
                                           reranker_fusion_scores, semantic, recent_n, recency_decay, rr)
            reranker_scores = rr["model"].predict(X)
            ranks = ranks_from_scores(reranker_scores)
            out["reranker"].append(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default="data/models/mind", help="for config.json (hyperparameters)")
    ap.add_argument("--method", choices=["bm25", "semantic", "fusion", "reranker", "all"], default="all")
    ap.add_argument("--fusion_alpha", type=float, default=0.7,
                     help="fusion's semantic weight (1-alpha on lexical); see retrieval/fusion.py -- "
                          "validate via scripts/evaluate_ranking.py before trusting a value")
    ap.add_argument("--chunk_size", type=int, default=20_000, help="behaviors.tsv rows read per streamed chunk")
    ap.add_argument("--batch_size", type=int, default=500, help="impressions per unit of work handed to a worker process")
    ap.add_argument("--n_workers", type=int, default=None,
                     help="defaults to all CPU cores but one; set 1 to disable multiprocessing")
    ap.add_argument("--recent_n", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--recency_decay", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--semantic_backend", choices=["lsa", "sbert"], default=None, help="defaults to config.json")
    ap.add_argument("--lsa_components", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--sbert_model", default=None, help="defaults to config.json")
    ap.add_argument("--bm25_k1", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--bm25_b", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--bm25_entity_boost", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--entity_weight", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--no_entity_fusion", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    recent_n = args.recent_n if args.recent_n is not None else config["recent_n"]
    recency_decay = args.recency_decay if args.recency_decay is not None else config["recency_decay"]
    semantic_backend = args.semantic_backend if args.semantic_backend is not None else config.get("semantic_backend", "lsa")
    lsa_components = args.lsa_components if args.lsa_components is not None else config["lsa_components"]
    sbert_model = args.sbert_model if args.sbert_model is not None else config.get("sbert_model")
    bm25_k1 = args.bm25_k1 if args.bm25_k1 is not None else config["bm25_k1"]
    bm25_b = args.bm25_b if args.bm25_b is not None else config["bm25_b"]
    bm25_entity_boost = args.bm25_entity_boost if args.bm25_entity_boost is not None else config.get("bm25_entity_boost", 1.0)
    entity_weight = args.entity_weight if args.entity_weight is not None else (config["entity_weight"] or 1.0)
    use_entity_fusion = (not args.no_entity_fusion) and config["entity_fusion"]
    train_dir = config["train_dir"]
    val_dir = config["val_dir"]
    n_workers = args.n_workers if args.n_workers is not None else max(1, (os.cpu_count() or 2) - 1)

    want_bm25 = args.method in ("bm25", "all")
    want_semantic = args.method in ("semantic", "all")
    want_fusion = args.method in ("fusion", "all")
    want_reranker = args.method in ("reranker", "all")

    print(f"Using hyperparameters from {args.model_dir}/config.json: k1={bm25_k1}, b={bm25_b}, "
          f"field_weights={config['bm25_field_weights']}, bm25_entity_boost={bm25_entity_boost}, "
          f"semantic_backend={semantic_backend}, lsa_components={lsa_components}, sbert_model={sbert_model}, "
          f"entity_fusion={use_entity_fusion}, recent_n={recent_n}, recency_decay={recency_decay}, "
          f"fusion_alpha={args.fusion_alpha}, n_workers={n_workers}")

    print("Building combined article/text/entity lookup (train+val+test) ...")
    (combined_ids, combined_texts, combined_entities, token_lookup, entity_lookup, test_articles,
     category_lookup, subcategory_lookup) = build_combined_lookups(args.data_dir)
    print(f"  {len(combined_ids):,} unique articles in combined lookup")

    rr = None
    if want_reranker:
        # Deliberately loaded BEFORE fit_indices() below: unpickling
        # reranker.pkl (LightGBM) AFTER a FAISS index has already been
        # built in this process (fit_indices -> SBERTIndex) segfaults
        # reliably on at least one dev machine -- confirmed by isolating
        # it to exactly this pair/order via a standalone repro. Loading
        # LightGBM first avoids it; don't reorder this without retesting.
        print("Loading trained re-ranker + its behavioural-feature lookups ...")
        rr_model, rr_config, ctr_lookup, impressions_lookup, first_seen_times, max_age = \
            load_reranker_assets(args.data_dir, args.model_dir)
        if rr_config["feature_columns"] != FEATURE_COLUMNS["mind"]:
            raise SystemExit(
                f"data/models/mind/reranker_config.json's feature_columns don't match "
                f"retrieval.reranker_data.FEATURE_COLUMNS['mind'] -- the re-ranker was trained on a "
                f"different feature set than this script builds. Retrain (scripts/train_reranker.py) "
                f"or reconcile the two before generating predictions with it."
            )
        rr = {
            "model": rr_model, "category_lookup": category_lookup, "subcategory_lookup": subcategory_lookup,
            "ctr_lookup": ctr_lookup, "impressions_lookup": impressions_lookup,
            "first_seen_times": first_seen_times, "max_age": max_age,
            "fusion_alpha": rr_config["fusion_alpha"],
        }
        print(f"  loaded reranker.pkl (trained on {len(rr_config['feature_columns'])} features, "
              f"recent_n={rr_config['recent_n']}, recency_decay={rr_config['recency_decay']})")

    entity_vectors = None
    if use_entity_fusion:
        entity_vectors = load_entity_vectors(
            os.path.join(train_dir, "entity_embedding.vec"),
            os.path.join(val_dir, "entity_embedding.vec"),
            os.path.join(args.data_dir, "MINDlarge_test", "entity_embedding.vec"),
        )
        print(f"  entity vectors: {len(entity_vectors):,} loaded")

    # This is a genuinely different corpus from data/models/mind/{bm25,semantic}.pkl
    # (MINDlarge_test's own articles, not train+val) -- can't load those
    # pickles here, but fit_indices() with these config-sourced hyperparameters
    # keeps this index on identical settings.
    bm25, semantic, test_doc_ids = fit_indices(
        test_articles, semantic_backend=semantic_backend, lsa_components=lsa_components,
        sbert_model=sbert_model, bm25_k1=bm25_k1, bm25_b=bm25_b,
        bm25_field_weights=config["bm25_field_weights"], bm25_entity_boost=bm25_entity_boost,
        entity_vectors=entity_vectors, entity_weight=entity_weight,
    )

    history_embeddings = {}
    t0 = time.time()
    combined_entity_mats = None
    if entity_vectors is not None:
        combined_entity_mats = [article_entity_embedding(ents, entity_vectors) for ents in combined_entities]
    emb = semantic.embed(combined_ids, combined_texts, entity_embeddings_list=combined_entity_mats)
    history_embeddings = dict(zip(combined_ids, emb))
    print(f"  projected {len(combined_ids):,} combined articles for history lookup ({time.time()-t0:.1f}s)")

    semantic_key = semantic_backend
    methods_to_write = [m for m, want in [("bm25", want_bm25), (semantic_key, want_semantic),
                                           ("fusion", want_fusion), ("reranker", want_reranker)] if want]
    out_paths = {}
    fhandles = {}
    for m in methods_to_write:
        d = os.path.join(args.data_dir, "predictions", "mind", m)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "prediction.txt")
        out_paths[m] = p
        fhandles[m] = open(p, "w")

    beh_path = os.path.join(args.data_dir, "MINDlarge_test", "behaviors.tsv")
    print(f"\nStreaming {beh_path} in chunks of {args.chunk_size:,} "
          f"(scored in batches of {args.batch_size:,} across {n_workers} worker process(es)) ...")
    n_written = 0
    t_start = time.time()
    reader = pd.read_csv(beh_path, sep="\t", header=None, names=BEH_COLS, quoting=3,
                          na_values=[""], keep_default_na=True, chunksize=args.chunk_size)

    # `rr` deliberately FIRST in this tuple, not last: under macOS's default
    # "spawn" multiprocessing start method, each worker process receives
    # this whole tuple by unpickling it as ONE pickle stream, and a
    # standard tuple's elements unpickle in stream (= tuple) order -- so
    # putting bm25/semantic (which rebuilds a FAISS index via
    # SBERTIndex.__setstate__) before `rr` (whose LightGBM model has to be
    # unpickled too) would reproduce, inside every single worker, the
    # exact FAISS-before-LightGBM segfault already fixed once for the main
    # process's own load order in load_reranker_assets's caller above.
    # This is precisely what was happening: a worker segfaulting silently
    # makes pool.imap() hang forever waiting for a result that will never
    # arrive, which is indistinguishable from "just slow" until you know
    # to look for it.
    init_args = (rr, bm25, semantic, token_lookup, entity_lookup, history_embeddings,
                 recent_n, recency_decay, args.fusion_alpha, want_bm25, want_semantic, want_fusion, semantic_key)

    pool = mp.Pool(n_workers, initializer=_init_worker, initargs=init_args) if n_workers > 1 else None
    try:
        for chunk in reader:
            imp_ids = chunk["impression_id"].tolist()
            # Needed only for --method reranker (article_days_since_first_seen);
            # parsed for every run regardless since it's a cheap column parse,
            # not worth branching the chunk-reading path on --method.
            imp_times = pd.to_datetime(chunk["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce").tolist()
            histories = [mind_loader._parse_history(h) for h in chunk["history"].tolist()]
            cand_labels = [mind_loader._parse_impressions(i) for i in chunk["impressions"].tolist()]
            impressions = [(imp_id, hist, cand_ids, imp_time) for imp_id, hist, (cand_ids, _labels), imp_time
                            in zip(imp_ids, histories, cand_labels, imp_times)]
            sub_batches = [impressions[i:i + args.batch_size] for i in range(0, len(impressions), args.batch_size)]

            if pool is not None:
                results = pool.imap(_process_batch, sub_batches)
            else:
                _init_worker(*init_args)
                results = (_process_batch(b) for b in sub_batches)

            for result in results:
                for method, lines in result.items():
                    if lines:
                        fhandles[method].write("\n".join(lines) + "\n")

            n_written += len(chunk)
            print(f"  {n_written:,} impressions written ({time.time()-t_start:.1f}s elapsed)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    for f in fhandles.values():
        f.close()

    print(f"\nDone. {n_written:,} predictions written.")
    for m, p in out_paths.items():
        zip_path = os.path.join(os.path.dirname(p), "prediction.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(p, arcname="prediction.txt")
        size_mb = os.path.getsize(zip_path) / 1e6
        print(f"  {m}: {p}  ->  {zip_path} ({size_mb:.1f} MB)")

    print("\nUpload the zip(s) to https://www.codabench.org/competitions/13967/")


if __name__ == "__main__":
    main()
