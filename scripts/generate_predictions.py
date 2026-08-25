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
    python scripts/generate_predictions.py --method bm25
    python scripts/generate_predictions.py --method semantic
    python scripts/generate_predictions.py --method fusion   # needs both scorers as input, writes only the fused ranking
    python scripts/generate_predictions.py --method all      # default: writes bm25 + semantic + fusion

Writes data/predictions/mind/{bm25,<semantic_backend>,fusion}/
{prediction.txt,prediction.zip} for whichever methods were requested (e.g.
.../sbert/... or .../lsa/... for the semantic one, named after whichever
backend config.json specifies). Each zip contains only prediction.txt at
its root, per submission guidelines.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import mind as mind_loader
from retrieval.build_indices import fit_indices
from retrieval.entity_embeddings import article_entity_embedding, load_entity_vectors
from retrieval.fusion import fuse_scores
from retrieval.lsa import mean_pool_user_vector
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
    """Text/entity/token lookups span train+val+test (a test impression's
    history can reference articles clicked during the train/val period),
    built once and reused across the whole streaming pass. Returns
    (combined_ids, combined_texts, combined_entities, token_lookup,
    entity_lookup, test_articles)."""
    train_val_articles = pd.read_parquet(os.path.join(data_dir, "processed", "mind", "articles.parquet"))
    test_news = mind_loader.load_news_raw(os.path.join(data_dir, "MINDlarge_test"))
    test_articles = mind_loader.articles_from_news_df(test_news)

    combined = pd.concat([train_val_articles[["article_id", "title", "abstract", "entities"]],
                           test_articles[["article_id", "title", "abstract", "entities"]]], ignore_index=True)
    combined = combined.drop_duplicates(subset="article_id", keep="first").reset_index(drop=True)

    texts = [article_text(t, a) for t, a in zip(combined["title"], combined["abstract"])]
    # Tokenize once per unique article (not once per impression -- the same
    # history article is referenced by many impressions/users) and dedupe
    # within-article, matching weighted-query semantics. Only used for BM25.
    token_lookup = {aid: set(tokenize(text)) for aid, text in zip(combined["article_id"], texts)}
    entity_lookup = dict(zip(combined["article_id"], combined["entities"]))
    return combined["article_id"].tolist(), texts, combined["entities"].tolist(), token_lookup, entity_lookup, test_articles


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


def _init_worker(bm25, semantic, token_lookup, entity_lookup, history_embeddings,
                  recent_n, recency_decay, fusion_alpha, want_bm25, want_semantic, want_fusion, semantic_key):
    """Runs once per worker process at pool startup -- the (already-fit)
    index objects are pickled to each worker here, not re-fit per task and
    not re-sent per task; this is the one-time cost that makes the
    per-impression parallelism a net win."""
    _WORKER.update(dict(
        bm25=bm25, semantic=semantic, token_lookup=token_lookup, entity_lookup=entity_lookup,
        history_embeddings=history_embeddings, recent_n=recent_n, recency_decay=recency_decay,
        fusion_alpha=fusion_alpha, want_bm25=want_bm25, want_semantic=want_semantic,
        want_fusion=want_fusion, semantic_key=semantic_key,
    ))


def _process_batch(batch):
    """batch: list of (impression_id, history, candidate_ids). Returns
    {method: [output lines]} for whichever methods were requested."""
    w = _WORKER
    bm25, semantic = w["bm25"], w["semantic"]
    token_lookup, entity_lookup = w["token_lookup"], w["entity_lookup"]
    history_embeddings = w["history_embeddings"]
    recent_n, recency_decay, fusion_alpha = w["recent_n"], w["recency_decay"], w["fusion_alpha"]
    want_bm25, want_semantic, want_fusion = w["want_bm25"], w["want_semantic"], w["want_fusion"]
    semantic_key = w["semantic_key"]
    need_bm25 = want_bm25 or want_fusion
    need_semantic = want_semantic or want_fusion

    out = {"bm25": [], semantic_key: [], "fusion": []}
    for imp_id, hist, cand_ids in batch:
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

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default="data/models/mind", help="for config.json (hyperparameters)")
    ap.add_argument("--method", choices=["bm25", "semantic", "fusion", "all"], default="all")
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

    print(f"Using hyperparameters from {args.model_dir}/config.json: k1={bm25_k1}, b={bm25_b}, "
          f"field_weights={config['bm25_field_weights']}, bm25_entity_boost={bm25_entity_boost}, "
          f"semantic_backend={semantic_backend}, lsa_components={lsa_components}, sbert_model={sbert_model}, "
          f"entity_fusion={use_entity_fusion}, recent_n={recent_n}, recency_decay={recency_decay}, "
          f"fusion_alpha={args.fusion_alpha}, n_workers={n_workers}")

    print("Building combined article/text/entity lookup (train+val+test) ...")
    combined_ids, combined_texts, combined_entities, token_lookup, entity_lookup, test_articles = \
        build_combined_lookups(args.data_dir)
    print(f"  {len(combined_ids):,} unique articles in combined lookup")

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
    methods_to_write = [m for m, want in [("bm25", want_bm25), (semantic_key, want_semantic), ("fusion", want_fusion)] if want]
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

    init_args = (bm25, semantic, token_lookup, entity_lookup, history_embeddings,
                 recent_n, recency_decay, args.fusion_alpha, want_bm25, want_semantic, want_fusion, semantic_key)

    pool = mp.Pool(n_workers, initializer=_init_worker, initargs=init_args) if n_workers > 1 else None
    try:
        for chunk in reader:
            imp_ids = chunk["impression_id"].tolist()
            histories = [mind_loader._parse_history(h) for h in chunk["history"].tolist()]
            cand_labels = [mind_loader._parse_impressions(i) for i in chunk["impressions"].tolist()]
            impressions = [(imp_id, hist, cand_ids) for imp_id, hist, (cand_ids, _labels)
                            in zip(imp_ids, histories, cand_labels)]
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
