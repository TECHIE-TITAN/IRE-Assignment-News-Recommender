#!/usr/bin/env python3
"""Q5: generate MIND Codabench prediction files from BM25F and the
semantic scorer (LSA or SBERT+FAISS, whichever data/models/mind/config.json
says scripts/build_indices.py was last run with).

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
sbert_model, k1, b, field weights, lsa_components, entity_weight,
recent_n, recency_decay) from that same data/models/mind/config.json by
default, so this test-corpus index stays guaranteed-consistent with the
train+val one rather than drifting on separately-specified CLI defaults.
Run scripts/build_indices.py first.

Streams data/MINDlarge_test/behaviors.tsv in chunks (never materializes all
2.37M rows at once) since the assignment explicitly calls out memory
efficiency for the large test sets.

    python scripts/build_indices.py             # once, or whenever hyperparameters change
    python scripts/generate_predictions.py --method bm25
    python scripts/generate_predictions.py --method semantic
    python scripts/generate_predictions.py --method both   # default

Writes data/predictions/mind/bm25/{prediction.txt,prediction.zip} and
data/predictions/mind/<semantic_backend>/{prediction.txt,prediction.zip}
(e.g. .../sbert/... or .../lsa/..., named after whichever backend
config.json specifies -- not a generic "semantic" folder, so the output
directory always says what actually produced it). Each zip contains only
prediction.txt at its root, per submission guidelines.
"""

import argparse
import json
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
from retrieval.lsa import mean_pool_user_vector
from retrieval.text_utils import article_text, recency_weights, tokenize

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
    """Text/token lookups span train+val+test (a test impression's history
    can reference articles clicked during the train/val period), built once
    and reused across the whole streaming pass. Returns
    (combined_ids, combined_texts, combined_entities, token_lookup, test_articles)."""
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
    return combined["article_id"].tolist(), texts, combined["entities"].tolist(), token_lookup, test_articles


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default="data/models/mind", help="for config.json (hyperparameters)")
    ap.add_argument("--method", choices=["bm25", "semantic", "both"], default="both")
    ap.add_argument("--chunk_size", type=int, default=20_000)
    ap.add_argument("--recent_n", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--recency_decay", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--semantic_backend", choices=["lsa", "sbert"], default=None, help="defaults to config.json")
    ap.add_argument("--lsa_components", type=int, default=None, help="defaults to config.json")
    ap.add_argument("--sbert_model", default=None, help="defaults to config.json")
    ap.add_argument("--bm25_k1", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--bm25_b", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--entity_weight", type=float, default=None, help="defaults to config.json")
    ap.add_argument("--no_entity_fusion", action="store_true")
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
    entity_weight = args.entity_weight if args.entity_weight is not None else (config["entity_weight"] or 1.0)
    use_entity_fusion = (not args.no_entity_fusion) and config["entity_fusion"]
    train_dir = config["train_dir"]
    val_dir = config["val_dir"]
    print(f"Using hyperparameters from {args.model_dir}/config.json: k1={bm25_k1}, b={bm25_b}, "
          f"field_weights={config['bm25_field_weights']}, semantic_backend={semantic_backend}, "
          f"lsa_components={lsa_components}, sbert_model={sbert_model}, "
          f"entity_fusion={use_entity_fusion}, recent_n={recent_n}, recency_decay={recency_decay}")

    # "bm25" always writes to data/predictions/mind/bm25/; the semantic
    # method writes to data/predictions/mind/<semantic_backend>/ (e.g.
    # .../sbert/ or .../lsa/), not a generic "semantic" folder, so the
    # output directory always names what actually produced it.
    methods = []
    for m in methods_requested:
        methods.append("bm25" if m == "bm25" else semantic_backend)

    print("Building combined article/text/entity lookup (train+val+test) ...")
    combined_ids, combined_texts, combined_entities, token_lookup, test_articles = build_combined_lookups(args.data_dir)
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
    bm25 = semantic = None
    history_embeddings = None
    if "bm25" in methods_requested or "semantic" in methods_requested:
        bm25_built, semantic_built, test_doc_ids = fit_indices(
            test_articles, semantic_backend=semantic_backend, lsa_components=lsa_components,
            sbert_model=sbert_model, bm25_k1=bm25_k1, bm25_b=bm25_b,
            bm25_field_weights=config["bm25_field_weights"],
            entity_vectors=entity_vectors, entity_weight=entity_weight,
        )
        bm25 = bm25_built if "bm25" in methods_requested else None
        semantic = semantic_built if "semantic" in methods_requested else None

    if semantic is not None:
        t0 = time.time()
        combined_entity_mats = None
        if entity_vectors is not None:
            combined_entity_mats = [article_entity_embedding(ents, entity_vectors) for ents in combined_entities]
        emb = semantic.embed(combined_ids, combined_texts, entity_embeddings_list=combined_entity_mats)
        history_embeddings = dict(zip(combined_ids, emb))
        print(f"  projected {len(combined_ids):,} combined articles for history lookup ({time.time()-t0:.1f}s)")

    out_paths = {}
    fhandles = {}
    for m in methods:
        d = os.path.join(args.data_dir, "predictions", "mind", m)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "prediction.txt")
        out_paths[m] = p
        fhandles[m] = open(p, "w")
    semantic_key = semantic_backend  # the actual dict/file key used for the semantic method below

    beh_path = os.path.join(args.data_dir, "MINDlarge_test", "behaviors.tsv")
    print(f"\nStreaming {beh_path} in chunks of {args.chunk_size:,} ...")
    n_written = 0
    t_start = time.time()
    reader = pd.read_csv(beh_path, sep="\t", header=None, names=BEH_COLS, quoting=3,
                          na_values=[""], keep_default_na=True, chunksize=args.chunk_size)

    for chunk in reader:
        imp_ids = chunk["impression_id"].tolist()
        histories = [mind_loader._parse_history(h) for h in chunk["history"].tolist()]
        cand_labels = [mind_loader._parse_impressions(i) for i in chunk["impressions"].tolist()]

        for imp_id, hist, (cand_ids, _labels) in zip(imp_ids, histories, cand_labels):
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
        print(f"  {n_written:,} impressions written ({time.time()-t_start:.1f}s elapsed)")

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
