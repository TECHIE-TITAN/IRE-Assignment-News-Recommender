#!/usr/bin/env python3
"""Fits BM25F + a semantic index (LSA or SBERT+FAISS) once over the
train+val article corpus and persists them
(data/models/mind/{bm25.pkl,semantic.pkl,config.json}), so
scripts/evaluate_retrieval.py (Q2/Q3) and scripts/evaluate_ranking.py (Q4)
load the exact same fitted index instance instead of each independently
re-fitting from raw text with separately-specified hyperparameters that
could silently drift out of sync between scripts.

scripts/generate_predictions.py (Q5) still fits its own index -- it
necessarily scores a *different* corpus (MINDlarge_test's own articles,
not train+val) -- but reads config.json for its hyperparameters, so all
three stay guaranteed-consistent on k1/b/field_weights/lsa_components/
entity_weight/recency/semantic_backend even where the fitted matrix itself
must differ.

    python scripts/build_indices.py                          # LSA (fast, no download)
    python scripts/build_indices.py --semantic_backend sbert  # sentence-transformers + FAISS

Re-run this (with new flags) any time you want to change a hyperparameter
that affects index *fitting* (k1, b, field weights, lsa_components/
sbert_model, entity_weight, semantic_backend) -- evaluate_retrieval.py/
evaluate_ranking.py no longer take those flags themselves, precisely so
they can't drift from this artifact. recent_n/recency_decay
(query-construction, not fitting) can still be overridden per eval run
without rebuilding.

--semantic_backend sbert requires `pip install sentence-transformers
faiss-cpu` (not installed by default -- see requirements.txt) and will
download the model (~80MB for the default all-MiniLM-L6-v2) on first run.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.build_indices import fit_indices, save_indices
from retrieval.entity_embeddings import load_entity_vectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--train_dir", default=None, help="MIND only, for entity_embedding.vec; defaults to data/MINDlarge_train")
    ap.add_argument("--val_dir", default=None, help="MIND only, for entity_embedding.vec; defaults to data/MINDlarge_dev")
    ap.add_argument("--out_dir", default=None, help="defaults to data/models/<dataset>")
    ap.add_argument("--semantic_backend", choices=["lsa", "sbert"], default="lsa")
    ap.add_argument("--lsa_components", type=int, default=128)
    ap.add_argument("--sbert_model", default=None,
                     help="defaults to an English model for mind, a multilingual (Danish-capable) one for ebnerd "
                          "-- see retrieval/sbert.py; EB-NeRD is Danish text, an English-only encoder would be a "
                          "real quality bug here, not just a suboptimal default")
    ap.add_argument("--sbert_batch_size", type=int, default=256)
    ap.add_argument("--sbert_device", default=None, help="e.g. 'cpu', 'mps', 'cuda'; None = auto-detect")
    ap.add_argument("--bm25_k1", type=float, default=3.0, help="tuned via scripts/tune_bm25.py (MIND value; retune per dataset)")
    ap.add_argument("--bm25_b", type=float, default=1.0, help="tuned via scripts/tune_bm25.py (MIND value; retune per dataset)")
    ap.add_argument("--bm25_title_weight", type=float, default=2.0)
    ap.add_argument("--bm25_abstract_weight", type=float, default=1.0)
    ap.add_argument("--bm25_entity_boost", type=float, default=1.0,
                     help="BM25's own entity-overlap boost (raw id/name overlap, works even for ebnerd "
                          "which has no entity embeddings) -- see retrieval/bm25.py")
    ap.add_argument("--entity_weight", type=float, default=1.0)
    ap.add_argument("--no_entity_fusion", action="store_true")
    ap.add_argument("--recent_n", type=int, default=20, help="query-construction only, not index fitting")
    ap.add_argument("--recency_decay", type=float, default=0.85, help="query-construction only, not index fitting")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join("data", "models", args.dataset)

    proc_dir = os.path.join(args.data_dir, "processed", args.dataset)
    articles = pd.read_parquet(os.path.join(proc_dir, "articles.parquet"))
    print(f"Loaded {len(articles):,} articles (train+val corpus)")

    # Resolved unconditionally for MIND (not just inside the entity-fusion
    # branch below) so config.json always records the actual paths used,
    # not the raw --train_dir/--val_dir CLI args -- those default to None,
    # and a prior version of this script stored that None straight into
    # config.json whenever entity fusion happened to be skipped for this
    # run, which then crashed scripts/generate_predictions.py the next time
    # it read train_dir/val_dir back out. Left as None for ebnerd, which
    # has no equivalent entity-embedding directories and no code path that
    # reads these back out of config.json.
    train_dir = (args.train_dir or "data/MINDlarge_train") if args.dataset == "mind" else args.train_dir
    val_dir = (args.val_dir or "data/MINDlarge_dev") if args.dataset == "mind" else args.val_dir

    if args.dataset == "ebnerd" and not args.no_entity_fusion:
        print("  entity fusion skipped: EB-NeRD ships no entity *embeddings* (unlike MIND's TransE "
              "vectors) -- ner_clusters/entity_groups are named-entity surface strings, not IDs with "
              "a matching pretrained vector table. Pass --no_entity_fusion to silence this note.")
    entity_vectors = None
    if not args.no_entity_fusion and args.dataset == "mind":
        entity_vectors = load_entity_vectors(
            os.path.join(train_dir, "entity_embedding.vec"),
            os.path.join(val_dir, "entity_embedding.vec"),
        )

    sbert_model = args.sbert_model
    if sbert_model is None and args.semantic_backend == "sbert" and args.dataset == "ebnerd":
        from retrieval.sbert import DEFAULT_MULTILINGUAL_MODEL
        sbert_model = DEFAULT_MULTILINGUAL_MODEL

    field_weights = {"title": args.bm25_title_weight, "abstract": args.bm25_abstract_weight}
    bm25, semantic, doc_ids = fit_indices(
        articles, semantic_backend=args.semantic_backend, lsa_components=args.lsa_components,
        sbert_model=sbert_model, sbert_batch_size=args.sbert_batch_size, sbert_device=args.sbert_device,
        bm25_k1=args.bm25_k1, bm25_b=args.bm25_b, bm25_field_weights=field_weights,
        bm25_entity_boost=args.bm25_entity_boost,
        entity_vectors=entity_vectors, entity_weight=args.entity_weight,
    )

    config = {
        "dataset": args.dataset,
        "n_docs": len(doc_ids),
        "semantic_backend": args.semantic_backend,
        "lsa_components": args.lsa_components,
        "sbert_model": getattr(semantic, "model_name", None),
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "bm25_field_weights": field_weights,
        "bm25_entity_boost": args.bm25_entity_boost,
        "entity_fusion": entity_vectors is not None,
        "entity_weight": args.entity_weight if entity_vectors is not None else None,
        "recent_n": args.recent_n,
        "recency_decay": args.recency_decay,
        "train_dir": train_dir,
        "val_dir": val_dir,
    }
    save_indices(bm25, semantic, config, out_dir)
    print(f"\nwrote {out_dir}/{{bm25.pkl, semantic.pkl, config.json}}")
    print(config)


if __name__ == "__main__":
    main()
