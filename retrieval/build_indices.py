"""Shared BM25F + semantic (LSA or SBERT+FAISS) index construction, used by
scripts/build_indices.py (fits once, over the train+val corpus),
scripts/evaluate_retrieval.py (Q2/Q3), scripts/evaluate_ranking.py (Q4),
and scripts/tune_bm25.py.

`save_indices`/`load_indices` are what let Q2/Q3 and Q4 share the literal
same fitted index instance (same train+val corpus) instead of each
independently re-fitting from raw text with separately-specified
hyperparameters that could silently drift out of sync between scripts.
scripts/generate_predictions.py (Q5) still fits its own index -- it
necessarily scores a *different* corpus (MINDlarge_test's own articles, not
train+val) -- but reads the same persisted config.json for its
hyperparameters, so all three stay guaranteed-consistent on k1/b/
field_weights/lsa_components/entity_weight/recency/semantic_backend even
though the fitted matrix itself can't be literally shared across a corpus
change.

`semantic_backend`: "lsa" (TF-IDF+SVD, retrieval/lsa.py) or "sbert"
(pretrained sentence-transformer + FAISS, retrieval/sbert.py) -- both
implement the same score_candidates/get_embedding/embed/embeddings/
id_to_row surface, so callers don't need to know which one they got except
where the Q2/Q3 full-corpus retrieval path branches on `search_topk`
availability (SBERTIndex has it, via FAISS; LSAIndex doesn't, and uses its
brute-force score_batch_full instead -- see scripts/evaluate_retrieval.py).
"""

import json
import os
import pickle
import time

from retrieval.bm25 import BM25Index
from retrieval.entity_embeddings import article_entity_embedding
from retrieval.lsa import LSAIndex
from retrieval.text_utils import article_text


def fit_indices(articles_df, semantic_backend="lsa", lsa_components=128,
                 sbert_model=None, sbert_batch_size=256, sbert_device=None,
                 bm25_k1=1.5, bm25_b=0.75, bm25_field_weights=None,
                 entity_vectors=None, entity_weight=1.0, verbose=True):
    """articles_df: DataFrame with article_id/title/abstract/entities
    columns (the unified schema from pipeline/schema.py). `entity_vectors`
    (optional): {wikidata_id: (100,) array} from
    retrieval.entity_embeddings.load_entity_vectors -- when given, the
    semantic index is fit with content+entity fusion (Q3 improvement);
    BM25 is unaffected by it (entity fusion is a semantic-side technique).
    Returns (bm25, semantic, doc_ids)."""
    doc_ids = articles_df["article_id"].tolist()
    titles = articles_df["title"].tolist()
    abstracts = articles_df["abstract"].tolist()
    texts = [article_text(t, a) for t, a in zip(titles, abstracts)]

    t0 = time.time()
    bm25 = BM25Index(k1=bm25_k1, b=bm25_b, field_weights=bm25_field_weights).fit(doc_ids, titles, abstracts)
    if verbose:
        print(f"BM25F index: {len(doc_ids):,} docs, vocab={bm25.bm25_matrix.shape[1]:,}, "
              f"k1={bm25.k1}, b={bm25.b}, field_weights={bm25.field_weights} ({time.time()-t0:.1f}s)")

    entity_embeddings = None
    if entity_vectors is not None:
        entity_embeddings = {
            aid: article_entity_embedding(ents, entity_vectors)
            for aid, ents in zip(doc_ids, articles_df["entities"])
        }
        n_with_entities = sum(1 for e in entity_embeddings.values() if e.any())
        if verbose:
            print(f"  entity vectors: {len(entity_vectors):,} loaded, "
                  f"{n_with_entities:,}/{len(doc_ids):,} articles have >=1 linked entity in the lookup")

    t0 = time.time()
    if semantic_backend == "sbert":
        from retrieval.sbert import DEFAULT_MODEL, SBERTIndex
        model_name = sbert_model or DEFAULT_MODEL
        semantic = SBERTIndex(model_name=model_name, batch_size=sbert_batch_size, device=sbert_device).fit(
            doc_ids, texts, entity_embeddings=entity_embeddings, entity_weight=entity_weight)
        if verbose:
            fused = f", entity-fused (+{semantic.entity_dim}d, weight={semantic.entity_weight})" if semantic.entity_dim else ""
            print(f"SBERT+FAISS index: {len(doc_ids):,} docs, model={model_name}, "
                  f"{semantic.embeddings.shape[1]}-dim{fused} ({time.time()-t0:.1f}s)")
    elif semantic_backend == "lsa":
        semantic = LSAIndex(n_components=lsa_components).fit(
            doc_ids, texts, entity_embeddings=entity_embeddings, entity_weight=entity_weight)
        if verbose:
            fused = f", entity-fused (+{semantic.entity_dim}d, weight={semantic.entity_weight})" if semantic.entity_dim else ""
            print(f"LSA index: {len(doc_ids):,} docs, {lsa_components}-dim{fused} ({time.time()-t0:.1f}s)")
    else:
        raise ValueError(f"unknown semantic_backend: {semantic_backend!r} (expected 'lsa' or 'sbert')")

    return bm25, semantic, doc_ids


def save_indices(bm25, semantic, config, out_dir):
    """Pickles the fitted BM25Index and semantic index (BM25Index/LSAIndex
    are plain attribute bags of sklearn/scipy/numpy objects, all natively
    picklable; SBERTIndex excludes its loaded model and FAISS index from
    the pickle itself via __getstate__/__setstate__, see retrieval/sbert.py)
    plus a JSON sidecar recording every hyperparameter that shaped them --
    scripts/generate_predictions.py reads that sidecar to keep its
    (necessarily separately-fit, different-corpus) index on identical
    hyperparameters rather than its own drift-prone CLI defaults."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(out_dir, "semantic.pkl"), "wb") as f:
        pickle.dump(semantic, f)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_indices(model_dir):
    """Returns (bm25, semantic, config) as saved by `save_indices`. Local,
    self-generated pickles only -- never load one from an untrusted source,
    pickle can execute arbitrary code on load. If `semantic.pkl` is an
    SBERTIndex, unpickling rebuilds its FAISS index from the stored
    embeddings automatically (see SBERTIndex.__setstate__); its
    transformer model stays unloaded until something calls `.embed(...)`."""
    with open(os.path.join(model_dir, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(model_dir, "semantic.pkl"), "rb") as f:
        semantic = pickle.load(f)
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    return bm25, semantic, config
