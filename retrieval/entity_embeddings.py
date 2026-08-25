"""MIND's own pretrained TransE knowledge-graph entity embeddings
(`entity_embedding.vec`) -- parsed into the article schema's `entities`
column by pipeline/mind.py but never used downstream until now. This module
loads them and fuses a per-article mean-of-linked-entities vector into the
LSA content embedding (retrieval/lsa.py), giving Q3 a genuinely richer
*unsupervised* semantic representation (content + knowledge graph) instead
of content alone -- still no click labels involved, still Q3-scoped.
"""

import numpy as np
import pandas as pd

ENTITY_DIM = 100


def load_entity_vectors(*paths):
    """Unions entity_embedding.vec across the given file paths (e.g. a
    train dir's and a val dir's own files). MIND's TransE embeddings are
    one shared Wikidata-entity space; each split's file just lists the
    subset of entities that appear in that split's articles, so a de-duped
    union by entity id (keep first) is a safe merge, not a re-fit."""
    dims = [f"d{i}" for i in range(ENTITY_DIM)]
    frames = []
    for p in paths:
        # Each line has a trailing tab -> ENTITY_DIM+2 fields, not ENTITY_DIM+1
        # (confirmed via the raw file: `awk -F'\t' '{print NF}'` reports 102
        # for a 100-dim file). Passing only entity_id+100 names against that
        # extra field makes pandas silently treat entity_id as an index
        # column instead, shifting every value over by one -- this must be
        # given the true column count explicitly.
        df = pd.read_csv(p, sep="\t", header=None, names=["entity_id"] + dims + ["_trailing"])
        frames.append(df[["entity_id"] + dims])
    all_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="entity_id", keep="first")
    return dict(zip(all_df["entity_id"], all_df[dims].to_numpy(dtype=np.float32)))


def article_entity_embedding(entity_ids, entity_vectors):
    """Mean of an article's linked entities' TransE vectors. Returns an
    all-zero (ENTITY_DIM,) vector if the article has no linked entities or
    none are in the lookup -- `fuse_embeddings` degrades this gracefully
    back to pure-content for such articles, no special-casing needed."""
    vecs = [entity_vectors[e] for e in entity_ids if e in entity_vectors]
    if not vecs:
        return np.zeros(ENTITY_DIM, dtype=np.float32)
    return np.mean(vecs, axis=0)


def fuse_embeddings(content_emb, entity_emb, entity_weight=1.0):
    """L2-normalizes each block independently, concatenates
    `[content ; entity_weight * entity]`, L2-normalizes the result.
    `content_emb`/`entity_emb`: (n, d_content) / (n, ENTITY_DIM) arrays.

    An all-zero entity row (no linked entities) is a no-op here: a zero
    block contributes nothing to the concatenated vector's norm, so
    renormalizing lands back exactly on the pure-content direction --
    articles with no entities silently fall back to content-only similarity
    rather than needing a separate code path.
    """
    c_norm = np.linalg.norm(content_emb, axis=-1, keepdims=True)
    c = np.divide(content_emb, c_norm, out=np.zeros_like(content_emb, dtype=np.float64), where=c_norm > 1e-12)
    e_norm = np.linalg.norm(entity_emb, axis=-1, keepdims=True)
    e = np.divide(entity_emb, e_norm, out=np.zeros_like(entity_emb, dtype=np.float64), where=e_norm > 1e-12)
    fused = np.concatenate([c, entity_weight * e], axis=-1)
    f_norm = np.linalg.norm(fused, axis=-1, keepdims=True)
    return np.divide(fused, f_norm, out=np.zeros_like(fused), where=f_norm > 1e-12)
