"""Recall@K computation for scripts/evaluate_retrieval.py (Q2/Q3), two
variants for two different top-K sources:

  - `recall_hits_matrix`: from a dense (batch x full-corpus) score matrix
    (BM25's `score_batch_full`, and LSAIndex's brute-force semantic path).
  - `recall_hits_from_topk`: from an ANN index's native top-K search output
    (SBERTIndex.search_topk via FAISS) -- no dense matrix ever
    materialized for this path.

Both return the same (B, len(ks)) per-impression-hit shape so
scripts/evaluate_retrieval.py can slice recall@K by an impression-level
attribute (cold-start vs warm, Q3.5) after the fact via np.nanmean on a
boolean mask, regardless of which backend produced the hits.
"""

import numpy as np


def recall_hits_matrix(score_matrix, doc_id_to_row, truth_article_ids_per_query, ks):
    """Q2.4/Q3.4: "how many ground-truth clicked articles appear in the
    top-K candidates". Returns a (B, len(ks)) array of per-impression hit
    indicators (1.0 = the impression's clicked article landed in the top-K
    retrieved set, 0.0 = it didn't), NaN where the impression isn't
    evaluable (no ground-truth click, or the clicked article isn't in the
    indexed corpus at all).

    Returning per-impression hits rather than a pre-averaged scalar is what
    lets scripts/evaluate_retrieval.py slice recall@K by an impression-level
    attribute (cold-start vs warm, Q3.5) after the fact via np.nanmean on a
    boolean mask over the concatenated batches, instead of only getting one
    aggregate number.

    score_matrix: dense (B x n_docs) scores over the *entire* indexed corpus.
    truth_article_ids_per_query: list[set[str]] of length B, ground-truth
    clicked article ids for that query.
    """
    ks = sorted(ks)
    max_k = max(ks)
    n_docs = score_matrix.shape[1]
    kth = min(max_k, n_docs - 1)
    top_idx = np.argpartition(-score_matrix, kth=kth, axis=1)[:, :max_k + 1]

    B = score_matrix.shape[0]
    out = np.full((B, len(ks)), np.nan)
    for qi, truths in enumerate(truth_article_ids_per_query):
        if not truths:
            continue
        truth_rows = {doc_id_to_row[a] for a in truths if a in doc_id_to_row}
        if not truth_rows:
            continue
        row_top = top_idx[qi]
        row_scores = score_matrix[qi, row_top]
        order = row_top[np.argsort(-row_scores)]
        for ki, k in enumerate(ks):
            hit = bool(truth_rows & set(order[:k].tolist()))
            out[qi, ki] = 1.0 if hit else 0.0
    return out


def recall_hits_from_topk(topk_row_indices, doc_ids, truth_article_ids_per_query, ks):
    """Same per-impression hit semantics as `recall_hits_matrix`, but for
    ANN search results already restricted to the top-max(ks) rows (e.g.
    SBERTIndex.search_topk's FAISS output) instead of a dense full-corpus
    score matrix.

    topk_row_indices: (B, max_k) row indices into `doc_ids`, already ranked
    descending by score (FAISS's native IndexFlatIP output order); a -1
    entry (fewer than max_k docs in the corpus) is treated as no match.
    doc_ids: list[str], row order of the fitted index (semantic.doc_ids).
    """
    ks = sorted(ks)
    B = topk_row_indices.shape[0]
    out = np.full((B, len(ks)), np.nan)
    for qi, truths in enumerate(truth_article_ids_per_query):
        if not truths:
            continue
        row_order = topk_row_indices[qi]
        for ki, k in enumerate(ks):
            topk_ids_k = {doc_ids[r] for r in row_order[:k] if r >= 0}
            out[qi, ki] = 1.0 if bool(truths & topk_ids_k) else 0.0
    return out
