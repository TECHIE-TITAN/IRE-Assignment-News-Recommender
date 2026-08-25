"""Recall@K over a dense (batch x full-corpus) score matrix -- shared by the
Q2 (BM25) and Q3 (LSA) evaluation runs in scripts/evaluate_retrieval.py.
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
