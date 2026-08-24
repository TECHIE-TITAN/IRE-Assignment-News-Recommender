"""Recall@K over a dense (batch x full-corpus) score matrix -- shared by the
Q2 (BM25) and Q3 (LSA) evaluation runs in scripts/evaluate_retrieval.py.
"""

import numpy as np


def recall_at_k_from_scores(score_matrix, doc_id_to_row, truth_article_ids_per_query, ks):
    """Q2.4/Q3.4: "how many ground-truth clicked articles appear in the
    top-K candidates" -- recall@K here is the fraction of impressions (that
    have at least one ground-truth click present in the indexed corpus)
    whose click lands in the top-K retrieved set.

    score_matrix: dense (B x n_docs) scores over the *entire* indexed corpus.
    truth_article_ids_per_query: list[set[str]] of length B, ground-truth
    clicked article ids for that query.
    """
    ks = sorted(ks)
    max_k = max(ks)
    n_docs = score_matrix.shape[1]
    kth = min(max_k, n_docs - 1)
    top_idx = np.argpartition(-score_matrix, kth=kth, axis=1)[:, :max_k + 1]

    hits = {k: [] for k in ks}
    for qi, truths in enumerate(truth_article_ids_per_query):
        if not truths:
            continue
        truth_rows = {doc_id_to_row[a] for a in truths if a in doc_id_to_row}
        if not truth_rows:
            continue
        row_top = top_idx[qi]
        row_scores = score_matrix[qi, row_top]
        order = row_top[np.argsort(-row_scores)]
        for k in ks:
            hit = bool(truth_rows & set(order[:k].tolist()))
            hits[k].append(1.0 if hit else 0.0)

    recall = {k: (float(np.mean(v)) if v else float("nan")) for k, v in hits.items()}
    n_eval = {k: len(v) for k, v in hits.items()}
    return recall, n_eval
