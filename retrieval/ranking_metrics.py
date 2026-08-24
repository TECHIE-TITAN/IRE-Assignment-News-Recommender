"""Q4.1: per-impression ranking metrics (AUC, MRR, nDCG@K), computed group-
wise -- one score per impression, then averaged across impressions -- which
is the same convention MIND's own official evaluate.py uses. Operates on a
single impression's (scores, labels) at a time; scripts/evaluate_ranking.py
calls these once per impression per method.
"""

import numpy as np
from scipy.stats import rankdata


def auc_score(labels, scores):
    """Mann-Whitney U / rank-sum formulation of AUC (equivalent to
    sklearn.metrics.roc_auc_score, tie-aware via average ranks) -- avoids
    sklearn's per-call dispatch overhead across ~73K small impressions.
    NaN when an impression has no negatives or no positives (AUC undefined)."""
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    sum_pos_ranks = ranks[labels == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mrr_score(labels, scores):
    """Reciprocal rank of the first clicked candidate in score-descending
    order. NaN when the impression has no clicked candidate."""
    labels = np.asarray(labels)
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranked_labels = labels[order]
    hits = np.where(ranked_labels == 1)[0]
    if len(hits) == 0:
        return float("nan")
    return float(1.0 / (hits[0] + 1))


def _dcg_at_k(ranked_labels, k):
    ranked_labels = np.asarray(ranked_labels[:k], dtype=float)
    if len(ranked_labels) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(ranked_labels) + 2))
    return float((ranked_labels * discounts).sum())


def ndcg_at_k(labels, scores, k):
    """Binary-relevance nDCG@k. NaN when the impression has no clicked
    candidate (ideal DCG would be 0, ratio undefined)."""
    labels = np.asarray(labels)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="stable")
    dcg = _dcg_at_k(labels[order], k)
    ideal = _dcg_at_k(np.sort(labels)[::-1], k)
    return float(dcg / ideal) if ideal > 0 else float("nan")
