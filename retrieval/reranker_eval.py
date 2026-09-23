"""Shared per-impression evaluation helpers for the A2 re-ranker, used by
both scripts/train_reranker.py (a quick before/after check right after
training) and scripts/evaluate_reranker.py (the full extended evaluation:
diversity/novelty/coverage, cold/warm + head/tail slicing, bootstrap CIs).
"""

import numpy as np

from retrieval.ranking_metrics import auc_score, mrr_score, ndcg_at_k


def per_impression_metrics(df, score_col, label_col="label", impression_col="impression_id"):
    """Returns arrays (one value per impression, aligned across all four)
    of AUC/MRR/nDCG@5/nDCG@10 -- NaN for an impression with no positive
    label, matching auc_score/mrr_score/ndcg_at_k's own convention (AUC/MRR/
    nDCG are genuinely undefined there, not zero). Callers that just want a
    single overall number should `np.nanmean` these; callers that want a
    *paired* comparison across two scoring methods (e.g. before vs. after
    re-ranking) should keep the per-impression arrays instead of
    immediately averaging, since a paired bootstrap needs to resample the
    same impression indices on both sides at once (see
    retrieval.bootstrap.bootstrap_ci_paired_delta)."""
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    for _, g in df.groupby(impression_col, sort=False, observed=True):
        labels = g[label_col].to_numpy()
        scores = g[score_col].to_numpy()
        aucs.append(auc_score(labels, scores))
        mrrs.append(mrr_score(labels, scores))
        ndcg5s.append(ndcg_at_k(labels, scores, 5))
        ndcg10s.append(ndcg_at_k(labels, scores, 10))
    return {
        "auc": np.array(aucs), "mrr": np.array(mrrs),
        "ndcg5": np.array(ndcg5s), "ndcg10": np.array(ndcg10s),
    }


def evaluate_scores(df, score_col, label_col="label", impression_col="impression_id"):
    """Convenience wrapper: per_impression_metrics, immediately nan-averaged
    to a single number per metric -- what scripts/train_reranker.py's quick
    post-training check uses. scripts/evaluate_reranker.py uses
    per_impression_metrics directly instead, since it needs the raw
    per-impression arrays for slicing and paired bootstrap CIs."""
    per_imp = per_impression_metrics(df, score_col, label_col, impression_col)
    out = {k: float(np.nanmean(v)) for k, v in per_imp.items()}
    out["n_impressions_scored"] = int(len(per_imp["auc"]))
    return out
