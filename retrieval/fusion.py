"""Score-level fusion of the lexical (BM25F, +optional entity boost) and
semantic (LSA or SBERT+FAISS) scorers into a single ranking.

Raw BM25 scores (unbounded sums of IDF weights) and cosine similarities
([-1, 1]) aren't on comparable scales, so a naive weighted sum would just
be dominated by whichever happens to have the larger numeric range for a
given impression. `fuse_scores` min-max normalizes each method's scores
*within that impression's own candidate list* first (so it's always a
fair, per-impression comparison, not a corpus-wide one), then takes a
weighted average -- `alpha` is the semantic weight, `1-alpha` the lexical
weight, so `alpha=1.0` degenerates to semantic-only and `alpha=0.0` to
lexical-only.

Whether fusion actually helps (vs. either scorer alone) is an empirical
question, not assumed -- validate via scripts/evaluate_ranking.py (which
computes it as a third method alongside bm25/semantic) before trusting it
for a submission.
"""

import numpy as np


def _minmax_normalize(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.zeros_like(x)


def fuse_scores(lexical_scores, semantic_scores, alpha=0.7):
    """`alpha`: weight on the semantic scorer (1-alpha on lexical). Default
    0.7 reflects Q4's own finding that the semantic scorer outperforms
    BM25F on MIND by a wide margin -- not a universal constant, and
    exactly what scripts/evaluate_ranking.py's fusion column exists to
    let you second-guess per dataset."""
    return alpha * _minmax_normalize(semantic_scores) + (1 - alpha) * _minmax_normalize(lexical_scores)
