"""Shared BM25 + LSA index construction, used by both
scripts/evaluate_retrieval.py (Q2/Q3 full-corpus recall@K) and
scripts/evaluate_ranking.py (Q4 within-impression ranking metrics) so the
two evaluations score against exactly the same indices/corpus.
"""

import time

from retrieval.bm25 import BM25Index
from retrieval.lsa import LSAIndex
from retrieval.text_utils import article_text


def fit_indices(articles_df, lsa_components=128, verbose=True):
    """articles_df: DataFrame with article_id/title/abstract columns (the
    unified schema from pipeline/schema.py). Returns (bm25, lsa, doc_ids)."""
    doc_ids = articles_df["article_id"].tolist()
    texts = [article_text(t, a) for t, a in zip(articles_df["title"], articles_df["abstract"])]

    t0 = time.time()
    bm25 = BM25Index().fit(doc_ids, texts)
    if verbose:
        print(f"BM25 index: {len(doc_ids):,} docs, vocab={bm25.bm25_matrix.shape[1]:,} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    lsa = LSAIndex(n_components=lsa_components).fit(doc_ids, texts)
    if verbose:
        print(f"LSA index: {len(doc_ids):,} docs, {lsa_components}-dim ({time.time()-t0:.1f}s)")

    return bm25, lsa, doc_ids
