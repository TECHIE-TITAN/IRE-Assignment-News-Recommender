"""Okapi BM25 lexical retrieval (Q2) over article titles + abstracts.

Implemented as a single sparse (doc x vocab) BM25-weight matrix rather than
a literal posting-list dict: `score(d, q) = sum_{t in q} bm25_weight(d, t)`.
This is the standard vectorized formulation of BM25 -- it behaves exactly
like an inverted index (only terms shared between d and q contribute) but
lets scoring be expressed as sparse linear algebra, which is what makes
scoring both the Q2.4 recall@K sweep and the Q5 2.37M-impression prediction
pass tractable in pure scikit-learn/scipy.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from retrieval.text_utils import tokenize


class BM25Index:
    def __init__(self, k1=1.5, b=0.75, min_df=2, max_df=0.6):
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.bm25_matrix = None  # CSR, (n_docs x vocab)
        self.doc_ids = None      # list[str], row order
        self.id_to_row = None    # dict[str, int]

    def fit(self, doc_ids, texts):
        """`min_df=2` drops hapax terms (typos, IDs) and `max_df=0.6` drops
        near-stopwords -- both shrink the vocab substantially, which matters
        once this runs over the ~121K-article MINDlarge_test corpus."""
        self.vectorizer = CountVectorizer(
            tokenizer=tokenize, preprocessor=lambda s: s, lowercase=False,
            min_df=self.min_df, max_df=self.max_df, token_pattern=None,
        )
        counts = self.vectorizer.fit_transform(texts)  # CSR (D x V), raw term counts
        n_docs, vocab_size = counts.shape

        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        avgdl = doc_len.mean() if n_docs else 0.0

        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        idf = np.clip(idf, 1e-6, None)  # Robertson-Sparck Jones floor: no negative weights

        coo = counts.tocoo()
        f = coo.data.astype(np.float64)
        len_d = doc_len[coo.row]
        denom = f + self.k1 * (1.0 - self.b + self.b * len_d / (avgdl if avgdl else 1.0))
        weight = idf[coo.col] * (f * (self.k1 + 1.0)) / denom

        self.bm25_matrix = sp.csr_matrix((weight, (coo.row, coo.col)), shape=(n_docs, vocab_size))
        self.doc_ids = list(doc_ids)
        self.id_to_row = {a: i for i, a in enumerate(self.doc_ids)}
        return self

    def score_candidates(self, query_tokens, candidate_ids):
        """Q5 path: score only a given impression's candidate list, never
        the other ~100K non-candidate docs -- this is what makes per-
        impression scoring cheap at 2.37M-impression scale."""
        vocab = self.vectorizer.vocabulary_
        cols = list({vocab[t] for t in query_tokens if t in vocab})
        scores = np.zeros(len(candidate_ids))
        if not cols:
            return scores
        rows = [self.id_to_row.get(c) for c in candidate_ids]
        valid = [(i, r) for i, r in enumerate(rows) if r is not None]
        if not valid:
            return scores
        idxs = [i for i, _ in valid]
        rs = [r for _, r in valid]
        sub = self.bm25_matrix[rs, :][:, cols]
        scores[idxs] = np.asarray(sub.sum(axis=1)).ravel()
        return scores

    def score_batch_full(self, query_token_lists):
        """Q2.3/Q2.4: dense (B x n_docs) score matrix for a batch of queries
        against the *entire* indexed corpus, for top-K recall@K evaluation.
        Caller should keep batches modest (a few hundred to ~2000 queries)
        since this densifies B x n_docs."""
        vocab = self.vectorizer.vocabulary_
        rows, cols = [], []
        for qi, qtok in enumerate(query_token_lists):
            for t in set(qtok):
                if t in vocab:
                    rows.append(qi)
                    cols.append(vocab[t])
        data = np.ones(len(rows))
        qmat = sp.csr_matrix((data, (rows, cols)),
                              shape=(len(query_token_lists), self.bm25_matrix.shape[1]))
        scores = qmat.dot(self.bm25_matrix.T)
        return np.asarray(scores.todense())
