"""BM25F: field-weighted Okapi BM25 (Q2) over article titles + abstracts.

Plain BM25 flattens title+abstract into one text field, scoring a title-word
match and an abstract-word match identically. Titles are a much
higher-precision relevance signal in short news headlines than abstracts,
so this implements BM25F (Zaragoza et al., "Simple BM25 Extension to
Multiple Weighted Fields", 2004): each field is length-normalized
*separately* (against its own average length) and combined into one
pseudo term-frequency via a per-field weight, before the usual BM25
saturation + IDF is applied once to the combined value:

    pseudo_tf(d,t) = Σ_field  w_field · tf(t,d,field) / (1-b + b·|d|_field/avgdl_field)
    weight(d,t)    = idf(t) · pseudo_tf(d,t)·(k1+1) / (pseudo_tf(d,t) + k1)

As with plain BM25, this is implemented as a single sparse (doc x vocab)
weight matrix rather than a literal posting-list dict -- scoring is
`score(d,q) = Σ_{t∈q} weight(t)·bm25_matrix[d,t]`, a sparse dot product,
which is what keeps both the Q2.4 recall@K sweep and the Q5
2.37M-impression prediction pass fast in pure scikit-learn/scipy.

Queries are now *weighted* term dicts (`{term: weight}`), not plain token
lists -- needed so recency-weighted query construction
(`text_utils.weighted_query_terms`) actually has an effect; a term that
matters more (recent click, or repeated across several recent clicks)
contributes proportionally more to the score, not just a binary presence
bit.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from retrieval.text_utils import identity_preprocessor, tokenize


class BM25Index:
    def __init__(self, k1=1.5, b=0.75, min_df=2, max_df=0.6, field_weights=None):
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.max_df = max_df
        self.field_weights = field_weights or {"title": 2.0, "abstract": 1.0}
        self.vectorizer = None
        self.bm25_matrix = None  # CSR, (n_docs x vocab)
        self.doc_ids = None      # list[str], row order
        self.id_to_row = None    # dict[str, int]

    @staticmethod
    def _length_normalize(counts, lens, avg_len, b):
        """Divides each nonzero (d,t) count by that *field's* own
        length-normalization denominator -- the core of BM25F: title and
        abstract are normalized against their own average length, not a
        single corpus-wide average over the flattened text."""
        coo = counts.tocoo()
        denom = 1.0 - b + b * lens[coo.row] / (avg_len if avg_len else 1.0)
        norm_data = coo.data / denom
        return sp.csr_matrix((norm_data, (coo.row, coo.col)), shape=counts.shape)

    def fit(self, doc_ids, titles, abstracts):
        """`min_df=2` drops hapax terms (typos, IDs) and `max_df=0.6` drops
        near-stopwords -- both shrink the vocab substantially, which matters
        once this runs over the ~121K-article MINDlarge_test corpus."""
        # `t or ""` would miss NaN (bool(nan) is True in Python, so a NaN
        # would pass through unchanged) -- missing abstracts arrive as
        # `None` when the corpus was round-tripped through parquet (e.g.
        # scripts/build_indices.py's train+val corpus) but as a float NaN
        # when read fresh from CSV (e.g. generate_predictions.py's
        # MINDlarge_test corpus), so both must be handled explicitly.
        titles = [t if isinstance(t, str) else "" for t in titles]
        abstracts = [a if isinstance(a, str) else "" for a in abstracts]

        self.vectorizer = CountVectorizer(
            tokenizer=tokenize, preprocessor=identity_preprocessor, lowercase=False,
            min_df=self.min_df, max_df=self.max_df, token_pattern=None,
        )
        # Fit vocab on the combined text (as with plain BM25), but transform
        # each field separately with that *same* fitted vocab, so column
        # indices line up between the two field-count matrices.
        self.vectorizer.fit([t + " " + a for t, a in zip(titles, abstracts)])
        title_counts = self.vectorizer.transform(titles)
        abstract_counts = self.vectorizer.transform(abstracts)
        n_docs, vocab_size = title_counts.shape

        title_len = np.asarray(title_counts.sum(axis=1)).ravel()
        abstract_len = np.asarray(abstract_counts.sum(axis=1)).ravel()
        avg_title_len = title_len.mean() if n_docs else 0.0
        avg_abstract_len = abstract_len.mean() if n_docs else 0.0

        norm_title = self._length_normalize(title_counts, title_len, avg_title_len, self.b)
        norm_abstract = self._length_normalize(abstract_counts, abstract_len, avg_abstract_len, self.b)
        pseudo_tf = (self.field_weights.get("title", 1.0) * norm_title
                     + self.field_weights.get("abstract", 1.0) * norm_abstract).tocsr()

        df = np.asarray((pseudo_tf > 0).sum(axis=0)).ravel()
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        idf = np.clip(idf, 1e-6, None)  # Robertson-Sparck Jones floor: no negative weights

        coo = pseudo_tf.tocoo()
        f = coo.data
        weight = idf[coo.col] * (f * (self.k1 + 1.0)) / (f + self.k1)

        self.bm25_matrix = sp.csr_matrix((weight, (coo.row, coo.col)), shape=(n_docs, vocab_size))
        self.doc_ids = list(doc_ids)
        self.id_to_row = {a: i for i, a in enumerate(self.doc_ids)}
        return self

    def score_candidates(self, query_weights, candidate_ids):
        """Q5 path: score only a given impression's candidate list, never
        the other ~100K non-candidate docs. `query_weights`: {term: weight}
        dict from `text_utils.weighted_query_terms`."""
        vocab = self.vectorizer.vocabulary_
        cols, wts = [], []
        for t, w in query_weights.items():
            col = vocab.get(t)
            if col is not None:
                cols.append(col)
                wts.append(w)
        scores = np.zeros(len(candidate_ids))
        if not cols:
            return scores
        rows = [self.id_to_row.get(c) for c in candidate_ids]
        valid = [(i, r) for i, r in enumerate(rows) if r is not None]
        if not valid:
            return scores
        idxs = [i for i, _ in valid]
        rs = [r for _, r in valid]
        wvec = np.asarray(wts)
        sub = self.bm25_matrix[rs, :][:, cols]
        scores[idxs] = np.asarray(sub.dot(wvec)).ravel()
        return scores

    def score_batch_full(self, query_weight_dicts):
        """Q2.3/Q2.4: dense (B x n_docs) score matrix for a batch of
        weighted queries against the *entire* indexed corpus, for top-K
        recall@K evaluation. Caller should keep batches modest (a few
        hundred to ~2000 queries) since this densifies B x n_docs."""
        vocab = self.vectorizer.vocabulary_
        rows, cols, data = [], [], []
        for qi, qweights in enumerate(query_weight_dicts):
            for t, w in qweights.items():
                col = vocab.get(t)
                if col is not None:
                    rows.append(qi)
                    cols.append(col)
                    data.append(w)
        qmat = sp.csr_matrix((data, (rows, cols)),
                              shape=(len(query_weight_dicts), self.bm25_matrix.shape[1]))
        scores = qmat.dot(self.bm25_matrix.T)
        return np.asarray(scores.todense())
