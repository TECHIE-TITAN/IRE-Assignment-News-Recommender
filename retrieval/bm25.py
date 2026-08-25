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

Optional entity-overlap boost: BM25F is otherwise purely lexical -- it
never uses the entities the pipeline already parses (only the semantic
side's entity fusion did, retrieval/entity_embeddings.py). If `fit()` is
given `entities` (one list of entity ids/names per doc), `score_candidates`
additionally accepts `query_entity_weights` (same recency-weighted-dict
shape as query terms, built the same way from a user's recent history) and
adds `entity_boost * sum(query_entity_weights[e] for e in doc's entities)`
on top of the lexical score -- a candidate sharing an entity with a
recently-clicked article gets a boost independent of whether it shares any
*words* with it.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from retrieval.text_utils import identity_preprocessor, tokenize


class BM25Index:
    def __init__(self, k1=1.5, b=0.75, min_df=2, max_df=0.6, field_weights=None, entity_boost=1.0):
        self.k1 = k1
        self.b = b
        self.min_df = min_df
        self.max_df = max_df
        self.field_weights = field_weights or {"title": 2.0, "abstract": 1.0}
        self.entity_boost = entity_boost
        self.vectorizer = None
        self.bm25_matrix = None  # CSR, (n_docs x vocab)
        self.doc_ids = None      # list[str], row order
        self.id_to_row = None    # dict[str, int]
        self.entity_to_rows = None  # dict[entity_id, set[int]] inverted index, or None if no entities given

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

    def fit(self, doc_ids, titles, abstracts, entities=None):
        """`min_df=2` drops hapax terms (typos, IDs) and `max_df=0.6` drops
        near-stopwords -- both shrink the vocab substantially, which matters
        once this runs over the ~121K-article MINDlarge_test corpus.
        `entities` (optional): list of entity-id/name lists, one per doc,
        same order as `doc_ids` -- enables the entity-overlap boost in
        `score_candidates`."""
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
        # `if e` would raise on a numpy array (parquet round-trip returns
        # list-typed columns as numpy arrays, not Python lists, and `bool()`
        # on a multi-element array is ambiguous) -- use `len(e)` instead,
        # same fix applied repeatedly elsewhere in this codebase for the
        # same underlying cause (build_query_text, evaluate_ranking.py,
        # evaluate_retrieval.py).
        self.doc_entities = ([set(e) if len(e) else set() for e in entities]
                              if entities is not None else None)
        return self

    def score_candidates(self, query_weights, candidate_ids, query_entity_weights=None):
        """Q5 path: score only a given impression's candidate list, never
        the other ~100K non-candidate docs. `query_weights`: {term: weight}
        dict from `text_utils.weighted_query_terms`. `query_entity_weights`
        (optional): {entity_id: weight} dict (same shape, built the same
        way from recent history's entities) -- adds
        `entity_boost * sum(query_entity_weights[e] for e in candidate's
        entities)` on top of the lexical score. Silently ignored if this
        index wasn't fit with `entities`."""
        vocab = self.vectorizer.vocabulary_
        cols, wts = [], []
        for t, w in query_weights.items():
            col = vocab.get(t)
            if col is not None:
                cols.append(col)
                wts.append(w)
        scores = np.zeros(len(candidate_ids))
        rows = [self.id_to_row.get(c) for c in candidate_ids]
        valid = [(i, r) for i, r in enumerate(rows) if r is not None]
        if not valid:
            return scores
        idxs = [i for i, _ in valid]
        rs = [r for _, r in valid]

        if cols:
            wvec = np.asarray(wts)
            sub = self.bm25_matrix[rs, :][:, cols]
            scores[idxs] = np.asarray(sub.dot(wvec)).ravel()

        if query_entity_weights and self.doc_entities is not None:
            for i, r in zip(idxs, rs):
                if self.doc_entities[r]:
                    scores[i] += self.entity_boost * sum(
                        query_entity_weights.get(e, 0.0) for e in self.doc_entities[r])
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

    def search_topk(self, query_weight_dicts, k):
        """Q2.3/Q2.4 exact top-K retrieval over the entire corpus, without
        ever densifying a (batch x n_docs) array the way `score_batch_full`
        does -- same sparse-sparse matmul, but the result is consumed
        row-by-row (only its nonzero entries touched), which is what
        speeds this up: a BM25 query typically only matches a small
        fraction of the corpus, so `.todense()` and a full-width
        `np.argpartition` were doing far more work than the data required.
        Mirrors SBERTIndex.search_topk's (scores, row_indices) interface
        (each (B, k)) so scripts/evaluate_retrieval.py can use the same
        `recall_hits_from_topk` for both backends.

        A query with fewer than k docs sharing any term pads the remaining
        slots with -1. Those unmatched docs all have score 0 (no lexical
        overlap at all); which one of potentially thousands of zero-score
        docs would fill a slot is not a meaningful distinction, so treating
        them as "not found" is more defensible than `score_batch_full`'s
        dense `np.argpartition`, which made an arbitrary, undocumented tie
        choice among them."""
        vocab = self.vectorizer.vocabulary_
        rows, cols, data = [], [], []
        for qi, qweights in enumerate(query_weight_dicts):
            for t, w in qweights.items():
                col = vocab.get(t)
                if col is not None:
                    rows.append(qi)
                    cols.append(col)
                    data.append(w)
        n_queries = len(query_weight_dicts)
        qmat = sp.csr_matrix((data, (rows, cols)), shape=(n_queries, self.bm25_matrix.shape[1]))
        raw = qmat.dot(self.bm25_matrix.T).tocsr()  # sparse (n_queries x n_docs)

        out_scores = np.full((n_queries, k), -np.inf)
        out_idx = np.full((n_queries, k), -1, dtype=int)
        for i in range(n_queries):
            start, end = raw.indptr[i], raw.indptr[i + 1]
            row_cols = raw.indices[start:end]
            row_data = raw.data[start:end]
            n = len(row_data)
            if n == 0:
                continue
            kk = min(k, n)
            top_part = np.argpartition(-row_data, kk - 1)[:kk] if n > kk else np.arange(n)
            order = top_part[np.argsort(-row_data[top_part])]
            out_scores[i, :kk] = row_data[order]
            out_idx[i, :kk] = row_cols[order]
        return out_scores, out_idx
