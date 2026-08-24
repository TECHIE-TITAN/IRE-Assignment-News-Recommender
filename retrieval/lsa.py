"""TF-IDF + truncated SVD (latent semantic analysis) article embeddings (Q3).

MIND ships no article embeddings (unlike EB-NeRD) and only sparse,
per-entity TransE vectors that miss articles with no linked entities -- so
this is the "compute your own" path Q3 allows for. LSA is the classic dense
semantic-retrieval baseline: it captures co-occurrence structure beyond
exact term overlap (two articles about the same event with no shared
vocabulary can still end up nearby), while staying pure scikit-learn with no
model download and near-instant fit/transform even over the ~121K-article
MINDlarge_test corpus.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from retrieval.text_utils import tokenize


class LSAIndex:
    def __init__(self, n_components=128, min_df=2, max_df=0.6, random_state=42):
        self.n_components = n_components
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        self.vectorizer = None
        self.svd = None
        self.doc_ids = None
        self.id_to_row = None
        self.embeddings = None  # (n_docs x n_components), L2-normalized

    def fit(self, doc_ids, texts):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, preprocessor=lambda s: s, lowercase=False,
            min_df=self.min_df, max_df=self.max_df, token_pattern=None,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        emb = self.svd.fit_transform(tfidf)
        self.embeddings = normalize(emb)  # so score = dot product = cosine similarity
        self.doc_ids = list(doc_ids)
        self.id_to_row = {a: i for i, a in enumerate(self.doc_ids)}
        return self

    def embed(self, texts):
        """Out-of-corpus embedding for text not in this index (e.g. a
        history article from outside the currently-indexed corpus). Reuses
        the fitted vectorizer+SVD so the result stays in the same space."""
        tfidf = self.vectorizer.transform(texts)
        emb = self.svd.transform(tfidf)
        return normalize(emb)

    def get_embedding(self, article_id):
        row = self.id_to_row.get(article_id)
        return self.embeddings[row] if row is not None else None

    def score_candidates(self, user_vec, candidate_ids):
        """Q5 path: cosine score for one impression's candidate list."""
        scores = np.zeros(len(candidate_ids))
        if user_vec is None:
            return scores
        rows = [self.id_to_row.get(c) for c in candidate_ids]
        valid = [(i, r) for i, r in enumerate(rows) if r is not None]
        if not valid:
            return scores
        idxs = [i for i, _ in valid]
        rs = [r for _, r in valid]
        scores[idxs] = self.embeddings[rs].dot(user_vec)
        return scores

    def score_batch_full(self, user_vecs):
        """Q3.3/Q3.4: dense (B x n_docs) cosine-score matrix for a batch of
        user vectors against the entire indexed corpus."""
        return user_vecs.dot(self.embeddings.T)


def mean_pool_user_vector(article_embeddings):
    """Q3.3: user representation = mean-pooled embeddings of clicked
    articles, re-normalized so downstream dot products stay cosine
    similarities."""
    vecs = [v for v in article_embeddings if v is not None]
    if not vecs:
        return None
    mean = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean
