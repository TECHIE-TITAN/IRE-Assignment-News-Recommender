"""TF-IDF + truncated SVD (latent semantic analysis) article embeddings
(Q3), optionally fused with MIND's own pretrained entity embeddings
(retrieval/entity_embeddings.py).

MIND ships no article embeddings (unlike EB-NeRD) and only sparse,
per-entity TransE vectors that miss articles with no linked entities -- so
content-only LSA is the "compute your own" path Q3 allows for. Fusing in
the entity vectors (when provided at fit time) is a genuinely richer
*unsupervised* semantic representation -- content co-occurrence structure
plus Wikidata knowledge-graph structure -- still no click labels involved,
still pure scikit-learn, no model download.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from retrieval.entity_embeddings import fuse_embeddings
from retrieval.text_utils import identity_preprocessor, tokenize


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
        self.embeddings = None       # (n_docs x d), L2-normalized, possibly entity-fused
        self.entity_dim = 0          # 0 if fit without entity fusion
        self.entity_weight = 1.0

    def fit(self, doc_ids, texts, entity_embeddings=None, entity_weight=1.0):
        """`entity_embeddings`: optional {article_id: (ENTITY_DIM,) array}
        (see retrieval/entity_embeddings.py). When given, the final
        `self.embeddings` is the L2-normalized concatenation of the content
        embedding and `entity_weight * entity_embedding` per article
        (`fuse_embeddings`), not content alone."""
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, preprocessor=identity_preprocessor, lowercase=False,
            min_df=self.min_df, max_df=self.max_df, token_pattern=None,
        )
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        content_emb = normalize(self.svd.fit_transform(tfidf))
        self.doc_ids = list(doc_ids)
        self.id_to_row = {a: i for i, a in enumerate(self.doc_ids)}

        if entity_embeddings is not None:
            entity_dim = next(iter(entity_embeddings.values())).shape[0]
            zero_entity = np.zeros(entity_dim)
            entity_mat = np.array([entity_embeddings.get(a, zero_entity) for a in self.doc_ids])
            self.entity_dim = entity_mat.shape[1]
            self.entity_weight = entity_weight
            self.embeddings = fuse_embeddings(content_emb, entity_mat, entity_weight)
        else:
            self.entity_dim = 0
            self.embeddings = content_emb
        return self

    def embed(self, article_ids, texts, entity_embeddings_list=None):
        """Out-of-corpus embedding for text not (necessarily) in this
        index. `article_ids` (same signature as SBERTIndex.embed, so
        callers like generate_predictions.py don't need to branch on
        backend): for any id already present in the fit corpus, reuses the
        already-computed embedding directly rather than re-transforming --
        cheap either way for LSA (TF-IDF+SVD transform is near-instant),
        but kept for interface parity with SBERTIndex, where this caching
        is the difference between encoding a combined article universe once
        vs. twice. Reuses the fitted vectorizer+SVD so newly-transformed
        text stays in the same space; if this index was fit with entity
        fusion, pass a parallel `entity_embeddings_list` (one (ENTITY_DIM,)
        array or None per *newly-transformed* text) to keep the fused space
        consistent -- articles with no entry get a zero entity block
        (degrades to content-only, see `fuse_embeddings`)."""
        d = self.embeddings.shape[1]
        out = np.zeros((len(article_ids), d))
        to_encode_idx = [i for i, aid in enumerate(article_ids) if self.id_to_row.get(aid) is None]
        for i, aid in enumerate(article_ids):
            row = self.id_to_row.get(aid)
            if row is not None:
                out[i] = self.embeddings[row]
        if not to_encode_idx:
            return out

        new_texts = [texts[i] for i in to_encode_idx]
        tfidf = self.vectorizer.transform(new_texts)
        content_emb = normalize(self.svd.transform(tfidf))
        if self.entity_dim > 0:
            entity_mat = np.array([
                entity_embeddings_list[i] if (entity_embeddings_list and entity_embeddings_list[i] is not None)
                else np.zeros(self.entity_dim)
                for i in to_encode_idx
            ])
            fused = fuse_embeddings(content_emb, entity_mat, self.entity_weight)
        else:
            fused = content_emb
        for j, i in enumerate(to_encode_idx):
            out[i] = fused[j]
        return out

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


def mean_pool_user_vector(article_embeddings, weights=None):
    """Q3.3: user representation = mean-pooled embeddings of clicked
    articles, re-normalized so downstream dot products stay cosine
    similarities. `weights` (optional): parallel list of per-article
    recency weights (see `text_utils.recency_weights`) -- when given, more
    recently clicked articles' embeddings pull the pooled vector harder
    instead of every recent click counting equally."""
    idxs = [i for i, v in enumerate(article_embeddings) if v is not None]
    if not idxs:
        return None
    vecs = np.array([article_embeddings[i] for i in idxs])
    if weights is None:
        pooled = vecs.mean(axis=0)
    else:
        w = np.array([weights[i] for i in idxs]).reshape(-1, 1)
        pooled = (vecs * w).sum(axis=0) / w.sum()
    norm = np.linalg.norm(pooled)
    return pooled / norm if norm > 0 else pooled
