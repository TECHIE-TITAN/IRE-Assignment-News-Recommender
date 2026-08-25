"""Sentence-transformer article embeddings + FAISS retrieval -- an
alternative Q3 semantic backend to retrieval/lsa.py's TF-IDF+SVD.

Where LSA is a linear, co-occurrence-based embedding (fast, no download,
but limited to what SVD can extract from term co-occurrence statistics),
this uses a pretrained transformer sentence encoder (default:
all-MiniLM-L6-v2, 384-dim) to produce genuinely contextual dense
embeddings. Still unsupervised with respect to MIND: the model is used
frozen, at inference time only -- no fine-tuning on MIND's click labels,
same role a pretrained word2vec/GloVe/BERT embedding would play. The
model's *own* training (by its original authors, on unrelated sentence-pair
data) did use supervision, which is a fair distinction to keep in mind but
doesn't touch the "no learning from click logs" boundary this pipeline is
holding to.

FAISS (`IndexFlatIP`, exact inner-product search over L2-normalized
vectors = exact cosine) replaces LSAIndex's brute-force `score_batch_full`
dense-matrix approach for the Q2/Q3 full-corpus top-K retrieval path --
`index.search` returns top-K directly, so a (batch x n_docs) dense matrix
is never materialized at all for this backend, an improvement in kind, not
just a swap. `IndexFlatIP` is *exact*, not approximate -- there's no
measured latency problem at this corpus size (100-121K articles) to trade
accuracy for.

Entity fusion, recency-weighted pooling, and the score_candidates/
get_embedding/embeddings/id_to_row surface are all identical in shape to
retrieval/lsa.py's LSAIndex, so this is a drop-in replacement wherever an
LSAIndex was used -- see retrieval/build_indices.py's `semantic_backend`
switch.
"""

import numpy as np

from retrieval.entity_embeddings import fuse_embeddings

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# EB-NeRD is Danish text; all-MiniLM-L6-v2 is trained (effectively)
# English-only, so using it on Danish would be a real quality bug, not just
# a suboptimal choice -- paraphrase-multilingual-MiniLM-L12-v2 covers 50+
# languages including Danish. scripts/build_indices.py selects this
# automatically for --dataset ebnerd unless --sbert_model is given explicitly.
DEFAULT_MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SBERTIndex:
    def __init__(self, model_name=DEFAULT_MODEL, batch_size=256, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None  # lazily loaded; never pickled, see __getstate__
        self.doc_ids = None
        self.id_to_row = None
        self.embeddings = None  # (n_docs, d), L2-normalized, possibly entity-fused
        self.entity_dim = 0
        self.entity_weight = 1.0
        self.index = None  # faiss.IndexFlatIP; never pickled, rebuilt from self.embeddings

    def __getstate__(self):
        """The loaded transformer model and the FAISS index are excluded
        from pickling: the model is a large, sometimes device-bound object
        better lazily reloaded by name than serialized, and FAISS indexes
        aren't reliably picklable at all. `__setstate__` rebuilds the FAISS
        index from `self.embeddings` (cheap -- a few hundred K vectors add
        in well under a second) and leaves the model unloaded until the
        next call that actually needs it (`embed`)."""
        state = self.__dict__.copy()
        state["model"] = None
        state["index"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.embeddings is not None:
            self._build_faiss_index()

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def _encode(self, texts):
        model = self._get_model()
        emb = model.encode(list(texts), batch_size=self.batch_size, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
        return np.ascontiguousarray(emb.astype(np.float32))

    def _build_faiss_index(self):
        import faiss
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(np.ascontiguousarray(self.embeddings.astype(np.float32)))

    def fit(self, doc_ids, texts, entity_embeddings=None, entity_weight=1.0):
        """`entity_embeddings`: optional {article_id: (ENTITY_DIM,) array}
        (see retrieval/entity_embeddings.py) -- same fusion as LSAIndex."""
        content_emb = self._encode(texts)
        self.doc_ids = list(doc_ids)
        self.id_to_row = {a: i for i, a in enumerate(self.doc_ids)}

        if entity_embeddings is not None:
            entity_dim = next(iter(entity_embeddings.values())).shape[0]
            zero_entity = np.zeros(entity_dim)
            entity_mat = np.array([entity_embeddings.get(a, zero_entity) for a in self.doc_ids])
            self.entity_dim = entity_mat.shape[1]
            self.entity_weight = entity_weight
            self.embeddings = fuse_embeddings(content_emb, entity_mat, entity_weight).astype(np.float32)
        else:
            self.entity_dim = 0
            self.embeddings = content_emb

        self._build_faiss_index()
        return self

    def embed(self, article_ids, texts, entity_embeddings_list=None):
        """Out-of-corpus embedding for text not (necessarily) in this
        index. For any `article_ids` entry already present in the fit
        corpus, reuses the already-computed (and possibly entity-fused)
        embedding directly instead of re-running the transformer --
        meaningful when called over a combined universe that heavily
        overlaps the fit corpus (generate_predictions.py's history lookup
        spans train+val+test, and the test articles are already embedded
        at fit time). Unlike LSA's near-instant TF-IDF transform, SBERT
        encoding is the expensive step here, so this caching isn't optional
        polish -- without it, the test corpus would be encoded twice."""
        d = self.embeddings.shape[1]
        out = np.zeros((len(article_ids), d), dtype=np.float32)
        to_encode_idx = []
        for i, aid in enumerate(article_ids):
            row = self.id_to_row.get(aid)
            if row is not None:
                out[i] = self.embeddings[row]
            else:
                to_encode_idx.append(i)

        if to_encode_idx:
            new_texts = [texts[i] for i in to_encode_idx]
            content_emb = self._encode(new_texts)
            if self.entity_dim > 0:
                entity_mat = np.array([
                    entity_embeddings_list[i] if (entity_embeddings_list and entity_embeddings_list[i] is not None)
                    else np.zeros(self.entity_dim)
                    for i in to_encode_idx
                ])
                fused = fuse_embeddings(content_emb, entity_mat, self.entity_weight).astype(np.float32)
            else:
                fused = content_emb
            for j, i in enumerate(to_encode_idx):
                out[i] = fused[j]
        return out

    def get_embedding(self, article_id):
        row = self.id_to_row.get(article_id)
        return self.embeddings[row] if row is not None else None

    def score_candidates(self, user_vec, candidate_ids):
        """Q5 path: cosine score for one impression's candidate list --
        plain numpy, not FAISS. FAISS is for full-corpus top-K search; a
        handful of specific candidate IDs is cheaper and simpler to score
        directly (same reasoning as BM25Index.score_candidates)."""
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

    def search_topk(self, user_vecs, k):
        """Q2.2/Q3.2 FAISS ANN retrieval: top-K over the *entire* indexed
        corpus for a batch of user vectors. Returns (scores, row_indices),
        each (B, k), already ranked descending (FAISS's native IP-index
        output order) -- no B x n_docs dense matrix ever materialized,
        unlike LSAIndex.score_batch_full's brute-force path. A missing
        slot (fewer than k docs in the corpus) comes back as row index -1;
        callers must filter those out."""
        user_vecs = np.ascontiguousarray(user_vecs.astype(np.float32))
        scores, idx = self.index.search(user_vecs, k)
        return scores, idx
