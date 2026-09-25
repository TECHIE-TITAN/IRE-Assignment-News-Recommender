"""NRMS (Wu et al., 2019) -- a from-scratch, deliberately small PyTorch
re-implementation, used as a neural comparison point against the LightGBM
`LGBMRanker` re-ranker (scripts/train_reranker.py). Not a port of the
official `ebnerd-benchmark`/Microsoft-Recommenders codebase -- this is a
new implementation of the ARCHITECTURE the paper describes (multi-head
self-attention news encoder -> multi-head self-attention user encoder ->
dot-product click score), scoped down for GPU-trainable-in-one-job
tractability: trainable (not pretrained) word embeddings, title-only text
(no abstract field), and small embedding/hidden dims.

Dataset-agnostic by construction: unlike the GBDT re-ranker's Q1 features
(pipeline/features_{mind,ebnerd}.py, genuinely dataset-specific), NRMS only
needs {article_id: title text} and the unified interactions schema
(history_article_ids/candidate_article_ids/labels -- pipeline/schema.py),
which both MIND and EB-NeRD already share. One model class, one training
script (scripts/train_nrms.py), --dataset switches only the data it reads.

Negative sampling, exactly matching the paper's own training procedure:
one training instance per POSITIVE click, paired with K negatives sampled
from that SAME impression's own displayed-but-not-clicked candidates (not
the full corpus -- consistent with this project's restricted-candidate
scoring convention elsewhere, see CLAUDE.md's score_candidates vs
score_batch_full distinction), scored jointly with a K+1-way softmax
cross-entropy loss. At evaluation time (scripts/train_nrms.py's val pass),
every candidate in an impression is scored, not just K+1 -- exactly how
scripts/train_reranker.py's LGBMRanker.predict() and Assignment 1's
score_candidates both already evaluate.
"""

import numpy as np
import torch
import torch.nn as nn

from retrieval.text_utils import tokenize

PAD_IDX = 0
UNK_IDX = 1


def build_vocab(titles, max_vocab_size=30_000):
    """`titles`: iterable of raw title strings (title-only, not
    title+abstract -- see module docstring). Returns {token: id}, with 0
    reserved for PAD and 1 for UNK/out-of-vocab, and the remaining ids
    assigned to the `max_vocab_size - 2` most frequent tokens -- capping
    vocab size bounds the embedding table's memory, which matters more
    here than for BM25/TF-IDF's sparse vectorizers since every embedding
    row is a live, gradient-tracked dense parameter."""
    counts = {}
    for t in titles:
        for tok in set(tokenize(t)):
            counts[tok] = counts.get(tok, 0) + 1
    most_common = sorted(counts.items(), key=lambda kv: -kv[1])[: max_vocab_size - 2]
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for tok, _ in most_common:
        vocab[tok] = len(vocab)
    return vocab


def title_to_ids(title, vocab, max_title_len):
    ids = [vocab.get(tok, UNK_IDX) for tok in tokenize(title)][:max_title_len]
    ids += [PAD_IDX] * (max_title_len - len(ids))
    return ids


class AdditiveAttention(nn.Module):
    """Standard NRMS pooling: a learned query vector attends over a
    sequence's already-contextualized (post-MHSA) representations and
    returns one weighted-sum vector. `-1e4`, not `-inf`, for masked
    positions: a fully-masked row (a cold-start user with zero real
    history entries) would otherwise soften to NaN (softmax of all -inf);
    -1e4 instead softens to a harmless uniform distribution over PAD
    positions, whose embeddings are zero (nn.Embedding's `padding_idx`
    rows never receive gradient updates), so a fully-masked row correctly
    pools to a zero vector -- the same "cold start -> neutral zero, not a
    fabricated guess" convention pipeline/features_common.py already uses
    for the GBDT's own history-embedding-similarity feature."""

    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.1)

    def forward(self, x, mask):
        # x: (batch, seq, dim); mask: (batch, seq) bool, True = real token
        scores = torch.tanh(self.proj(x)) @ self.query  # (batch, seq)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class NewsEncoder(nn.Module):
    """title token ids -> (batch, embed_dim) news vector. Multi-head
    self-attention over the title's own tokens, then additive-attention
    pooling -- exactly the paper's news encoder, minus its optional
    category/subcategory embedding arms (out of scope here; the GBDT
    re-ranker already covers category-aware features separately)."""

    def __init__(self, vocab_size, embed_dim=100, num_heads=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool = AdditiveAttention(embed_dim)

    def forward(self, title_ids):  # (N, title_len)
        mask = title_ids != PAD_IDX
        x = self.dropout(self.embedding(title_ids))
        # A title that's entirely PAD (shouldn't happen for a real
        # article, but defends against one with empty/unparseable text)
        # would give MultiheadAttention a fully-masked key_padding_mask,
        # which PyTorch turns into NaN outputs -- guarded by forcing at
        # least the first position "visible" to attention for the
        # few rows this affects, while AdditiveAttention above still masks
        # correctly on the real `mask` so the pooled result stays governed
        # by real tokens whenever any exist.
        # attn_mask is `mask` with at least one position forced True per
        # row (for MultiheadAttention's key_padding_mask, which NaNs on a
        # fully-masked row) -- identical to `mask` for every row that
        # already had a real token, so it's always safe to pass to the
        # pooling step below too, not just the empty rows.
        attn_mask = mask.clone()
        empty_rows = ~attn_mask.any(dim=1)
        if empty_rows.any():
            attn_mask[empty_rows, 0] = True
        attn_out, _ = self.mhsa(x, x, x, key_padding_mask=~attn_mask)
        attn_out = self.dropout(attn_out)
        return self.pool(attn_out, attn_mask)


class UserEncoder(nn.Module):
    """Sequence of recent history news vectors -> (batch, embed_dim) user
    vector, same MHSA-then-additive-attention shape as NewsEncoder, over
    news vectors instead of word embeddings -- "attention over history",
    one of Q3's own example principled improvements."""

    def __init__(self, news_dim=100, num_heads=4, dropout=0.2):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(news_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool = AdditiveAttention(news_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, history_vecs, mask):  # (batch, hist_len, dim), (batch, hist_len)
        attn_mask = mask.clone()
        empty_rows = ~attn_mask.any(dim=1)
        if empty_rows.any():
            attn_mask[empty_rows, 0] = True
        attn_out, _ = self.mhsa(history_vecs, history_vecs, history_vecs, key_padding_mask=~attn_mask)
        attn_out = self.dropout(attn_out)
        return self.pool(attn_out, attn_mask)


class NRMS(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_heads=4, dropout=0.2):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, embed_dim, num_heads, dropout)
        self.user_encoder = UserEncoder(embed_dim, num_heads, dropout)
        self.embed_dim = embed_dim

    def encode_news(self, title_ids):
        """title_ids: (N, title_len) -> (N, embed_dim). Flat, so callers
        batch arbitrary sets of articles (e.g. "every unique article in
        this eval chunk") through one call instead of per-impression."""
        return self.news_encoder(title_ids)

    def encode_user(self, history_title_ids, history_mask):
        """history_title_ids: (batch, hist_len, title_len); history_mask:
        (batch, hist_len) bool."""
        batch, hist_len, title_len = history_title_ids.shape
        news_vecs = self.news_encoder(history_title_ids.view(batch * hist_len, title_len))
        news_vecs = news_vecs.view(batch, hist_len, self.embed_dim)
        return self.user_encoder(news_vecs, history_mask)

    def forward(self, history_title_ids, history_mask, cand_title_ids):
        """cand_title_ids: (batch, n_cand, title_len) -- training calls
        this with n_cand = num_negatives + 1 (1 positive first); eval calls
        it with n_cand = that impression's full candidate list. Returns
        (batch, n_cand) raw dot-product scores (softmax/sigmoid applied by
        the caller depending on training vs. inference)."""
        user_vec = self.encode_user(history_title_ids, history_mask)  # (batch, dim)
        batch, n_cand, title_len = cand_title_ids.shape
        cand_vecs = self.encode_news(cand_title_ids.view(batch * n_cand, title_len))
        cand_vecs = cand_vecs.view(batch, n_cand, self.embed_dim)
        return torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)  # (batch, n_cand)


def pad_history(history_ids, max_history_len):
    """Most-recent-`max_history_len` of `history_ids`, left-padded with
    None so real entries stay right-aligned (most recent at the end,
    matching this project's own history-ordering convention elsewhere --
    see pipeline.features_common.recent_history_with_weights). Returns
    (padded_ids, mask) where mask[i] is False for a None/pad slot."""
    recent = list(history_ids)[-max_history_len:]
    pad_n = max_history_len - len(recent)
    padded = [None] * pad_n + recent
    mask = [False] * pad_n + [True] * len(recent)
    return padded, mask
