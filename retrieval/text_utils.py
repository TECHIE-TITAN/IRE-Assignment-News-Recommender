"""Shared text-prep and query-construction helpers for lexical (BM25) and
semantic (LSA) retrieval.
"""

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def identity_preprocessor(text):
    """Passed to CountVectorizer/TfidfVectorizer as `preprocessor=` so they
    skip their own built-in lowercasing/accent-stripping (already handled
    by `tokenize`) without disabling preprocessing entirely. Must be a
    named module-level function, not a lambda -- BM25Index/LSAIndex store
    the fitted vectorizer on `self`, and a lambda closure isn't picklable
    (scripts/build_indices.py persists these with pickle)."""
    return text


def tokenize(text):
    if text is None or (isinstance(text, float)):
        return []
    return _TOKEN_RE.findall(str(text).lower())


def article_text(title, abstract):
    """Q2.1 / Q3.1: article text = title + abstract."""
    parts = [p for p in (title, abstract) if p and isinstance(p, str)]
    return " ".join(parts)


def recency_weights(n, decay=0.85):
    """Weight for the i-th (oldest-to-newest, 0-indexed) of `n` recent
    history items: the most recent (index n-1) gets weight `decay**0 = 1.0`,
    each item further back is discounted by one more factor of `decay`.
    Shared by BM25's `weighted_query_terms` and LSA's recency-weighted
    mean pooling so both sides of retrieval use the same decay curve."""
    return [decay ** (n - 1 - i) for i in range(n)]


def weighted_query_terms(history_ids, article_text_lookup, recent_n=20, decay=0.85):
    """Q2.2 / Q3.3, extended with recency weighting: builds a
    `{term: weight}` query from the user's last `recent_n` history
    articles' *full* text (title+abstract via `article_text_lookup` --
    richer than titles alone, matching what the BM25 corpus itself is
    indexed on), where more recently clicked articles contribute more
    weight. Terms appearing in multiple recent articles accumulate weight
    naturally (a term seen in 3 of the last 20 clicks is a stronger signal
    than one seen in 1).

    `decay=1.0` degenerates every recent article to weight 1.0 regardless
    of position -- i.e. the original equal-weight behavior, just with a
    richer per-article text source. MIND's `history` field is
    chronologically ascending, so the *last* entry is the most recent click.
    """
    if len(history_ids) == 0:
        return {}
    recent = list(history_ids)[-recent_n:]
    n = len(recent)
    weights_per_article = recency_weights(n, decay)
    terms = {}
    for aid, w in zip(recent, weights_per_article):
        text = article_text_lookup.get(aid, "")
        for t in set(tokenize(text)):
            terms[t] = terms.get(t, 0.0) + w
    return terms
