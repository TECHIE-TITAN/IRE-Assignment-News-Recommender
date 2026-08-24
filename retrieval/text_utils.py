"""Shared text-prep helpers for lexical (BM25) and semantic (LSA) retrieval."""

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    if text is None or (isinstance(text, float)):
        return []
    return _TOKEN_RE.findall(str(text).lower())


def article_text(title, abstract):
    """Q2.1 / Q3.1: article text = title + abstract."""
    parts = [p for p in (title, abstract) if p and isinstance(p, str)]
    return " ".join(parts)


def build_query_text(history_ids, article_title_lookup, recent_n=20):
    """Q2.2 / Q3.3: "concatenate titles of recently clicked articles". Uses
    only the last `recent_n` history entries (MIND's `history` field is
    chronologically ordered) both because the assignment asks for *recent*
    history specifically, and because it bounds query size for the
    2.37M-impression prediction pass (Q5)."""
    if len(history_ids) == 0:
        return ""
    recent = history_ids[-recent_n:]
    titles = [article_title_lookup.get(a, "") for a in recent]
    return " ".join(t for t in titles if t)
