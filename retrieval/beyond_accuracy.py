"""Q4.2: beyond-accuracy metrics -- intra-list diversity, novelty, coverage.

Diversity and novelty are per-recommended-list (i.e. per impression, over
its top-K items); coverage is inherently corpus-level (a set union over
*all* evaluated impressions' top-K lists), so it's computed directly in
scripts/evaluate_ranking.py rather than here -- see that script's
vectorized boolean-array coverage + bootstrap implementation.
"""

import numpy as np


def intra_list_diversity(item_embeddings):
    """1 - mean pairwise cosine similarity among a recommended list's items.
    `item_embeddings` rows are assumed L2-normalized (so a dot product is a
    cosine similarity). NaN for lists with fewer than 2 items (no pairs)."""
    n = len(item_embeddings)
    if n < 2:
        return float("nan")
    mat = np.asarray(item_embeddings)
    sim = mat @ mat.T
    iu = np.triu_indices(n, k=1)
    return float(1.0 - sim[iu].mean())


def novelty_score(article_ids, popularity_prob_lookup):
    """Self-information novelty: mean over the list of -log2(p(item)),
    where p(item) is a (Laplace-smoothed) train-split popularity
    probability -- see `train_popularity_prob` for how that's built.
    Never-clicked/out-of-lookup items get the lookup's smoothed floor
    probability rather than an undefined/infinite novelty."""
    if not article_ids:
        return float("nan")
    default = popularity_prob_lookup.get("__default__")
    vals = [-np.log2(popularity_prob_lookup.get(a, default)) for a in article_ids]
    return float(np.mean(vals))


def train_popularity_prob(article_features_df):
    """Builds the {article_id: p(item)} lookup novelty_score needs:
    p(item) = (train_clicks(item) + 1) / (total_train_clicks + n_articles)
    (add-one/Laplace smoothing over the item catalog, so unseen items get a
    small-but-nonzero, well-defined probability instead of 0/inf novelty).
    `"__default__"` holds that smoothed floor for articles not in this
    table at all (e.g. absent from the train split entirely)."""
    clicks = article_features_df["train_clicks"].fillna(0)
    total_clicks = float(clicks.sum())
    n_articles = len(article_features_df)
    denom = total_clicks + n_articles
    probs = (clicks + 1.0) / denom
    lookup = dict(zip(article_features_df["article_id"], probs))
    lookup["__default__"] = 1.0 / denom
    return lookup
