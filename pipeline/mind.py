"""Load raw MIND TSV files and map them into the unified schema defined in
pipeline/schema.py.

MIND quirks that shape this loader:
  - `news.tsv` has no article body field -> unified schema has no `body` col.
  - `behaviors.tsv`'s `history` field has no per-click timestamps, so it
    cannot be used to build a timestamped click-history table. Instead,
    timestamped clicks are derived from positive-labelled impression
    candidates (a click on a shown candidate has a known timestamp: the
    impression time).
  - news_id / user_id namespaces are shared across train/dev/test (no
    remapping needed). impression_id is only unique *within* a file, so it
    gets a split-name prefix when files are concatenated.
  - The large test split's `impressions` field carries no `-0`/`-1` labels
    (just bare news IDs) since it's the Codabench-held-out set.
"""

import json
import os

import pandas as pd

from pipeline.schema import ARTICLE_COLUMNS, INTERACTION_COLUMNS

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def load_news_raw(dir_path):
    fp = os.path.join(dir_path, "news.tsv")
    return pd.read_csv(fp, sep="\t", header=None, names=NEWS_COLS, quoting=3,
                        na_values=[""], keep_default_na=True)


def load_behaviors_raw(dir_path):
    fp = os.path.join(dir_path, "behaviors.tsv")
    df = pd.read_csv(fp, sep="\t", header=None, names=BEH_COLS, quoting=3,
                      na_values=[""], keep_default_na=True)
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    return df


def _extract_entity_ids(entity_json):
    """MIND stores title/abstract entities as a JSON array of dicts with a
    WikidataId field. Returns a de-duplicated, order-preserving list of ids."""
    if pd.isna(entity_json) or not str(entity_json).strip():
        return []
    try:
        items = json.loads(entity_json)
    except (json.JSONDecodeError, TypeError):
        return []
    seen, out = set(), []
    for item in items:
        wid = item.get("WikidataId")
        if wid and wid not in seen:
            seen.add(wid)
            out.append(wid)
    return out


def _parse_history(hist):
    if pd.isna(hist):
        return []
    return str(hist).split()


def _parse_impressions(imp):
    """Returns (candidate_ids, labels) as two parallel lists. `labels`
    entries are None for the unlabeled test split (bare news IDs, no
    "-0"/"-1" suffix -- MIND news IDs never contain "-", so this split is
    unambiguous)."""
    if pd.isna(imp):
        return [], []
    cand_ids, labels = [], []
    for tok in str(imp).split():
        if "-" in tok:
            nid, lab = tok.rsplit("-", 1)
            try:
                lab = int(lab)
            except ValueError:
                continue
        else:
            nid, lab = tok, None
        cand_ids.append(nid)
        labels.append(lab)
    return cand_ids, labels


def articles_from_news_df(news):
    """Maps a raw news.tsv-shaped dataframe to the unified article schema."""
    title_entities = news["title_entities"].apply(_extract_entity_ids)
    abstract_entities = news["abstract_entities"].apply(_extract_entity_ids)
    entities = [sorted(set(a) | set(b), key=(a + b).index) if (a or b) else []
                for a, b in zip(title_entities, abstract_entities)]

    out = pd.DataFrame({
        "article_id": news["news_id"],
        "title": news["title"],
        "abstract": news["abstract"],
        "category": news["category"],
        "subcategory": news["subcategory"],
        "entities": entities,
        "url": news["url"],
    })
    return out[ARTICLE_COLUMNS]


def load_mind_articles(*dir_paths):
    """Union of news.tsv across the given directories, deduplicated by
    article_id, mapped to the unified article schema."""
    news = pd.concat([load_news_raw(d) for d in dir_paths], ignore_index=True)
    news = news.drop_duplicates(subset="news_id", keep="first")
    return articles_from_news_df(news)


def interactions_from_behaviors_df(beh, split):
    history = beh["history"].apply(_parse_history)
    cand_labels = beh["impressions"].apply(_parse_impressions)
    candidates = cand_labels.apply(lambda cl: cl[0])
    labels = cand_labels.apply(lambda cl: cl[1])

    out = pd.DataFrame({
        "impression_id": f"{split}_" + beh["impression_id"].astype(str),
        "user_id": beh["user_id"],
        "impression_time": beh["time"],
        "history_article_ids": history,
        "candidate_article_ids": candidates,
        "labels": labels,
        "split": split,
    })
    return out[INTERACTION_COLUMNS]


def load_mind_interactions(split_dirs):
    """`split_dirs`: dict mapping split name ("train"/"val"/"test") -> raw
    directory path. Split is assigned directly from which file a row came
    from (MIND's own train/dev/large_test files are already temporally
    disjoint -- see pipeline/split.py for the boundary-monotonicity check
    that verifies this)."""
    frames = []
    for split, dir_path in split_dirs.items():
        beh = load_behaviors_raw(dir_path)
        frames.append(interactions_from_behaviors_df(beh, split))
    return pd.concat(frames, ignore_index=True)[INTERACTION_COLUMNS]


def derive_mind_click_history(interactions):
    """The only timestamped clicks available in MIND are positive-labelled
    impression candidates: a click on a shown candidate has a known
    timestamp (the impression time). Unlabeled (test-split) rows have
    labels == None, which naturally drops out of the `labels == 1` filter,
    so this is safe to call on the full train+val+test interactions table."""
    df = interactions[["user_id", "impression_time", "candidate_article_ids", "labels", "split"]].copy()
    df = df.explode(["candidate_article_ids", "labels"])
    df = df[df["labels"] == 1]
    ch = df.rename(columns={"candidate_article_ids": "article_id", "impression_time": "click_time"})
    return ch[["user_id", "article_id", "click_time", "split"]].reset_index(drop=True)
