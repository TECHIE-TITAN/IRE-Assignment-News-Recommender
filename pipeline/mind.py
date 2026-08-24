"""Load MIND (MINDsmall_train / MINDsmall_dev) raw files and map them into the
unified schema defined in pipeline/schema.py.

MIND quirks that shape this loader:
  - `news.tsv` has no article body field -> unified `body` is always None.
  - `behaviors.tsv`'s `history` field has no per-click timestamps, so it cannot
    be used to build a timestamped click-history table. Instead, timestamped
    clicks are derived from positive-labelled impression candidates (a click
    on a shown candidate has a known timestamp: the impression time).
  - train/dev news_id and user_id namespaces are shared (no remapping needed);
    impression_id is only unique *within* a file, so it gets a dataset+source
    prefix when train and dev are concatenated.
"""

import json
import os

import pandas as pd

from pipeline.schema import ARTICLE_COLUMNS, INTERACTION_COLUMNS

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def _load_news_raw(dir_path):
    fp = os.path.join(dir_path, "news.tsv")
    return pd.read_csv(fp, sep="\t", header=None, names=NEWS_COLS, quoting=3,
                        na_values=[""], keep_default_na=True)


def _load_behaviors_raw(dir_path):
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
    """Returns (candidate_ids, labels) as two parallel lists."""
    if pd.isna(imp):
        return [], []
    cand_ids, labels = [], []
    for tok in str(imp).split():
        if "-" not in tok:
            continue
        nid, lab = tok.rsplit("-", 1)
        try:
            lab = int(lab)
        except ValueError:
            continue
        cand_ids.append(nid)
        labels.append(lab)
    return cand_ids, labels


def load_mind_articles(train_dir, dev_dir):
    """Union of train + dev news.tsv, deduplicated by news_id, mapped to the
    unified article schema."""
    news = pd.concat([_load_news_raw(train_dir), _load_news_raw(dev_dir)], ignore_index=True)
    news = news.drop_duplicates(subset="news_id", keep="first")

    entities = (news["title_entities"].apply(_extract_entity_ids) if "title_entities" in news
                else pd.Series([[]] * len(news)))
    abstract_entities = news["abstract_entities"].apply(_extract_entity_ids)
    entities = [sorted(set(a) | set(b), key=(a + b).index) if (a or b) else []
                for a, b in zip(entities, abstract_entities)]

    out = pd.DataFrame({
        "dataset": "mind",
        "article_id": news["news_id"],
        "title": news["title"],
        "abstract": news["abstract"],
        "body": None,
        "category": news["category"],
        "subcategory": news["subcategory"],
        "entities": entities,
        "url": news["url"],
        "published_time": pd.NaT,
    })
    return out[ARTICLE_COLUMNS]


def load_mind_interactions(train_dir, dev_dir):
    """Concatenation of train + dev behaviors.tsv mapped to the unified
    interactions schema. `split` is left unset (filled in later by
    pipeline.split on the merged, time-sorted pool)."""
    frames = []
    for source, dir_path in [("train", train_dir), ("dev", dev_dir)]:
        beh = _load_behaviors_raw(dir_path)
        history = beh["history"].apply(_parse_history)
        cand_labels = beh["impressions"].apply(_parse_impressions)
        candidates = cand_labels.apply(lambda cl: cl[0])
        labels = cand_labels.apply(lambda cl: cl[1])

        frames.append(pd.DataFrame({
            "dataset": "mind",
            "impression_id": "mind_" + source + "_" + beh["impression_id"].astype(str),
            "user_id": beh["user_id"],
            "impression_time": beh["time"],
            "history_article_ids": history,
            "candidate_article_ids": candidates,
            "labels": labels,
            "split": None,
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[INTERACTION_COLUMNS]


def derive_mind_click_history(interactions):
    """MIND has no separately-timestamped click history, so the only
    timestamped clicks available are positive-labelled impression candidates:
    a click on a shown candidate has a known timestamp (the impression time).
    Returns long-format rows: dataset, user_id, article_id, click_time."""
    df = interactions[["user_id", "impression_time", "candidate_article_ids", "labels"]].copy()
    df = df.explode(["candidate_article_ids", "labels"])
    df = df[df["labels"] == 1]
    ch = df.rename(columns={"candidate_article_ids": "article_id", "impression_time": "click_time"})
    ch = ch[["user_id", "article_id", "click_time"]].reset_index(drop=True)
    ch.insert(0, "dataset", "mind")
    return ch
