#!/usr/bin/env python3
"""Q5: generate MIND Codabench prediction files from BM25 and LSA scorers.

For each MINDlarge_test impression, ranks *only that impression's own
candidate list* (not full-corpus retrieval -- Q2/Q3's top-K retrieval is a
separate diagnostic, see scripts/evaluate_retrieval.py) using the user's
recent click history as the query, and writes one line per the official
format (mind_submission_guidelines.txt):

    ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]

Streams data/MINDlarge_test/behaviors.tsv in chunks (never materializes all
2.37M rows at once) since the assignment explicitly calls out memory
efficiency for the large test sets.

    python scripts/generate_predictions.py --method bm25
    python scripts/generate_predictions.py --method lsa
    python scripts/generate_predictions.py --method both   # default

Writes data/predictions/mind/<method>/prediction.txt and prediction.zip
(zip contains only prediction.txt at its root, per submission guidelines).
"""

import argparse
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import mind as mind_loader
from retrieval.bm25 import BM25Index
from retrieval.lsa import LSAIndex, mean_pool_user_vector
from retrieval.text_utils import article_text, tokenize

BEH_COLS = mind_loader.BEH_COLS


def ranks_from_scores(scores):
    """1-indexed ranks, 1 = highest score. Ties broken deterministically by
    original candidate order (stable sort), so every candidate gets a
    distinct rank -- required by the submission format."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def build_combined_lookups(data_dir):
    """Title-token lookup spans train+val+test (a test impression's history
    can reference articles clicked during the train/val period), keyed by
    article_id -> pre-tokenized title. Built once, reused across the whole
    streaming pass."""
    train_val_articles = pd.read_parquet(os.path.join(data_dir, "processed", "mind", "articles.parquet"))
    test_news = mind_loader.load_news_raw(os.path.join(data_dir, "MINDlarge_test"))
    test_articles = mind_loader.articles_from_news_df(test_news)

    combined = pd.concat([train_val_articles[["article_id", "title", "abstract"]],
                           test_articles[["article_id", "title", "abstract"]]], ignore_index=True)
    combined = combined.drop_duplicates(subset="article_id", keep="first").reset_index(drop=True)

    title_tokens = {aid: tokenize(t) for aid, t in zip(combined["article_id"], combined["title"])}
    texts = [article_text(t, a) for t, a in zip(combined["title"], combined["abstract"])]
    return combined["article_id"].tolist(), texts, title_tokens, test_articles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--method", choices=["bm25", "lsa", "both"], default="both")
    ap.add_argument("--chunk_size", type=int, default=20_000)
    ap.add_argument("--recent_n", type=int, default=20)
    ap.add_argument("--lsa_components", type=int, default=128)
    args = ap.parse_args()
    methods = ["bm25", "lsa"] if args.method == "both" else [args.method]

    print("Building combined article/title lookup (train+val+test) ...")
    combined_ids, combined_texts, title_tokens, test_articles = build_combined_lookups(args.data_dir)
    print(f"  {len(combined_ids):,} unique articles in combined lookup")

    test_doc_ids = test_articles["article_id"].tolist()
    test_texts = [article_text(t, a) for t, a in zip(test_articles["title"], test_articles["abstract"])]

    bm25 = lsa = None
    history_embeddings = None
    if "bm25" in methods:
        t0 = time.time()
        bm25 = BM25Index().fit(test_doc_ids, test_texts)
        print(f"BM25 index over MINDlarge_test corpus: {len(test_doc_ids):,} docs, "
              f"vocab={bm25.bm25_matrix.shape[1]:,} ({time.time()-t0:.1f}s)")
    if "lsa" in methods:
        t0 = time.time()
        lsa = LSAIndex(n_components=args.lsa_components).fit(test_doc_ids, test_texts)
        print(f"LSA index over MINDlarge_test corpus: {len(test_doc_ids):,} docs, "
              f"{args.lsa_components}-dim ({time.time()-t0:.1f}s)")
        t0 = time.time()
        emb = lsa.embed(combined_texts)  # project the *combined* universe into this fit space
        history_embeddings = dict(zip(combined_ids, emb))
        print(f"  projected {len(combined_ids):,} combined articles for history lookup ({time.time()-t0:.1f}s)")

    out_paths = {}
    fhandles = {}
    for m in methods:
        d = os.path.join(args.data_dir, "predictions", "mind", m)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "prediction.txt")
        out_paths[m] = p
        fhandles[m] = open(p, "w")

    beh_path = os.path.join(args.data_dir, "MINDlarge_test", "behaviors.tsv")
    print(f"\nStreaming {beh_path} in chunks of {args.chunk_size:,} ...")
    n_written = 0
    t_start = time.time()
    reader = pd.read_csv(beh_path, sep="\t", header=None, names=BEH_COLS, quoting=3,
                          na_values=[""], keep_default_na=True, chunksize=args.chunk_size)

    for chunk in reader:
        imp_ids = chunk["impression_id"].tolist()
        histories = [mind_loader._parse_history(h) for h in chunk["history"].tolist()]
        cand_labels = [mind_loader._parse_impressions(i) for i in chunk["impressions"].tolist()]

        for imp_id, hist, (cand_ids, _labels) in zip(imp_ids, histories, cand_labels):
            recent = hist[-args.recent_n:] if hist else []

            if bm25 is not None:
                q_tokens = []
                for a in recent:
                    q_tokens.extend(title_tokens.get(a, []))
                scores = bm25.score_candidates(q_tokens, cand_ids)
                ranks = ranks_from_scores(scores)
                fhandles["bm25"].write(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]\n")

            if lsa is not None:
                embs = [history_embeddings.get(a) for a in recent]
                user_vec = mean_pool_user_vector(embs)
                scores = lsa.score_candidates(user_vec, cand_ids)
                ranks = ranks_from_scores(scores)
                fhandles["lsa"].write(f"{imp_id} [{','.join(map(str, ranks.tolist()))}]\n")

        n_written += len(chunk)
        print(f"  {n_written:,} impressions written ({time.time()-t_start:.1f}s elapsed)")

    for f in fhandles.values():
        f.close()

    print(f"\nDone. {n_written:,} predictions written.")
    for m, p in out_paths.items():
        zip_path = os.path.join(os.path.dirname(p), "prediction.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(p, arcname="prediction.txt")
        size_mb = os.path.getsize(zip_path) / 1e6
        print(f"  {m}: {p}  ->  {zip_path} ({size_mb:.1f} MB)")

    print("\nUpload the zip(s) to https://www.codabench.org/competitions/13967/")


if __name__ == "__main__":
    main()
