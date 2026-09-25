#!/usr/bin/env python3
"""A2 Q3: trains the from-scratch NRMS model (retrieval/nrms.py) as a
neural comparison point against the LightGBM re-ranker
(scripts/train_reranker.py), then reports a THREE-way per-impression
comparison on val -- Assignment-1 fusion score / GBDT re-ranker / NRMS --
each pair with a paired bootstrap 95% CI, matching Q3.4's significance
requirement.

Dataset-agnostic (--dataset mind|ebnerd): NRMS only needs {article_id:
title} and the unified interactions schema both datasets already share
(pipeline/schema.py) -- unlike the GBDT's genuinely dataset-specific Q1
features. GPU-first but not GPU-only: --device auto picks CUDA if
available, else CPU, so a small `--train_sample_size --epochs 1` run is a
fast local correctness check before spending real GPU time (e.g. on an
ADA/SLURM job -- see scripts/train_nrms.sbatch) on a full run.

Reads:
    data/processed/<dataset>/{articles,interactions}.parquet   (A1 Q1)
    data/models/<dataset>/{bm25,semantic,reranker}.pkl + configs (A1 Q2/Q3, A2 Q2)
    data/feature_store/<dataset>/{candidate_features_val,retrieval_scores_val_*}.parquet (A2 Q1/Q2, cached)
Writes:
    data/models/<dataset>/nrms.pt + nrms_vocab.json + nrms_config.json
    data/reports/<dataset>_nrms_eval.json

    python scripts/train_nrms.py --dataset mind --train_sample_size 200000 --device auto
    python scripts/train_nrms.py --dataset ebnerd --train_sample_size 200000 --device auto

Training instance construction, matching the NRMS paper exactly: one
instance per POSITIVE click, paired with `--num_negatives` negatives
sampled (with replacement if the impression doesn't have enough) from
that SAME impression's own non-clicked candidates -- never from the full
corpus, consistent with this project's restricted-candidate-scoring
convention (score_candidates, not score_batch_full) everywhere else. An
impression with no positive or no negative candidates contributes no
training instance (nothing to contrast).

Evaluation scores every candidate in an impression (not just num_negatives+1
-- that would only be fair for training, not comparison against the GBDT/
fusion baselines, which are both scored on the impression's real,
full-size candidate list).
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bootstrap import bootstrap_ci_paired_delta
from retrieval.build_indices import load_indices
from retrieval.nrms import NRMS, PAD_IDX, build_vocab, pad_history, title_to_ids
from retrieval.ranking_metrics import auc_score, mrr_score, ndcg_at_k
from retrieval.reranker_data import FEATURE_COLUMNS, load_split_features
from retrieval.text_utils import article_text


class NRMSTrainDataset(Dataset):
    """One item per (impression, positive-click) pair -- see module
    docstring for the negative-sampling construction. Built eagerly in
    __init__ (not streamed) since even MIND's full train impression count
    fits comfortably as a list of (list[str], list[str]) id tuples; only
    __getitem__ does the (cheap, vocab-lookup-only) tensor conversion."""

    def __init__(self, split_interactions, title_ids_lookup, max_history_len, max_title_len,
                 num_negatives, seed=42):
        self.title_ids_lookup = title_ids_lookup
        self.max_history_len = max_history_len
        self.max_title_len = max_title_len
        self.empty_title = [PAD_IDX] * max_title_len

        rng = np.random.default_rng(seed)
        self.examples = []
        for imp in split_interactions.itertuples(index=False):
            cand = imp.candidate_article_ids
            labels = imp.labels
            if cand is None or len(cand) == 0 or labels is None or len(labels) != len(cand):
                continue
            pos_idx = [i for i in range(len(labels)) if labels[i] == 1]
            neg_idx = [i for i in range(len(labels)) if labels[i] == 0]
            if not pos_idx or not neg_idx:
                continue
            hist = imp.history_article_ids
            for pi in pos_idx:
                neg_choice = rng.choice(neg_idx, size=num_negatives, replace=len(neg_idx) < num_negatives)
                cand_ids = [cand[pi]] + [cand[j] for j in neg_choice]
                self.examples.append((hist, cand_ids))

    def __len__(self):
        return len(self.examples)

    def _title_ids(self, article_id):
        if article_id is None:
            return self.empty_title
        return self.title_ids_lookup.get(article_id, self.empty_title)

    def __getitem__(self, idx):
        hist, cand_ids = self.examples[idx]
        padded_hist, hist_mask = pad_history(hist, self.max_history_len)
        hist_title_ids = np.array([self._title_ids(a) for a in padded_hist], dtype=np.int64)
        cand_title_ids = np.array([self._title_ids(a) for a in cand_ids], dtype=np.int64)
        return hist_title_ids, np.array(hist_mask, dtype=bool), cand_title_ids


def _score_val_batch(model, batch, title_ids_lookup, max_history_len, max_title_len, device):
    """Scores every real candidate for each impression in `batch` (a list
    of interactions rows), padding candidate lists to the batch's own max
    length -- padded slots are dropped again right after scoring (sliced
    out per-row below), never fed into a metric. Returns
    [(impression_id, labels_array, scores_array), ...]."""
    empty_title = [PAD_IDX] * max_title_len

    def title_ids(a):
        if a is None:
            return empty_title
        return title_ids_lookup.get(a, empty_title)

    max_cand = max(len(imp.candidate_article_ids) for imp in batch)
    hist_batch, hist_mask_batch, cand_batch, cand_counts = [], [], [], []
    for imp in batch:
        padded_hist, hmask = pad_history(imp.history_article_ids, max_history_len)
        hist_batch.append([title_ids(a) for a in padded_hist])
        hist_mask_batch.append(hmask)
        cand = list(imp.candidate_article_ids)
        cand_counts.append(len(cand))
        row = [title_ids(a) for a in cand] + [empty_title] * (max_cand - len(cand))
        cand_batch.append(row)

    hist_t = torch.tensor(hist_batch, dtype=torch.long, device=device)
    hmask_t = torch.tensor(hist_mask_batch, dtype=torch.bool, device=device)
    cand_t = torch.tensor(cand_batch, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model(hist_t, hmask_t, cand_t).cpu().numpy()

    out = []
    for i, imp in enumerate(batch):
        n = cand_counts[i]
        labels = np.asarray(imp.labels[:n], dtype=float)
        out.append((imp.impression_id, labels, scores[i, :n]))
    return out


def evaluate_nrms_on_val(model, val_interactions, title_ids_lookup, max_history_len, max_title_len,
                           device, batch_size=32, eval_limit=None):
    """Returns {impression_id: {"auc":..., "mrr":..., "ndcg5":..., "ndcg10":...}}
    -- keyed by impression_id (not a positional array) so it can be
    aligned against the GBDT/fusion per-impression metrics below even if
    iteration order differs between the two data-loading paths."""
    model.eval()
    rows = val_interactions if eval_limit is None else val_interactions.head(eval_limit)
    out = {}
    batch = []
    n_done = 0
    t0 = time.time()
    for imp in rows.itertuples(index=False):
        if imp.candidate_article_ids is None or len(imp.candidate_article_ids) == 0:
            continue
        batch.append(imp)
        if len(batch) == batch_size:
            for imp_id, labels, scores in _score_val_batch(model, batch, title_ids_lookup, max_history_len,
                                                              max_title_len, device):
                out[imp_id] = {"auc": auc_score(labels, scores), "mrr": mrr_score(labels, scores),
                                "ndcg5": ndcg_at_k(labels, scores, 5), "ndcg10": ndcg_at_k(labels, scores, 10)}
            n_done += len(batch)
            batch = []
            if n_done % (batch_size * 200) == 0:
                print(f"    scored {n_done:,}/{len(rows):,} val impressions ({time.time()-t0:.1f}s)")
    if batch:
        for imp_id, labels, scores in _score_val_batch(model, batch, title_ids_lookup, max_history_len,
                                                          max_title_len, device):
            out[imp_id] = {"auc": auc_score(labels, scores), "mrr": mrr_score(labels, scores),
                            "ndcg5": ndcg_at_k(labels, scores, 5), "ndcg10": ndcg_at_k(labels, scores, 10)}
    return out


def per_impression_metrics_by_id(df, score_col, label_col="label", impression_col="impression_id"):
    """Same per-impression AUC/MRR/nDCG computation as
    retrieval.reranker_eval.per_impression_metrics, but keyed by
    impression_id (a dict) instead of returned as positional arrays -- so
    it can be intersected/aligned against evaluate_nrms_on_val's dict by
    key, regardless of row order in either DataFrame."""
    out = {}
    for imp_id, g in df.groupby(impression_col, sort=False, observed=True):
        labels = g[label_col].to_numpy()
        scores = g[score_col].to_numpy()
        out[imp_id] = {"auc": auc_score(labels, scores), "mrr": mrr_score(labels, scores),
                        "ndcg5": ndcg_at_k(labels, scores, 5), "ndcg10": ndcg_at_k(labels, scores, 10)}
    return out


def paired_delta_by_key(metrics_a, metrics_b, metric, n_boot=1000, seed=42):
    """(after=b) - (before=a), on the intersection of impression_ids both
    dicts have a value for, in one fixed shared key order -- the alignment
    bootstrap_ci_paired_delta needs, built explicitly here since `a`/`b`
    can come from two differently-ordered sources (NRMS's own eval loop
    vs. a DataFrame groupby)."""
    keys = sorted(set(metrics_a) & set(metrics_b))
    a = np.array([metrics_a[k][metric] for k in keys])
    b = np.array([metrics_b[k][metric] for k in keys])
    return bootstrap_ci_paired_delta(a, b, n_boot=n_boot, random_state=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_dir", default=None, help="defaults to data/models/<dataset>")
    ap.add_argument("--train_sample_size", type=int, default=200_000,
                     help="cap on TRAIN impressions before generating positive/negative instances "                          "(capped-subsample-first, see design note) -- a no-op if the split has fewer")
    ap.add_argument("--max_vocab_size", type=int, default=30_000)
    ap.add_argument("--max_title_len", type=int, default=20)
    ap.add_argument("--max_history_len", type=int, default=20, help="matches recent_n elsewhere in this project")
    ap.add_argument("--num_negatives", type=int, default=4, help="K in the paper's K-negative sampling")
    ap.add_argument("--embed_dim", type=int, default=100)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_limit", type=int, default=None, help="cap on val impressions scored (default: full val)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                           else args.device if args.device != "auto" else "cpu")
    print(f"Dataset={args.dataset}, device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model_dir = args.model_dir or os.path.join("data", "models", args.dataset)
    proc_dir = os.path.join(args.data_dir, "processed", args.dataset)

    print("Loading articles + interactions ...")
    articles = pd.read_parquet(os.path.join(proc_dir, "articles.parquet"))
    interactions = pd.read_parquet(os.path.join(proc_dir, "interactions.parquet"))
    train_interactions = interactions[interactions["split"] == "train"].reset_index(drop=True)
    val_interactions = interactions[interactions["split"] == "val"].reset_index(drop=True)

    if args.train_sample_size and args.train_sample_size < len(train_interactions):
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(train_interactions), size=args.train_sample_size, replace=False)
        train_interactions = train_interactions.iloc[keep].reset_index(drop=True)
    print(f"  train: {len(train_interactions):,} impressions (after sampling); "
          f"val: {len(val_interactions):,} impressions")

    # Title-only (see retrieval/nrms.py module docstring for why: shrinks
    # vocab vs. title+abstract, and keeps this a fair "text-only" neural
    # baseline rather than one that also sees abstract text the GBDT's own
    # features never look at).
    print(f"Building vocab (title-only, max_vocab_size={args.max_vocab_size}) ...")
    vocab = build_vocab(articles["title"].tolist(), max_vocab_size=args.max_vocab_size)
    print(f"  vocab size: {len(vocab):,}")
    title_ids_lookup = {
        aid: title_to_ids(title, vocab, args.max_title_len)
        for aid, title in zip(articles["article_id"], articles["title"])
    }

    print("Building training instances (one per positive click, "
          f"{args.num_negatives} sampled in-impression negatives each) ...")
    train_ds = NRMSTrainDataset(train_interactions, title_ids_lookup, args.max_history_len,
                                  args.max_title_len, args.num_negatives, seed=args.seed)
    print(f"  {len(train_ds):,} training instances")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, drop_last=False)

    model = NRMS(vocab_size=len(vocab), embed_dim=args.embed_dim, num_heads=args.num_heads,
                 dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NRMS model: {n_params:,} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining ({args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}) ...")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_seen = 0.0, 0
        for hist_ids, hist_mask, cand_ids in train_loader:
            hist_ids, hist_mask, cand_ids = hist_ids.to(device), hist_mask.to(device), cand_ids.to(device)
            scores = model(hist_ids, hist_mask, cand_ids)  # (batch, num_negatives+1)
            target = torch.zeros(scores.size(0), dtype=torch.long, device=device)  # positive always index 0
            loss = loss_fn(scores, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * scores.size(0)
            n_seen += scores.size(0)
        print(f"  epoch {epoch+1}/{args.epochs}: loss={total_loss/max(n_seen,1):.4f} ({time.time()-t0:.1f}s)")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "nrms.pt"))
    with open(os.path.join(model_dir, "nrms_vocab.json"), "w") as f:
        json.dump(vocab, f)
    nrms_config = {
        "dataset": args.dataset, "max_vocab_size": args.max_vocab_size, "max_title_len": args.max_title_len,
        "max_history_len": args.max_history_len, "num_negatives": args.num_negatives,
        "embed_dim": args.embed_dim, "num_heads": args.num_heads, "dropout": args.dropout,
        "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
        "train_sample_size": args.train_sample_size, "n_train_instances": len(train_ds), "n_params": n_params,
    }
    with open(os.path.join(model_dir, "nrms_config.json"), "w") as f:
        json.dump(nrms_config, f, indent=2)
    print(f"Wrote {model_dir}/nrms.pt, nrms_vocab.json, nrms_config.json")

    print(f"\nEvaluating on val ({'full' if args.eval_limit is None else args.eval_limit} impressions, "
          f"scoring every real candidate per impression) ...")
    nrms_metrics = evaluate_nrms_on_val(model, val_interactions, title_ids_lookup, args.max_history_len,
                                          args.max_title_len, device, args.eval_batch_size, args.eval_limit)
    print(f"  scored {len(nrms_metrics):,} val impressions")

    print("Loading fusion score + trained GBDT re-ranker score for the same val impressions "
          "(reuses cached Assignment-1/2 data if present) ...")
    # reranker.pkl (LightGBM) loaded BEFORE load_indices (which can
    # construct a FAISS index, if this run used --semantic_backend sbert):
    # unpickling LightGBM AFTER FAISS is already live in the same process
    # segfaults reliably on this project's dev machine -- see
    # scripts/generate_predictions.py's module docstring for the
    # bisection that found this. Every other A2 script loads in this
    # order for the same reason.
    with open(os.path.join(model_dir, "reranker.pkl"), "rb") as f:
        reranker = pickle.load(f)
    bm25, semantic, idx_config = load_indices(model_dir)
    recent_n, recency_decay = idx_config["recent_n"], idx_config["recency_decay"]
    text_lookup = dict(zip(articles["article_id"],
                            (article_text(t, a) for t, a in zip(articles["title"], articles["abstract"]))))
    entity_lookup = dict(zip(articles["article_id"], articles["entities"]))
    val_df = load_split_features(args.dataset, "val", args.data_dir, bm25, semantic, text_lookup, entity_lookup,
                                   recent_n, recency_decay, 0.7, feature_cols=FEATURE_COLUMNS[args.dataset])
    val_df["reranker_score"] = reranker.predict(val_df[FEATURE_COLUMNS[args.dataset]])

    fusion_metrics = per_impression_metrics_by_id(val_df, "fusion_score")
    reranker_metrics = per_impression_metrics_by_id(val_df, "reranker_score")

    def overall(metrics_dict, metric):
        vals = [v[metric] for v in metrics_dict.values() if not np.isnan(v[metric])]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n=== {args.dataset}: fusion (Q3 baseline) vs. GBDT re-ranker vs. NRMS ===")
    print(f"{'metric':<8}{'fusion':>10}{'GBDT':>10}{'NRMS':>10}")
    summary = {}
    for m in ["auc", "mrr", "ndcg5", "ndcg10"]:
        f_v, r_v, n_v = overall(fusion_metrics, m), overall(reranker_metrics, m), overall(nrms_metrics, m)
        print(f"{m:<8}{f_v:>10.4f}{r_v:>10.4f}{n_v:>10.4f}")
        summary[m] = {"fusion": f_v, "gbdt_reranker": r_v, "nrms": n_v}

    print("\n=== paired bootstrap 95% CI: NRMS vs. fusion, and NRMS vs. GBDT re-ranker ===")
    paired = {"nrms_vs_fusion": {}, "nrms_vs_gbdt": {}}
    for m in ["auc", "mrr", "ndcg5", "ndcg10"]:
        d_fusion = paired_delta_by_key(fusion_metrics, nrms_metrics, m, args.n_boot, args.seed)
        d_gbdt = paired_delta_by_key(reranker_metrics, nrms_metrics, m, args.n_boot, args.seed)
        paired["nrms_vs_fusion"][m] = d_fusion
        paired["nrms_vs_gbdt"][m] = d_gbdt
        print(f"  {m}: vs fusion delta={d_fusion['delta']:+.4f} CI=[{d_fusion['ci_low']:+.4f},"
              f"{d_fusion['ci_high']:+.4f}] excludes_zero={d_fusion['excludes_zero']}  |  "
              f"vs GBDT delta={d_gbdt['delta']:+.4f} CI=[{d_gbdt['ci_low']:+.4f},{d_gbdt['ci_high']:+.4f}] "
              f"excludes_zero={d_gbdt['excludes_zero']}")

    report = {
        "dataset": args.dataset, "config": nrms_config, "n_val_impressions_scored": len(nrms_metrics),
        "overall": summary, "paired_bootstrap_vs_nrms": paired,
    }
    report_path = os.path.join(args.data_dir, "reports", f"{args.dataset}_nrms_eval.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
