# News Recommender — MIND & EB-NeRD

[Github Repository Link](https://github.com/TECHIE-TITAN/IRE-Assignment-News-Recommender)

A reproducible
retrieval pipeline for news recommendation, built against both the
**MIND** and **EB-NeRD** datasets. Covers a unified data pipeline with
leakage-guarded temporal splits, lexical (BM25F) and semantic (LSA /
sentence-transformer + FAISS) candidate scoring, score-level fusion, an
offline evaluation harness (AUC/MRR/nDCG, diversity/novelty/coverage,
bootstrap CIs), and leaderboard prediction-file generation for both
datasets.

## Setup

```bash
git clone <GITHUB_REPO_URL>
cd IRE-Assignment-News-Recommender

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The sentence-transformer semantic backend (`--semantic_backend sbert`) is
optional and pulls in `sentence-transformers` + `faiss-cpu` (~1GB with
`torch`) — install it only if you plan to use that backend:

```bash
pip install sentence-transformers faiss-cpu
```

## Data

**MIND** — [HuggingFace: yjw1029/MIND](https://huggingface.co/datasets/yjw1029/MIND)

```bash
hf download yjw1029/MIND --repo-type dataset --local-dir data
# or individually: MINDsmall_train.zip, MINDsmall_dev.zip (dev),
# MINDlarge_train.zip, MINDlarge_dev.zip (full-scale training),
# MINDlarge_test.zip (required for leaderboard submission)
```

**EB-NeRD** — [S3: ebnerd-dataset](https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/)

```bash
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip     # quick iteration
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip    # final training
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip           # full-scale training
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/articles_large_only.zip
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip  # required for leaderboard submission
```

## Quickstart

Each stage reads only what an earlier stage wrote under `data/`, so
stages can be re-run independently once the pipeline has been built once.

**MIND:**

```bash
python build_pipeline.py --dataset mind                                   # unify + split + feature store
python scripts/build_indices.py --dataset mind --semantic_backend sbert   # fit + persist BM25F + semantic index
python scripts/tune_bm25.py --dataset mind                                # optional: k1/b grid search
python scripts/evaluate_retrieval.py --dataset mind                       # full-corpus recall@K diagnostic
python scripts/evaluate_ranking.py --dataset mind                        # restricted-candidate eval (AUC/MRR/nDCG/...)
python scripts/generate_predictions.py --method all                      # bm25 + semantic + fusion prediction files
```

**EB-NeRD:**

```bash
python build_pipeline.py --dataset ebnerd
python scripts/build_indices.py --dataset ebnerd --semantic_backend sbert
python scripts/tune_bm25.py --dataset ebnerd
python scripts/evaluate_retrieval.py --dataset ebnerd
python scripts/evaluate_ranking.py --dataset ebnerd
python scripts/generate_predictions_ebnerd.py --method both
```

- Swap `--semantic_backend lsa` for the dependency-free TF-IDF+SVD backend
instead of sentence-transformers. 
- `scripts/build_indices.py` also accepts `--bm25_k1`/`--bm25_b`/`--bm25_title_weight`/`--bm25_abstract_weight`/`--bm25_entity_boost`/`--entity_weight`/`--no_entity_fusion`/`--recent_n`/`--recency_decay`
- Everything downstream reads these back out of the persisted `data/models/<dataset>/config.json`, so they only need to be set once, at build time. 
- `generate_predictions.py`/`generate_predictions_ebnerd.py`
override any of them via matching flags if you need to score with
different hyperparameters than what's persisted.

Approximate runtimes (MIND, large scale): `build_pipeline.py` ~1–2 min,
`evaluate_retrieval.py` ~20 min (full-corpus scoring), `evaluate_ranking.py`
~1–2 min, `generate_predictions.py` ~10 min (2.37M impressions,
multiprocessed). EB-NeRD (`ebnerd_small`) is faster throughout — a
~20K-article, ~250K-impression corpus.

## Assignment 2 — feature engineering, re-ranking, evaluation

Builds a supervised LightGBM `LGBMRanker` re-ranker on top of Assignment
1's retrieval scores, adding recency-weighted category/embedding-similarity
behavioural features, article popularity/freshness, and (EB-NeRD only)
session/engagement history. Reads only what Assignment 1's scripts already
wrote (`data/processed/`, `data/models/<dataset>/{bm25,semantic}.pkl`) —
run the Assignment-1 quickstart above first.

**MIND:**

```bash
python scripts/build_features.py --dataset mind --split val             # Q1 behavioural features
python scripts/build_features.py --dataset mind --split train           # ~2-2.5hrs on full MINDlarge_train
python scripts/train_reranker.py --dataset mind                         # trains LGBMRanker, reports before/after
python scripts/evaluate_reranker.py --dataset mind                      # cold/warm + head/tail slices, bootstrap CIs
python scripts/ablation_reranker.py --dataset mind                      # which feature groups actually help
python scripts/benchmark_serving.py --dataset mind                      # index memory + p99 latency + cost/QPS
python scripts/generate_predictions.py --method reranker                # Codabench prediction file, reranker-scored
```

**EB-NeRD:**

```bash
python scripts/build_features.py --dataset ebnerd --split val
python scripts/build_features.py --dataset ebnerd --split train
python scripts/train_reranker.py --dataset ebnerd
python scripts/evaluate_reranker.py --dataset ebnerd
python scripts/ablation_reranker.py --dataset ebnerd
python scripts/benchmark_serving.py --dataset ebnerd
python scripts/generate_predictions_ebnerd.py --method reranker
```

Notes:
- `train_reranker.py`/`ablation_reranker.py` accept `--train_sample_size`
  (default unset for `train_reranker.py`, 200,000 for `ablation_reranker.py`
  since it trains several models) to keep training tractable at MIND's
  83.5M-candidate-row train scale; val is always scored in full.
- `ablation_reranker.py` trains `full` / `retrieval_only` (Assignment-1
  scores alone, no Q1 features) / each `no_<feature-group>` variant, and
  reports a paired-bootstrap-CI drop vs. `full` for each — see
  `data/reports/<dataset>_ablation.json`.
- `benchmark_serving.py` measures real process RSS (not pickle file size)
  incrementally as each index/model loads, and per-request p99 latency for
  the full candidate-generation + re-ranking path, timed one impression at
  a time — see `data/reports/<dataset>_serving_benchmark.json`.
- On this project's dev machine (Apple M2/arm64), unpickling a LightGBM
  model AFTER a FAISS-backed semantic index is already constructed in the
  same process segfaults reliably — every A2 script above loads
  `reranker.pkl` BEFORE the semantic index for this reason; see
  `generate_predictions.py`'s module docstring if adapting this order
  elsewhere (including inside `multiprocessing.Pool` worker `initargs`,
  which unpickle as one ordered tuple).

## Project structure

```
.
├── README.md                            — this file: setup, run, structure
├── requirements.txt                      — pinned deps (core + optional sentence-transformers/faiss)
├── .gitignore                            — excludes raw dataset dirs/zips, data/processed/, venv, caches
├── design_note.md                        — chronological changelog of design/strategy revisions
├── design_note.pdf                       — compiled ≤4-page design note (choices, alternatives, observations, 10x scale)
├── analyse.md                            — cross-cutting analysis of the 3 strategies (metrics, engineering, tool choices)
├── mind_analysis.ipynb                   — exploratory MIND data analysis (not wired into the pipeline)
├── ebnerd_analysis.ipynb                 — exploratory EB-NeRD data analysis (not wired into the pipeline)
├── Screenshot ... 7.10.53 PM.png         — MIND leaderboard row: plain BM25+LSA submission, score 0.5805
├── Screenshot ... 7.11.21 PM.png         — MIND leaderboard row: BM25F+tuned+SBERT submission, score 0.5872
├── Screenshot ... 7.11.38 PM.png         — MIND leaderboard row: entity-boost+fusion submission, score 0.5953
├── Screenshot ... 7.17.21 PM.png         — EB-NeRD submission status (predictions.zip submitted, not yet scored)
├── build_pipeline.py                     — entry point: raw MIND/EB-NeRD -> unified schema + feature store
│
├── pipeline/
│   ├── __init__.py                       — package marker
│   ├── schema.py                         — shared column contract (article/interaction/click-history schemas)
│   ├── mind.py                           — MIND TSV loaders/parsers -> unified schema
│   ├── ebnerd.py                         — EB-NeRD parquet loaders/parsers -> unified schema
│   ├── split.py                          — temporal split + no-future-click-leakage assertions
│   ├── feature_store.py                  — article/user feature builders (train-only CTR, leakage-safe history)
│   ├── features_common.py                — A2 Q1: shared dataset-agnostic feature math (recency weights, freshness gate)
│   ├── features_mind.py                  — A2 Q1: MIND candidate-level behavioural feature set (8 features)
│   └── features_ebnerd.py                — A2 Q1: EB-NeRD candidate-level feature set (10, incl. session/engagement)
│
├── retrieval/
│   ├── __init__.py                       — package marker
│   ├── text_utils.py                     — tokenization, query text/entity construction, recency weighting
│   ├── bm25.py                           — BM25F index: sparse field-weighted matrix + entity-overlap boost
│   ├── lsa.py                            — TF-IDF + TruncatedSVD semantic index (+ entity fusion)
│   ├── sbert.py                          — sentence-transformer + FAISS semantic index
│   ├── entity_embeddings.py              — MIND TransE entity-vector loading + fusion into content embeddings
│   ├── fusion.py                         — per-impression score normalization + weighted lexical/semantic fusion
│   ├── build_indices.py                  — fit_indices/save_indices/load_indices: shared index persistence
│   ├── ranking_metrics.py                — AUC/MRR/nDCG@k, per-impression then averaged
│   ├── beyond_accuracy.py                — intra-list diversity, novelty, train-popularity helper
│   ├── eval_utils.py                     — recall@K helpers for full-corpus top-k retrieval results
│   ├── bootstrap.py                      — bootstrap 95% CIs (plain mean, set-union coverage, A2 paired delta)
│   ├── reranker_scores.py                — A2 Q2: per-impression Assignment-1 retrieval scores as reranker features
│   ├── reranker_data.py                  — A2 Q2: FEATURE_COLUMNS + chunked/cached feature-table loading
│   └── reranker_eval.py                  — A2 Q2: shared per-impression AUC/MRR/nDCG evaluation helpers
│
├── scripts/
│   ├── build_indices.py                  — CLI: fits + persists a BM25F/semantic index once per corpus
│   ├── tune_bm25.py                      — k1/b grid search via restricted-candidate scoring on a subsample
│   ├── evaluate_retrieval.py             — full-corpus top-K recall diagnostic (lexical vs. semantic)
│   ├── evaluate_ranking.py               — restricted-candidate reranking eval (AUC/MRR/nDCG/diversity/...)
│   ├── generate_predictions.py           — MIND prediction file generation (bm25/semantic/fusion/reranker/all), multiprocessed
│   ├── generate_predictions_ebnerd.py    — EB-NeRD prediction file generation (bm25/semantic/reranker/both)
│   ├── build_features.py                 — A2 Q1: builds the candidate-level behavioural feature table
│   ├── train_reranker.py                 — A2 Q2: trains LGBMRanker, reports before/after AUC/MRR/nDCG
│   ├── evaluate_reranker.py              — A2 Q2/Q4: extended eval (cold/warm, head/tail, bootstrap CIs)
│   ├── ablation_reranker.py              — A2 Q3: feature-group ablation study with paired-bootstrap significance
│   └── benchmark_serving.py              — A2 Q4: index memory + single-request p99 latency + cost/QPS estimate
│
└── data/                                 — mostly gitignored (see Data, above); tracked contents:
```

See `design_note.md` for the chronological record of what changed and why
across each strategy revision.
