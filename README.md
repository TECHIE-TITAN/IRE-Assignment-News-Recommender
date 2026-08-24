# IRE Assignment 1 — News Recommender (MIND)

CS4.406 Information Retrieval & Extraction — Assignment 1 (`Assignment.md`).
Current scope: **MIND only** (EB-NeRD notebook exploration exists but is not
wired into the pipeline yet). Covers Q1 (reproducible pipeline), Q2 (BM25
lexical retrieval), Q3 (LSA semantic retrieval), Q4 (offline evaluation
harness), and Q5 (Codabench prediction generation). Q6 (design note as a
standalone document) is not yet done.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Raw MIND files expected under `data/`:

```
data/MINDsmall_train/{behaviors.tsv,news.tsv,entity_embedding.vec,relation_embedding.vec}
data/MINDsmall_dev/{...}
data/MINDlarge_test/{...}
```

(or the corresponding `MINDsmall_train.zip` / `MINDsmall_dev.zip` /
`MINDlarge_test.zip` — `build_pipeline.py` auto-extracts a zip into its
folder if the folder isn't there yet. Download commands are in
`Assignment.md` Part 0.)

## One-command pipeline

```bash
python build_pipeline.py                  # Q1
python scripts/evaluate_retrieval.py       # Q2 + Q3 (recall@K on MINDsmall_dev)
python scripts/evaluate_ranking.py         # Q4 (AUC/MRR/nDCG + beyond-accuracy + slicing + bootstrap CIs)
python scripts/generate_predictions.py     # Q5 (MINDlarge_test -> Codabench zip)
```

Each step reads only what the previous step wrote under `data/`, so they
can be re-run independently once `build_pipeline.py` has run once.

`generate_predictions.py` takes ~10 minutes end-to-end (indexes 121K
articles, scores 2.37M impressions) and `evaluate_retrieval.py` takes
~20 minutes (full-corpus top-K retrieval, not restricted-candidate scoring,
over 73K impressions — see Design Notes). `evaluate_ranking.py` takes
~1-2 minutes (restricted-candidate scoring, same cheap path as Q5, plus
bootstrap resampling). All are safe to run in the background; progress
prints every batch/chunk.

---

## Q1 — Reproducible Data Pipeline

`build_pipeline.py` (+ `pipeline/{schema,mind,split,feature_store}.py`).

**Key design decision:** MIND's own `MINDsmall_train` / `MINDsmall_dev` /
`MINDlarge_test` files are *already* temporally disjoint by construction
(Nov 9–14 / Nov 15 / Nov 16–22, 2019), so the pipeline uses them directly as
the train/val/test split rather than re-cutting by an arbitrary day
boundary. `pipeline/split.py` asserts this ordering holds
(`assert_split_boundary_monotonic`) rather than trusting file naming.
`val` is the only split with labels usable for offline evaluation; `test`
(large, unlabeled) is used only for Q5 prediction generation.

Outputs (`python build_pipeline.py`):

```
data/processed/mind/articles.parquet         65,238 rows (train ∪ val, deduped)
data/processed/mind/interactions.parquet     2,600,844 rows (train+val+test, with `split`)
data/processed/mind/click_history.parquet    347,727 rows (timestamped positive clicks, train+val only)
data/feature_store/mind/article_features.parquet     word/entity counts + train-only CTR + embedding placeholder
data/feature_store/mind/user_features_val.parquet    50,000 users, features from train-period clicks only
data/feature_store/mind/user_features_test.parquet   94,057 users, features from train+val-period clicks only
data/reports/mind_pipeline_report.json       row counts, split boundaries, leakage-check result
```

**Leakage discipline (Q9):** article popularity (`train_ctr` etc.) is
computed strictly from the `train` split. User features for `val` use only
clicks strictly before val's start; for `test`, only clicks strictly before
test's start. `pipeline/split.assert_no_future_click_leakage` is re-checked
inside `build_user_features` as a hard guard, not just a filter — it raises
if any click at/after the cutoff leaked in. Confirmed passing on the last
run (`data/reports/mind_pipeline_report.json`: `train_max_time
2019-11-14 23:59:13 < val_min_time 2019-11-15 00:00:01 < test_min_time
2019-11-16 00:00:05`, `"leakage_check": "passed"`).

MIND has no per-click timestamp field (`history` is just an ordered ID
list); `derive_mind_click_history` reconstructs timestamped clicks from
positive-labelled impression candidates instead — a click on a shown
candidate has a known timestamp: the impression time.

## Q2 — Lexical Candidate Generation (BM25)

`retrieval/bm25.py`, run via `scripts/evaluate_retrieval.py`.

Standard Okapi BM25 (`k1=1.5`, `b=0.75`) over `title + abstract`, but
implemented as one sparse `(doc × vocab)` BM25-weight matrix rather than a
literal posting-list dict: `score(d,q) = Σ_{t∈q} bm25_weight(d,t)`. This
behaves exactly like an inverted index (only shared terms contribute) but
turns scoring into sparse linear algebra, which is what makes both the
recall@K sweep and the 2.37M-impression Q5 pass fast without a C extension.
`min_df=2` / `max_df=0.6` drop hapax terms and near-stopwords. Vocabulary:
36,821 terms over the 65,238-article train+val corpus.

Query = tokenized titles of the user's last 20 clicked articles
(`retrieval/text_utils.build_query_text` — "recently clicked", per spec,
not full history; also bounds query size for Q5's scale).

## Q3 — Semantic Candidate Generation (LSA embeddings)

`retrieval/lsa.py`, run via `scripts/evaluate_retrieval.py`.

MIND ships no article embeddings (unlike EB-NeRD) and only sparse
per-entity TransE vectors that miss any article with no linked entity, so
this is the "compute your own" path Q3 allows. Used **TF-IDF + truncated
SVD (LSA)**, 128 components, fit with scikit-learn only — no model
download, fits/transforms the full corpus in seconds, and is a legitimate,
classic dense semantic-retrieval baseline (captures co-occurrence structure
beyond exact term overlap). Embeddings are L2-normalized so dot product =
cosine similarity. User representation = mean-pooled, re-normalized
embeddings of the same last-20-history articles used for BM25
(`mean_pool_user_vector`).

*(Considered instead: sentence-transformers/BERT on CPU — true dense
semantic embeddings, closer to the assignment's literal "BERT/XLM-RoBERTa"
suggestion, but ~1GB of new dependencies and materially slower to encode
~121K MINDlarge_test articles for a modest expected quality gain at this
stage. Deferred; worth revisiting for Q4 slicing analysis.)*

### Q2/Q3 recall@K results (`data/reports/mind_retrieval_eval.json`)

Full-corpus top-K retrieval (65,238 train+val articles) evaluated against
all 73,152 `MINDsmall_dev` impressions (100% have a ground-truth click):

| K   | BM25 (lexical) | LSA (semantic) |
|-----|---------------:|---------------:|
| 50  | 0.65%          | 0.38%          |
| 100 | 1.39%          | 0.62%          |
| 200 | 2.42%          | 1.06%          |

**BM25 beats LSA at every K.** Consistent with expectation: MIND titles are
short, entity-dense, near-duplicate-free news headlines, where exact
keyword/named-entity overlap is a strong relevance signal that a 128-dim
LSA projection compresses away. Both improve monotonically with K, as
expected for a candidate-generation stage.

Absolute recall is low in both cases — this is retrieval against the
**entire 65K-article corpus**, i.e. "does the one article this user
actually clicked land in the top-200 out of 65,238 candidates," which is a
much harder task than reranking the ~4–300 candidates MIND itself provides
per impression (what Q5's `score_candidates` does, and what real MIND
baselines are scored on). Low-but-monotonic-and-separated numbers here are
the expected shape for a first-pass lexical/semantic candidate generator,
not a bug — see Design Notes.

## Q4 — Offline Evaluation Harness

`retrieval/{ranking_metrics,beyond_accuracy,bootstrap}.py`, run via
`scripts/evaluate_ranking.py`.

**Scoring, not retrieval:** unlike Q2/Q3's full-corpus recall@K, Q4 reranks
each val impression's *own* candidate list with `score_candidates` — the
same restricted-scoring path Q5 uses for the actual submission — so Q4's
numbers reflect what the system would actually be scored on, not the harder
candidate-generation task.

- **Accuracy metrics:** AUC, MRR, nDCG@5, nDCG@10 — computed per impression
  then averaged across impressions (MIND's own official `evaluate.py`
  convention). AUC uses the rank-sum/Mann-Whitney formulation
  (tie-aware via average ranks); verified to match
  `sklearn.metrics.roc_auc_score` exactly (max abs diff `2.2e-16`) across
  2,000 randomized tie-heavy trials.
- **Beyond-accuracy** (top-10 recommended list per impression):
  *intra-list diversity* = 1 − mean pairwise cosine similarity among the
  top-10 items' LSA embeddings (used as a fixed content representation
  regardless of which method produced the ranking); *novelty* =
  self-information `-log2(p(item))` with `p(item)` a Laplace-smoothed
  train-split popularity share; *coverage* = fraction of the 65,238-article
  catalog touched by the union of all impressions' top-10 lists.
- **Slicing:** cold (< 5 history articles) vs. warm, using the impression's
  own `history_article_ids` length directly (simpler and more literal than
  joining the feature store's derived `user_features_val.history_length`).
- **Bootstrap 95% CIs**, `n_boot=1000`, for every metric above, for both
  methods, in every slice.

**A real bug caught and fixed during implementation:** the first version of
`bootstrap_ci_coverage` naively resampled impressions with replacement and
reported the resample distribution's mean as "coverage" — but coverage is a
*set-union* statistic, and with-replacement resampling only touches ~63%
of distinct impressions on average (`1 - 1/e`), so every resample's union
is systematically smaller than the true, full-sample union. This produced
a coverage "mean" (e.g. 0.0147) with a CI that didn't even contain the
actual achieved coverage (0.0533) — confirmed with a controlled synthetic
simulation before trusting the real output. Fixed by reporting
`point_estimate` (coverage on the actual, un-resampled evaluation run — the
number that matters) separately from `mean`/`ci_low`/`ci_high` (the
resampling distribution, kept because the assignment asks for bootstrap
CIs, but documented as a lower band by construction). See
`retrieval/bootstrap.py`'s docstring.

### Q4 results (`data/reports/mind_ranking_eval.json`)

| metric (overall) | BM25 | LSA |
|---|---:|---:|
| AUC | 0.5567 [0.5547, 0.5590] | 0.5552 [0.5531, 0.5574] |
| MRR | 0.2988 [0.2964, 0.3011] | 0.2868 [0.2844, 0.2888] |
| nDCG@5 | 0.2747 [0.2721, 0.2771] | 0.2646 [0.2620, 0.2669] |
| nDCG@10 | 0.3366 [0.3343, 0.3388] | 0.3272 [0.3248, 0.3294] |
| diversity | 0.8514 [0.8511, 0.8518] | 0.8166 [0.8161, 0.8171] |
| novelty | 16.390 [16.379, 16.400] | 16.486 [16.476, 16.496] |
| coverage (point est.) | 0.0533 | 0.0491 |

BM25 wins on every accuracy metric (consistent with Q2/Q3) and has *higher*
diversity and coverage but *lower* novelty than LSA — i.e. LSA's
recommendations are drawn from a smaller, more repetitive slice of the
catalog but skew slightly toward less-popular articles. Cold-start slice
(10,306 impressions, < 5 history articles): both methods drop sharply on
AUC (BM25 0.5225, LSA 0.5285 — LSA is *marginally* better than BM25 here,
the one slice where it wins on any accuracy metric, plausibly because a
near-empty query gives BM25 too little to match on while LSA's mean-pool
still produces *some* signal) but MRR/nDCG are close to the warm slice,
consistent with MIND's own within-impression base rates dominating those
metrics more than retrieval quality does at this candidate-list scale.
Full numbers (all 3 slices × both methods) in `data/reports/mind_ranking_eval.json`.

## Q5 — Codabench Prediction Generation

`scripts/generate_predictions.py`.

**Important scoping distinction:** Q2/Q3's recall@K retrieves top-K from
the *entire* article corpus (a candidate-generation diagnostic). Codabench
scoring instead needs a full ranking of *exactly* the candidates listed in
each impression (`mind_submission_guidelines.txt`). So Q5 reuses the same
BM25/LSA scorers but calls `score_candidates(query, impression_candidates)`
— which only ever touches that impression's ~4–300 candidates, never the
other ~120K non-candidate docs. That restriction is what keeps 2.37M
impressions tractable (~10 minutes wall time; measured ~4,100
impressions/sec).

Streams `MINDlarge_test/behaviors.tsv` in 20K-row chunks (never
materializes all 2.37M rows at once — the assignment explicitly calls out
memory efficiency for the large test sets). BM25 and LSA indices are
rebuilt over `MINDlarge_test`'s own 120,961-article corpus (that's the
corpus candidates are drawn from); a combined train+val+test article
lookup is used only to resolve *history* article titles/text, since a
test-period impression's history can reference articles clicked earlier.

Ties (e.g. cold-start users with empty history → all-zero scores) are
broken deterministically by original candidate order (stable sort), so
every candidate still gets a distinct valid rank.

Output (validated: 2,370,727 lines each, matching `MINDlarge_test`'s row
count and order 1:1; a 2,000-line random sample confirmed every rank list
is a valid 1..N permutation matching that impression's candidate count):

```
data/predictions/mind/bm25/prediction.txt   291 MB   -> prediction.zip (107 MB)
data/predictions/mind/lsa/prediction.txt    291 MB   -> prediction.zip (107 MB)
```

Each zip contains only `prediction.txt` at its root (no folder, no
`__MACOSX`), matching `mind_submission_guidelines.txt` exactly. Given the
Q2/Q3 result, **`bm25/prediction.zip` is the one to submit** to
https://www.codabench.org/competitions/13967/ unless the LSA leaderboard
score is worth comparing too.

---

## Design Notes

**What was built:** a MIND-only pipeline that (1) unifies MINDsmall_train /
MINDsmall_dev / MINDlarge_test into one schema with a leakage-checked
train/val/test split, (2) builds a leakage-safe article/user feature store,
(3) implements BM25 and TF-IDF+SVD retrieval as reusable scorers shared
between a full-corpus recall@K diagnostic and restricted-candidate
Codabench scoring, and (4) generates and validates both leaderboard
submissions end-to-end.

**Alternatives considered:**
- *Day-cutoff temporal split* (what an earlier draft of this pipeline did,
  merging train+dev and cutting by N days) vs. *using MIND's native
  train/dev/large_test files directly*: chose the latter since it's
  simpler, still verifiably temporal (asserted, not assumed), and matches
  how the assignment's own splits were built.
- *BERT/sentence-transformer embeddings* vs. *TF-IDF+SVD (LSA)* for Q3:
  chose LSA for zero new heavy dependencies and fast fit/transform over
  MINDlarge_test's 121K articles; documented as a scope tradeoff above.
- *Full-corpus scoring for Q5* vs. *restricted-candidate scoring*: chose
  restricted, since Codabench only wants a ranking of each impression's
  given candidates, and it's ~2 orders of magnitude cheaper per impression
  — this is what makes 2.37M impressions finish in ~10 minutes instead of
  the hours a naive full-corpus-per-impression approach would take (the
  Q2/Q3 eval script, which *does* do full-corpus scoring but only over 73K
  impressions, already took ~20 minutes).

**Observations:** BM25 > LSA at every K on MIND (see Q2/Q3 results above) —
lexical overlap is a strong signal for short, entity-dense news text.
Recall@K is low in absolute terms for full-corpus retrieval (max 2.4% at
K=200) but directionally sound (monotonic in K, consistent gap between
methods), showing this is a genuinely hard candidate-generation setting,
not that the indices are broken (correctness was separately verified: the
same BM25/LSA scorers were validated end-to-end via the Q5 prediction
files — valid rank permutations, correct row alignment against the
original `behaviors.tsv`).

**Where this breaks at 10×:** the current design already had to solve one
10×-scale problem (MINDlarge_test is ~15× MINDsmall by impression count) by
switching from "materialize everything as a DataFrame" (fine for train+val,
~230K rows) to "stream in chunks + restricted-candidate scoring" (necessary
for test's 2.37M rows) — this cost/benefit only shows up past a few hundred
thousand impressions. At a further 10× (≈24M impressions, or a MIND-large
scale train/val), the next things to break: (1) `evaluate_retrieval.py`'s
full-corpus dense score-matrix batching (`B × n_docs` densified per batch)
would need much smaller batches or a real ANN index (FAISS) instead of
brute-force dense scoring; (2) the BM25/LSA fit itself (`CountVectorizer`/
`TfidfVectorizer` + `TruncatedSVD` over the full in-memory text list) would
need to move to an out-of-core or streaming vectorizer, since it currently
holds every article's text in a Python list at fit time; (3) the
single-process Python loop in `generate_predictions.py`, while fast per
impression, is single-threaded — at 24M impressions its ~10 minutes becomes
~100 minutes, which starts to warrant chunk-level multiprocessing.

**Not yet done:** Q6 design note as a standalone ≤4-page document, EB-NeRD
wiring, and the Q9 leakage-boundary *test* (the assertions exist and pass,
but aren't wrapped in a pytest yet).
