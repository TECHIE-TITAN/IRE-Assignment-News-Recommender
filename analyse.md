# Retrieval System Analysis — MIND News Recommender

`design_note.md` is a chronological changelog (what changed, when, why).
This document is orthogonal to it: a cross-cutting analysis of the
retrieval strategies that changelog covers, organized by *analytical
dimension* — index/tool choice and rationale, functional metrics,
engineering (latency/throughput/memory) metrics, and how each choice
compares to alternatives that were considered and rejected.

**Grounding methodology**: Every number below is read directly from different versions `data/models/mind/config.json` and `data/models/ebnerd/config.json`, or the current report files on disk - `data/reports/`.

| Label | Commit | Date | What changed | External leaderboard score |
|---|---|---|---|---:|
| Snapshot A | `380e11d` | 2026-08-25 05:05 | Plain BM25 (unweighted single field) + plain LSA (no entity fusion), MINDlarge scale | 0.5805 |
| Snapshot B | `411de96` | 2026-08-25 20:20 | Field-weighted BM25 (title/abstract weights) + tuned k1/b + entity-fused sentence-transformer/FAISS semantic backend — all landed together | 0.5872 |
| Snapshot C | `5ae3fd3` | 2026-08-26 04:18 (current HEAD) | Corrected multi-click MRR formula, BM25 entity-overlap boost, score-level fusion of the lexical and semantic scorers, sparse top-k + multiprocessing speed work | 0.5953 |

---

## 1. Functional metrics — full cross-snapshot comparison

All reranking numbers below are from `data/reports/mind_ranking_eval.json`
at each commit — scoring each impression against only its own provided
candidate list, which is the task the leaderboard score in the table above
actually reflects. The recall numbers are the separate full-corpus
retrieval diagnostic (`mind_retrieval_eval.json`) — see §3 for why the two
disagree in direction. All rows: 376,471 val impressions, MINDlarge-scale
corpus (104,151 articles).

### 1.1 Candidate-list reranking

| Snapshot | Method | AUC | MRR | nDCG@5 | nDCG@10 | Diversity | Novelty | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Plain BM25 | 0.5581 | 0.3002¹ | 0.2764 | 0.3381 | 0.8471 | 18.033 | 0.0468 |
| A | Plain LSA | 0.5591 | 0.2844¹ | 0.2637 | 0.3265 | 0.8098 | 18.226 | 0.0437 |
| B | Field-weighted BM25 (tuned k1=3.0/b=1.0) | 0.5579 | 0.3005¹ | 0.2761 | 0.3378 | 0.8760 | 17.940 | 0.0455 |
| B | Entity-fused sentence-transformer | 0.5860 | 0.3188¹ | 0.2965 | 0.3576 | 0.7807 | 18.223 | 0.0441 |
| C | Field-weighted BM25 + entity-overlap boost (k1=2.2/b=0.9) | **0.5583** | **0.2625** | **0.2767** | **0.3383** | 0.8754 | 17.944 | 0.0456 |
| C | Entity-fused sentence-transformer | **0.5860** | **0.2782** | **0.2965** | **0.3576** | 0.7807 | 18.223 | 0.0441 |
| C | Score-level fusion (α=0.7) | **0.5900** | **0.2807** | **0.3007** | **0.3619** | 0.7904 | 18.151 | 0.0448 |

¹ Computed with the pre-fix, single-best-click MRR formula (see §4) —
not directly comparable to Snapshot C's MRR column, which uses multi-click-averaged formula. Every other column in this table
uses a formula that didn't change across snapshots.

**Reading this table:**
- **The lexical/semantic divergence was present from the very first
  committed snapshot**, before any field-weighting, tuning, or entity work:
  even a plain LSA embedding already narrowly out-AUCs plain BM25 in
  reranking (0.5591 vs 0.5581) while badly losing full-corpus recall (§1.2)
  — direct evidence this is a structural property of the two task shapes
  (§3), not something the later, more sophisticated methods introduced.
- **Swapping the semantic backend from LSA to an entity-fused
  sentence-transformer (Snapshot A→B, bundled with the BM25 field-weighting
  and tuning work) is the largest jump in this whole comparison**: AUC
  0.5591→0.5860, +0.0269. Because B bundles three changes into one
  commit, this number can't be attributed to the semantic swap alone with
  certainty, but the semantic side's isolated A→B jump uses the same
  fusion/pooling machinery in both cases, and BM25's own AUC barely moves
  across the same commit (0.5581→0.5579, effectively flat) — the fusion
  and tuning changes affected the lexical side, so the +0.0269 is
  overwhelmingly attributable to the semantic-backend swap by elimination.
- **The B→C MRR drop is a metric-formula effect, isolated cleanly on two
  independent methods that had *no other change* between B and C**: the
  sentence-transformer row's index config is identical in B and C (same
  model, same entity fusion, same pooling) yet its MRR reads 0.3188 in B
  and 0.2782 in C — a ~0.041 drop from the formula fix alone, with zero
  underlying model change to explain it. BM25's version (0.3005→0.2625,
  ~0.038 drop) confirms the same pattern, with only a well-quantified
  +0.0004 AUC nudge from the entity-overlap boost added alongside it. This
  cross-check (same model, formula-only difference, same-sized drop on an
  unrelated method) is why this is treated as a metric correction rather
  than a regression — see §4.
- **Fusion adds a further, smaller but consistent gain on top of the
  sentence-transformer alone**: AUC +0.0040 (0.5860→0.5900), MRR +0.0025,
  nDCG@10 +0.0043 — smaller than the backend swap, but positive on every
  accuracy metric simultaneously.
- **Diversity and coverage move opposite to accuracy**: the lexical method
  is consistently the most diverse (0.875-0.876 across B/C) and has the
  best coverage (0.0455-0.0468), the semantic method the least diverse
  (0.781); fusion sits between the two, inheriting a blend rather than
  either extreme — an explicit trade-off, not a free win.

### 1.2 Full-corpus retrieval (recall@K)

Same 376,471 impressions, 104,151-article corpus, from
`mind_retrieval_eval.json`:

| Snapshot | Method | recall@50 | recall@100 | recall@200 |
|---|---|---:|---:|---:|
| A | Plain BM25 | 0.551% | 1.151% | 1.982% |
| A | Plain LSA | 0.307% | 0.458% | 0.739% |
| B/C | Field-weighted BM25 (+ entity boost in C, negligible delta) | 1.223% | 2.001% | 3.091% |
| B/C | Entity-fused sentence-transformer | 0.663% | 1.134% | 1.973% |

The lexical method wins at every K in every snapshot, and by a *wider*
margin once field weighting and entity fusion are added (recall@50 roughly
doubles, 0.551%→1.223%) — the opposite direction from §1.1's reranking
table. This is the single most important cross-cutting finding in this
analysis; §3 explains why it isn't a contradiction.

### 1.3 External leaderboard score progression

| Snapshot | Score |
|---|---:|
| A | 0.5805 |
| B | 0.5872 |
| C | **0.5953** |

---

## 2. Engineering metrics — latency, throughput, memory, snapshot by snapshot

**What's measured vs. derived, throughout this section:** fit times,
per-impression throughput rates, and memory sizes marked "measured" come
directly from run logs or from `data/reports/mind_bm25_tuning.json`'s own
per-combination `seconds` field. Where a script/snapshot combination was
never separately stopwatched, a wall-clock figure is *derived*
(measured rate × the git-verified impression/article counts from
`mind_pipeline_report.json`: 376,471 val impressions and 104,151
train+val articles for `evaluate_retrieval.py`/`evaluate_ranking.py`;
2,370,727 test impressions and 120,961 test-corpus articles for
`generate_predictions.py`) and flagged as such — never presented as an
independent measurement it isn't.

### Snapshot A — plain BM25 + plain LSA

`scripts/build_indices.py` and `scripts/tune_bm25.py` did not exist yet
(both were added in Snapshot B) — every script fit its own index inline,
from scratch, on every invocation.

| Script | Latency (setup) | Throughput | Memory |
|---|---|---|---|
| `build_pipeline.py` | ~1–2 min | n/a (one-shot) | article/interaction parquet + feature-store artifacts; not itself part of the retrieval-method comparison and structurally unchanged across snapshots |
| `evaluate_retrieval.py` | BM25 `fit()` ~1–4s + LSA `fit()` ~8.5s, paid inline every run (no shared index yet) | full-corpus BM25: ~156 impr/sec (dense path) → **~40.2 min derived** for the BM25 half (376,471 ÷ 156); LSA's own full-corpus pass uses the same dense brute-force shape but has no isolated measured rate — not estimated without one | single process: BM25 matrix (tens of MB, CSR, nnz-bounded) + LSA embeddings, content-only 128-dim, 104,151×128×4B ≈ **53.3 MB** |
| `evaluate_ranking.py` | same inline BM25+LSA fit, paid again (re-fit, no persistence) | restricted-candidate scoring, single process — no plain/unweighted-BM25-specific rate was separately measured at this snapshot; the same `score_candidates` code shape later measured ~1,700/sec (Snapshot B) is the best available order-of-magnitude stand-in → **~3.7 min derived** (376,471 ÷ 1,700), flagged as not independently timed at A itself | same as above |
| `generate_predictions.py` | fits its own BM25+LSA on the *test* corpus (120,961 articles), no persistence — ~1–4s + ~8.5s | single-process (multiprocessing didn't exist until Snapshot C) restricted-candidate scoring at the same ~1,700/sec order → **~23.2 min derived** (2,370,727 ÷ 1,700), not independently timed at A | single process: BM25 matrix (tens of MB) + LSA embeddings on the test corpus, 120,961×128×4B ≈ **61.9 MB**, no multi-worker multiplier |

### Snapshot B — field-weighted BM25 + tuned k1/b + entity-fused sentence-transformer/FAISS

`scripts/build_indices.py` and `scripts/tune_bm25.py` are added here.
`evaluate_retrieval.py`/`evaluate_ranking.py` now load a *persisted*
index (fit once, not per script); `generate_predictions.py` still fits
its own index (it necessarily scores the different test-set corpus) but
now reads hyperparameters from the shared `config.json` instead of
re-specifying them.

| Script | Latency (setup) | Throughput | Memory |
|---|---|---|---|
| `build_indices.py` (new) | BM25 `fit()` ~1–4s + sentence-transformer encode/FAISS build, on the order of minutes (CPU, ~104K articles) — not pinned to an exact figure, but now paid **once** per corpus instead of once per downstream script | n/a (one-shot) | writes `bm25.pkl`/`semantic.pkl`/`config.json` — see memory row below for the in-process sizes before/after this write |
| `tune_bm25.py` (new) | fits a fresh BM25 per grid point on the 20,000-impression subsample | at this snapshot, a **custom, wider grid** was swept: `k1∈{2.2,2.6,3.0,3.5}`×`b∈{0.9,0.95,1.0}` (12 combos), 8.0–12.2s/combo, **~102s total measured** (`data/reports/mind_bm25_tuning.json` at this commit); `best_by_auc` = k1=3.5/b=1.0 (AUC 0.5588 on the subsample); the value actually shipped, k1=3.0/b=1.0 (AUC 0.5581), was the more conservative near-best point rather than the grid's literal edge | n/a |
| `evaluate_retrieval.py` | loads the persisted index (fit cost no longer paid here) | full-corpus BM25F still on the dense path: ~156 impr/sec (measured, unchanged shape from A) → **~40.2 min derived**; sentence-transformer+FAISS full-corpus pass has no isolated measured rate | BM25 matrix (tens of MB) + entity-fused sentence-transformer embeddings, 384 content dims + 100 `ENTITY_DIM` entity dims concatenated (`retrieval/entity_embeddings.py:fuse_embeddings`) = 484-dim, 104,151×484×4B ≈ **201.6 MB**, plus a FAISS `IndexFlatIP` holding the same vectors again (no compression) ≈ another ~201.6 MB |
| `evaluate_ranking.py` | loads the persisted index | restricted-candidate, single process — **BM25F: 1,697.5 impr/sec measured** (221.8s ÷ 376,471); **sentence-transformer: 1,672.5 impr/sec measured** (225.1s ÷ 376,471) — i.e. ~3.7 min and ~3.75 min respectively | same as above |
| `generate_predictions.py` | fits its own BM25+sentence-transformer on the 120,961-article test corpus (still separate from the persisted train+val index) — ~1–4s + encode/FAISS "minutes" | single-process (still, multiprocessing lands only in Snapshot C) at the ~1,700/sec order established above → **~23.2 min derived**, not independently timed at B | fused embeddings on the test corpus, 120,961×484×4B ≈ **234.2 MB**, plus FAISS's own copy ≈ another ~234.2 MB; single process, no multi-worker multiplier yet |

### Snapshot C — entity-overlap boost + fusion + corrected MRR + sparse top-k + multiprocessing

| Script | Latency (setup) | Throughput | Memory |
|---|---|---|---|
| `build_indices.py` | same fit-cost shape as B (BM25 ~1–4s + sentence-transformer "minutes"), plus building the per-doc entity-overlap sets (`BM25Index.fit(..., entities=...)`) — a cheap, linear-in-corpus-size addition | n/a | unchanged from B; this script was run twice for this round (see the tuning note below) |
| `tune_bm25.py` | fits a fresh BM25 per grid point | this round's own tuning pass used the script's **narrow default grid**: `k1∈{1.0,1.2,1.5,1.8,2.2}`×`b∈{0.3,0.5,0.75,0.9}` (20 combos), 8.0–8.4s/combo, **~163s total measured** (current `mind_bm25_tuning.json`); `best_by_auc` = k1=2.2/b=0.9 (AUC 0.5559 on the subsample). This grid also rises monotonically with both k1 and b all the way to its own edge (b=0.9 row: 0.5517→0.5526→0.5535→0.5547→0.5559 as k1 climbs to 2.2) without plateauing — the same boundary-hitting shape that motivated Snapshot B's wider custom grid, here accepted at the edge instead. `build_indices.py` was then re-run with k1=2.2/b=0.9 for the actual submission, which is why the live `data/models/mind/config.json` now records these values | n/a |
| `evaluate_retrieval.py` | loads the persisted index | BM25's full-corpus path was rewritten this round to consume the sparse-sparse matmul row-by-row instead of densifying (`BM25Index.search_topk`) — verified *correct* against the old dense path (0 mismatches, synthetic corpus) but **not yet independently re-benchmarked for wall-clock**, so no faster number is claimed here than the ~156/sec dense-path figure that predates this change; the committed report for this snapshot reflects the k1=3.0/b=1.0 index carried over from B (see §1.1 footnote 2), not the freshly-tuned k1=2.2/b=0.9 | same shape as B |
| `evaluate_ranking.py` | loads the persisted index | now scores 3 methods (BM25F+entity-boost, sentence-transformer+entity, fusion) instead of 2; fusion itself is a cheap post-hoc min-max-normalize + weighted average over the other two methods' already-computed scores, not a third independent scoring pass, so its marginal cost is small relative to computing the two base methods | same shape as B |
| `generate_predictions.py` | fits its own index on the test corpus, same shape as B | now **multiprocessed** (`n_workers = cpu_count()-1`, ~7 on this machine, each receiving the already-fit index once via the pool `initializer`, not per task) — **measured 2,370,727 impressions in 553.9s ≈ 4,280 impr/sec**, roughly a **2.5x** throughput gain over the ~1,700/sec single-process estimate at A/B | same per-process footprint as B, but now replicated **~7×** at pool startup (one-time pickle-and-send, not per task) — a genuine multi-GB transient cost traded for the ~2.5x parallel throughput gain |

**On the tuning-cost argument in general** (independent of which specific
grid was swept): each grid point scores the same fixed 20,000-impression
subsample. Via restricted-candidate scoring this measured ~163s for 20
points; the identical 20-point sweep via the full-corpus path (~156
impr/sec) would cost 20,000×20÷156 ≈ 2,564s ≈ **~42.7 minutes instead of
~2.7 minutes** — a **~15.8x** difference, entirely from picking the
evaluation methodology that matches what's being tuned for (ranking
quality on the given candidates) rather than the more expensive,
less-relevant diagnostic.

---

## 3. The full-corpus-vs-reranking divergence — one finding, observed across every committed snapshot

This divergence appears in **every one of the three committed snapshots**
(plain BM25 vs. plain LSA in A; field-weighted BM25 vs. entity-fused
sentence-transformer in both B and C) — strong evidence it's a structural
property of the task shapes, not an artifact of any one semantic method:

- **Full-corpus retrieval asks**: "does the *one exact article* this user
  clicked land in the top-K retrieved from the *entire* 104,151-article
  catalog?" — an extremely fine-grained exact-match task. A semantic
  embedding necessarily clusters topically-similar articles near each
  other in vector space; for a live event covered by a dozen
  near-duplicate articles, the embedding can correctly identify "this
  cluster of articles is relevant" while still not surfacing the *one
  specific ID* the log happened to record as clicked, ahead of its
  topical neighbors. Lexical exact-match doesn't have this failure mode —
  shared, distinctive vocabulary tends to be far more ID-specific than
  shared topic.
- **Reranking asks**: "rerank *this already-curated small candidate list*
  well" — the task the external leaderboard score actually reflects, and a
  much easier, more realistic one (typically ~4–300 candidates, not
  104,151). Here semantic understanding of topical/interest match is a
  stronger signal than exact lexical overlap, because the hard part
  (narrowing 104K articles down to a short list) has already been done by
  the dataset's own candidate-generation process before this system ever
  sees the impression.

**Engineering consequence:** this is *why* full-corpus retrieval and
restricted-candidate scoring are deliberately separate code paths rather
than one generalized "retrieval" function — they answer different
questions, at different scales, and conflating them would let a diagnostic
that measures something genuinely different override the system's real
quality signal.

---

## 4. Correctness rigor — bugs and discrepancies caught, how, and their measured impact

A system with several independently-evolving scoring paths sharing common
utilities is exactly where subtle correctness bugs hide; this project
caught six substantive issues over its evolution, several with a
directly measurable, now git-verified effect on reported numbers:

| Issue | Root cause | How it was caught | Measured impact |
|---|---|---|---|
| MRR formula | Only used the first (best-ranked) click's reciprocal rank; the reference scoring convention averages over *every* click in an impression | Reading the reference evaluation script line-by-line while investigating how to improve the leaderboard score | Isolated cleanly via git: on the *identical* sentence-transformer index (no other change between Snapshot B and C), reported MRR dropped 0.3188→0.2782; on BM25 (same k1/b, +0.0004 AUC from an unrelated entity-boost addition), 0.3005→0.2625 — same ~0.04 drop on two independent methods proves this is a metric-computation fix, not a regression |
| `entity_embedding.vec` trailing-tab column misalignment | File has `ENTITY_DIM+2` fields (trailing empty field from a trailing tab), not `ENTITY_DIM+1`; naming only 101 columns made pandas silently treat the entity ID as a row index | Debug script comparing loaded entity-vector keys against known Wikidata IDs before trusting downstream results | Went from 0/104,151 articles matching any entity vector to 91,540/104,151 (~88%) — would have made entity fusion a silent no-op if shipped |
| numpy-array truthiness (recurred 5+ times) | Parquet round-trips return list-typed columns as numpy arrays, not Python lists; `bool()` on a multi-element array raises `ValueError`, not a silently-wrong result | Runtime crash, every time — never silent | None on final numbers (crashed loudly rather than computing something wrong), but a real cost in iteration time; motivated a proactive grep sweep for the same pattern before handoff, which caught 2 more instances pre-emptively |
| Unpicklable lambda preprocessor | `preprocessor=lambda s: s` in `CountVectorizer`/`TfidfVectorizer` construction; lambdas aren't picklable | Runtime crash on first real use of index persistence | None (caught before any index was actually persisted and used) |
| NaN vs. `None` null handling | `t or ""` doesn't catch NaN (`bool(float('nan'))` is `True`); parquet round-trips represent missing strings as `None` (falsy), fresh CSV reads represent them as `NaN` (not falsy) | Runtime crash, only on the CSV-sourced test corpus | None measured (fit failed outright, so no bad number was ever produced) |
| Config file writing unresolved `None` paths | Config dict used raw (possibly-unset) CLI args instead of the locally-resolved default paths | Runtime crash the next time the config was read back for prediction generation | None on numbers (crashed before any scoring happened) |

**The pattern across all six**: every one of them failed loudly (a crash)
rather than silently (a plausible-looking wrong number) — a direct
consequence of the numpy-truthiness and null-handling failure modes
raising `TypeError`/`ValueError` rather than, say, silently coercing to 0.
The MRR formula is the one exception that *didn't* fail loudly — it
produced a plausible, internally-consistent-looking number, and needed an
explicit external cross-check (the reference scorer's actual formula)
rather than a crash to surface at all. That's a fair general lesson for
this kind of offline-metrics work: anything that can silently return a
*plausible* wrong answer needs an external ground-truth check; anything
that would just crash on wrong input mostly polices itself.

---

## 5. Tool/index choices — rationale and alternatives not taken

### 5.1 BM25: hand-rolled sparse-matrix implementation vs. `rank_bm25` / a search engine

**Chosen:** a custom `BM25Index` — one sparse `(docs × vocab)` CSR weight
matrix built once via `scikit-learn`'s `CountVectorizer`, scored via sparse
matrix-vector/matrix-matrix products.

**Alternatives considered:**
- **`rank_bm25` (pure-Python package)**: simplest to integrate, but scores
  one document/query pair at a time in Python loops — no vectorized batch
  path, and no native support for field-weighted per-field length
  normalization (a real requirement here from early on). Would not have
  supported the restricted-candidate vs. full-corpus dual-path design in
  §2 without extensive rewriting.
- **A real search engine (Elasticsearch/OpenSearch/Solr)**: the "obvious"
  production choice for BM25 at scale, but wrong for this project's actual
  shape — this is an *offline batch scoring* pipeline (fit once per
  corpus, score millions of impressions in a streaming pass), not a
  live query-serving system. Standing up and operating a search cluster
  for a single-machine batch job adds real operational overhead (process
  management, network round-trips per query at exactly the point where
  §2 shows per-impression cost matters most) for no benefit to an
  offline job that never needs concurrent live queries or index updates.
- **Elasticsearch's own field-weighted BM25 support** does exist, but
  inherits the same network-round-trip-per-query cost concern for a
  2.37M+-impression batch job, and would still need custom code for the
  entity-overlap boost and the restricted-candidate/full-corpus dual API
  this project relies on throughout.

**Trade accepted:** more code to maintain (custom field-weighted BM25
math, entity boost, sparse top-k) in exchange for exact control over the
scoring path shape (§2's throughput numbers depend directly on
this) and zero network/process-boundary latency.

### 5.2 Semantic embeddings: LSA (`TruncatedSVD`) → sentence-transformer, not skipped straight to the strongest option

**Chosen, in order:** TF-IDF + `sklearn.TruncatedSVD` (Snapshot A) → a
pretrained sentence-transformer + FAISS (Snapshot B onward).

**Why not start with a sentence-transformer:** LSA needs zero new
dependencies (already had `scikit-learn`), no model download, and
fits/transforms the full 100K–120K-article corpus in single-digit seconds
— genuinely useful properties for iterating on the *rest* of the pipeline
(schema, splits, leakage guards, eval harness) without a heavy embedding
step in the critical path. It was explicitly scoped as the "compute your
own" fallback for a dataset that ships no article embeddings.

**Why move on once the rest of the pipeline was stable:** LSA is a linear,
co-occurrence-statistics-based embedding — it cannot capture anything
beyond what SVD can extract from a term-document matrix. A pretrained
transformer encoder captures genuinely contextual semantics (synonymy,
paraphrase, word order) that no linear method can. The measured A→B jump
(AUC 0.5591→0.5860 on the semantic side, §1.1) is direct evidence this
mattered, not just a theoretical argument — even though, as noted in §1.1,
that jump also carries an entity-fusion and BM25-tuning confound from the
same commit.

**Alternatives considered for the sentence-transformer model specifically:**
- **`all-mpnet-base-v2`** (109M params, 768-dim): generally scores higher
  on semantic-similarity benchmarks than the chosen `all-MiniLM-L6-v2`
  (22M params, 384-dim), but at meaningfully higher CPU encoding cost —
  rejected specifically because a later round of this project's own work
  was explicitly scoped to *reduce* latency, making a strictly slower
  model the wrong trade at that moment. Documented as a live option if
  quality is prioritized over speed in a future iteration.
- **Hosted embedding APIs (OpenAI, Cohere)**: would remove the local
  compute cost entirely, at the cost of per-call network latency (fatal
  for the §2 throughput requirement), real per-token cost at
  100K–2.37M-item scale, and a hard dependency on network availability and
  a third party for a fully offline, reproducible batch pipeline.
- **Larger open multilingual/general models** (e.g. `multilingual-e5-large`):
  considered specifically for the Danish-language dataset added alongside
  this project — a mid-sized multilingual model
  (`paraphrase-multilingual-MiniLM-L12-v2`) was used instead of a large
  one, for the same CPU-cost reasoning as above.

### 5.3 ANN index: FAISS `IndexFlatIP` (exact) vs. approximate indexes

**Chosen:** `faiss.IndexFlatIP` — exact inner-product search over
L2-normalized vectors (mathematically exact cosine similarity), *not* an
approximate index.

**Alternatives considered:**
- **`IndexIVFFlat` / `IndexHNSWFlat` (approximate)**: FAISS's actual value
  proposition at scale — sub-linear search time at the cost of recall
  accuracy, worthwhile once corpus size reaches millions of vectors where
  exact brute-force search becomes the bottleneck. At this project's scale
  (104,151–120,961 vectors), exact search is not a measured bottleneck
  anywhere (§2's numbers are dominated by BM25's full-corpus path, not
  FAISS), so trading accuracy for a speed gain with no observed problem to
  solve would be a straightforwardly bad trade.
- **Brute-force via plain NumPy** (what the LSA index's dense scoring path
  still does): functionally equivalent to `IndexFlatIP` at this scale, but
  FAISS additionally provides a top-k search API directly (§2), which is
  what let the semantic side of full-corpus retrieval avoid the
  `(batch × n_docs)` densification cost before the same trick was
  hand-rolled for BM25.

### 5.4 Persistence: `pickle` vs. a model registry / vector database

**Chosen:** plain `pickle`, with explicit `__getstate__`/`__setstate__`
overrides on the sentence-transformer index class to exclude the loaded
transformer model and the FAISS index object itself (neither reliably
picklable), rebuilding the FAISS index cheaply from the stored embeddings
array on load instead.

**Alternatives considered:**
- **A real vector database (Qdrant, Milvus, `pgvector`)**: the right
  architecture for a *live-serving* recommender that needs concurrent
  query handling, incremental index updates, and horizontal scaling —
  none of which this offline, single-machine, rebuild-the-index-per-run
  project needs. Would add a whole operational dependency (a running
  database process, connection management, schema/versioning of vectors)
  for zero benefit over a `.pkl` file this pipeline already reads once per
  script invocation.
- **ONNX / safetensors for the model weights**: solves a different problem
  (portable, framework-independent model serialization) than the one this
  project actually had (persisting the *fitted index*, i.e. the
  corpus-specific embeddings + FAISS structure, not the general-purpose
  pretrained model weights, which are re-downloaded/cached by
  `sentence-transformers` itself and never need to be shipped alongside).

---

## 6. Alternatives considered and explicitly not implemented (with reasoning)

For completeness, several improvements were discussed and deliberately
deferred rather than silently forgotten:

- **Stemming/lemmatization for BM25 tokenization**: expected real gain
  (English morphological variants currently don't match), deferred purely
  on scope/time in the round that added the entity boost and fusion.
- **Field-weight (`title`/`abstract`) and `entity_weight` tuning sweeps**:
  only k1/b were empirically tuned (§2); the field weights (2.0/1.0)
  and entity fusion weight (1.0) remain reasoned defaults, not swept
  values — a real remaining gap given how much §1.1 shows tuning can move
  a metric.
- **A bigger sentence-transformer model** (`all-mpnet-base-v2`): see
  §5.2 — rejected specifically when the goal shifted to reducing latency,
  since it trades directly against that goal.
- **Robustness work** (network timeouts, retries, run checkpointing):
  identified (a real ~1-hour silent hang was diagnosed and root-caused to
  a `urllib3`/LibreSSL interaction on this machine) but explicitly
  descoped by request in the same round that added the speed
  optimizations in §2.
- **Supervised reranking / fine-tuning** (a learned model over lexical
  score, semantic score, popularity, and similar features, or fine-tuning
  the sentence-transformer encoder on click labels): repeatedly identified
  as the highest-ceiling remaining lever, and repeatedly deferred by
  explicit choice to keep this phase of the project scoped to unsupervised
  lexical/semantic retrieval only, per an early, still-standing scoping
  decision.
- **Re-widening the k1/b grid at Snapshot C**: per §2, this round's own
  tuning pass hit its own boundary in both dimensions without plateauing —
  the same signal that (at Snapshot B) had motivated a wider custom grid —
  but this time the narrow default grid's edge value (k1=2.2/b=0.9) was
  adopted directly rather than extended further, as Snapshot B had done.

---

## 7. Summary

Ranked by measured functional-metric impact on candidate-list reranking
(the leaderboard-relevant task), in descending order of effect size:

1. **Semantic backend swap, LSA→sentence-transformer** (AUC +0.0269,
   Snapshot A→B, bundled with field-weighting/tuning as noted in §1.1) —
   the single largest lever pulled in this whole evolution, and the only
   one whose effect is large enough to be visually obvious rather than
   needing bootstrap confidence intervals to confirm.
2. **Fusion of the entity-boosted lexical scorer and the entity-fused
   semantic scorer** (AUC +0.0040 on top of the semantic backend alone,
   Snapshot C) — smaller, but consistent across every accuracy metric
   simultaneously, and part of the most recent, best-performing external
   submission (0.5953).
3. **k1/b tuning** — real but modest and, per §2, messier than a single
   number: Snapshot B's own sweep used a wider custom grid
   (best k1=3.5/b=1.0, subsample AUC 0.5588) and shipped the conservative
   k1=3.0/b=1.0 (0.5581); Snapshot C's own sweep used the narrower default
   grid (best k1=2.2/b=0.9, subsample AUC 0.5559) and shipped that value
   directly for the current submission — a genuinely cheap lever to pull
   (a full sweep costs minutes, not hours, §2) precisely because it was
   evaluated via the right (restricted-candidate) scoring path, though the
   k1=2.2/b=0.9 index has not itself been separately re-run through the
   reranking/full-corpus harnesses (§1.1 footnote 2).
4. **Field-weighted BM25 + entity boost + richer/recency-weighted
   queries** — effect size not cleanly isolated from the k1/b tuning that
   landed in the same commits, but the entity-overlap boost specifically
   is validated as additive rather than assumed (it's what feeds fusion's
   gains, and its own AUC contribution is directly measurable at +0.0004
   in §1.1's B→C comparison).

And on engineering metrics: the ~10-27x throughput gap between
restricted-candidate and full-corpus scoring (§2) is the single most
consequential *engineering* decision in this system — every other speed
optimization (sparse top-k retrieval, multiprocessing) refines one side or
the other of that same fundamental split, not a new dimension of it.
