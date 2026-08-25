# Design Note — MIND News Recommender

A running log of design changes: what existed before, what changed, why,
and what was learned (including bugs caught along the way). Updated as the
pipeline evolves; not a snapshot of only the current state — see
`README.md` for that. `Assignment.md` is the spec being implemented;
`CLAUDE.md` is the architecture reference for the current codebase.

---

## Revision 1 — Initial pipeline (Q1–Q5), MINDsmall scale

**Data scope:** `MINDsmall_train` (train, 156,965 impressions) /
`MINDsmall_dev` (val, 73,152 impressions) / `MINDlarge_test` (test,
2,370,727 impressions, unlabeled) — MIND's own files used directly as the
temporal split, asserted monotonic rather than re-derived by day-cutoff.

- **Q1**: unified schema pipeline (`pipeline/{schema,mind,split,feature_store}.py`),
  leakage-safe article/user feature store.
- **Q2**: plain BM25 — title+abstract flattened into one field, no field
  weighting — over the 65,238-article train+val corpus.
- **Q3**: LSA (TF-IDF + truncated SVD, 128-dim), content-only, no entity
  signal, brute-force cosine (spec-permitted at this scale).
- **Q4**: AUC/MRR/nDCG@5/@10, diversity/novelty/coverage, cold-vs-warm
  slicing, bootstrap 95% CIs.
- **Q5**: predictions for all 2,370,727 `MINDlarge_test` impressions,
  restricted-candidate scoring (`score_candidates`, not full-corpus
  retrieval) so the run finishes in ~10 minutes.

**Baseline result:** BM25 beat LSA on *every* metric — recall@K, AUC, MRR,
nDCG, diversity, coverage. Val AUC: BM25 0.5567, LSA 0.5552. Submitted
BM25 predictions to Codabench: **0.5805**.

---

## Revision 2 — Scale change: MINDsmall → MINDlarge train/val

| | before | after |
|---|---:|---:|
| articles (train+val) | 65,238 | 104,151 |
| train impressions | 156,965 | 2,232,748 |
| val impressions | 73,152 | 376,471 |

**Why:** explore whether a larger, more representative train/val corpus
improves retrieval quality — a lever floated in discussion and then acted
on directly by editing `build_pipeline.py`'s raw-dir arguments.

**Also added in this revision:** recall@K slicing (cold-start vs. warm) in
`evaluate_retrieval.py` — previously only Q4 sliced results, Q2/Q3's
recall@K was a single aggregate number, which was an incomplete answer to
Q3.5 ("compare lexical vs. semantic... on which slices"). Required changing
`retrieval/eval_utils.py`'s `recall_at_k_from_scores` (returned a
pre-averaged scalar) into `recall_hits_matrix` (returns a per-impression
hit array), so slicing could be done post-hoc by masking rows.

**Result at this point (algorithm unchanged, data scale only):** recall@K
dropped slightly in absolute terms (bigger haystack, same top-K budget:
K=200 BM25 2.42%→1.98%) but the pattern held — BM25 still ahead of LSA at
every K, every slice, cold worse than warm for both methods.

---

## Revision 3 — Retrieval algorithm improvements

| | before | after |
|---|---|---|
| BM25 fielding | title+abstract flattened, one field | **BM25F**: title/abstract length-normalized *separately* against their own average length, combined via per-field weights (title=2.0, abstract=1.0) before BM25 saturation (Zaragoza et al. formulation) |
| Query text source | history article **titles** only | history article **title+abstract** (`article_text`) — matches what the corpus itself is indexed on |
| Query weighting | every recent history article counted equally | **recency-weighted**: exponential decay (default 0.85/rank-step) on both BM25 query-term weights and LSA mean-pooling weights, via a shared `text_utils.recency_weights` curve |
| BM25 query representation | binary token **set** (`score = Σ 1·weight`) | weighted term **dict** (`score = Σ w(t)·weight`) — the set representation had to go; recency weighting is meaningless if repeated/weighted terms still collapse to a single binary presence bit |
| LSA embedding | content-only (TF-IDF+SVD) | **entity-fused**: MIND's pretrained TransE knowledge-graph embeddings (parsed into the schema back in Q1, never used until now) mean-pooled per article and L2-normalize-concatenate-renormalize-fused with the content embedding (`retrieval/entity_embeddings.py`) |
| LSA mean pooling | unweighted mean of history embeddings | recency-weighted mean (same decay curve as BM25's query weighting) |

**Bugs caught and fixed during this revision:**

1. **numpy-array truthiness** (`if not history_ids: ...`) — recurred three
   separate times across the conversation (`text_utils.build_query_text`,
   then again in `evaluate_ranking.py`, then again in
   `evaluate_retrieval.py`) because parquet-sourced list columns come back
   as numpy arrays, not Python lists, and `bool()` on a multi-element array
   raises `ValueError: truth value of an array... is ambiguous`. Fixed each
   time by switching to `len(x) == 0`.
2. **`entity_embedding.vec` trailing-tab column misalignment** — the file
   has `ENTITY_DIM + 2` fields per line (entity_id + 100 dims + one empty
   trailing field from a trailing tab), not `ENTITY_DIM + 1`. Naming only
   101 columns made pandas silently treat `entity_id` as an index column
   instead, shifting every value over by one — entity IDs came back as
   **floats**, and 0/104,151 articles matched any loaded vector. Caught via
   a debug script before trusting downstream results; confirmed the fix
   with 91,540/104,151 (~88%) matching afterward. Fixed by explicitly
   naming and dropping the trailing column in `load_entity_vectors`.

**Result:** LSA(+entity) **reversed** the earlier pattern and beat BM25F on
AUC (0.5625 vs 0.5579), though BM25F still won MRR/nDCG@5/nDCG@10/
diversity/coverage. **Caveat, stated plainly:** this comparison was run
*after* Revision 4's k1/b tuning was already applied, and no clean
"large-scale, Revision-3-only, default-k1/b" Q4 baseline was preserved (an
earlier attempt at that exact run was killed mid-flight for the entity bug
above) — so Revision 3's isolated effect vs. Revision 4's isolated effect
on this particular AUC reversal isn't separately measured, only their
combination is.

---

## Revision 4 — Empirical k1/b tuning

**Before:** `k1=1.5, b=0.75` — textbook defaults, never validated against
this data.

**Change:** `scripts/tune_bm25.py` — grid search evaluated via restricted-
candidate scoring (AUC/MRR/nDCG@10, the same `score_candidates` path Q4/Q5
use — cheaper and more relevant than Q2/Q3's full-corpus recall@K) on a
fixed-seed 20,000-impression subsample of val, holding the Revision 3
algorithm (BM25F, richer/recency-weighted query) fixed as the base and
varying only k1/b — so this comparison *is* a clean, isolated one.

**Result:** AUC improved monotonically toward `b=1.0` (the formula's
natural ceiling — length normalization is bounded at 1.0 by definition, so
this isn't an open-ended search) and with increasing `k1`, with clearly
diminishing returns (`k1=2.2→3.5` at `b=1.0` added only +0.0019 AUC).
Adopted **k1=3.0, b=1.0** — a defensible plateau point (AUC 0.5581 on the
tuning subsample vs. 0.5528 at textbook defaults, **+0.0053**), not the
literal grid-boundary optimum (`k1=3.5` measured only +0.0007 higher, well
within likely subsample noise).

---

## Revision 5 — Index persistence refactor

**Before:** `evaluate_retrieval.py`, `evaluate_ranking.py`, and
`generate_predictions.py` each independently re-fit BM25F+LSA from raw
text, with hyperparameters (`k1`, `b`, `lsa_components`, `entity_weight`,
...) passed as separate CLI flags per script. Redundant fitting cost was
trivial (~10s), but there was a real risk: if one script's flags/defaults
were updated without updating the others, the three could silently
diverge and still *look* like they were comparing the same method.

**Change:** new `scripts/build_indices.py` fits BM25F+LSA **once** over the
train+val corpus and persists them (`pickle`) to
`data/models/mind/{bm25.pkl, lsa.pkl, config.json}`.
`evaluate_retrieval.py`/`evaluate_ranking.py` now **load that literal
fitted object** instead of refitting — no hyperparameter flags left to
drift on, because there's nothing left to independently specify.
`generate_predictions.py` still must fit its *own* index (it necessarily
scores a different corpus — `MINDlarge_test`'s own articles, not
train+val — so there's no single object to share across a corpus change),
but now reads every hyperparameter from that same `config.json` rather than
its own separately-defaulted flags, so it can't drift from the other two
either.

**Bugs caught and fixed during this revision:**

1. **Unpicklable lambda** — `preprocessor=lambda s: s` inside both
   `CountVectorizer`/`TfidfVectorizer` construction broke `pickle.dump`
   (`AttributeError: Can't pickle local object 'BM25Index.fit.<locals>.<lambda>'`)
   the first time persistence was actually exercised — lambdas/closures
   aren't picklable, only named module-level objects are. Fixed by
   replacing both with a shared named function,
   `text_utils.identity_preprocessor`. Verified with a pickle round-trip on
   synthetic data (dump → load → confirm scores match) before declaring it
   fixed.
2. **NaN vs. `None` null handling** — `BM25Index.fit()` guarded missing
   titles/abstracts with `t or ""`, which misses NaN (`bool(float('nan'))`
   is `True` in Python, so a NaN passed through unchanged and crashed on
   string concatenation). This worked by accident for the train+val corpus
   (parquet/pyarrow represents a missing string as `None`, which *is*
   falsy) but crashed for the test corpus (`generate_predictions.py` reads
   `MINDlarge_test/news.tsv` fresh via `pd.read_csv`, which represents a
   missing string as float `NaN`) — two different null sentinels from two
   different data-loading paths, only one of them falsy. Fixed with an
   explicit `isinstance(x, str)` check, matching the pattern
   `text_utils.article_text()` already used correctly (which is why the
   LSA side of the pipeline never hit this).

**Status:** implemented and unit-verified (pickle round-trip, NaN/None
synthetic test); the user has since re-run `build_indices.py` and
`generate_predictions.py` successfully against the real data after these
fixes. `evaluate_retrieval.py`/`evaluate_ranking.py` have not yet been
re-run against the persisted (Revision 5) index as of this writing.

---

## Revision 6 — SBERT + FAISS semantic backend

**Before:** Q3's only semantic option was LSA (TF-IDF+SVD, optionally
entity-fused).

**Change:** new `retrieval/sbert.py` (`SBERTIndex`) -- a pretrained
sentence-transformer (default `all-MiniLM-L6-v2`, 384-dim) encodes article
text; `faiss.IndexFlatIP` (exact inner-product = exact cosine on
normalized vectors, not approximate -- no latency problem at 100-121K
articles to trade accuracy for) replaces LSA's brute-force dense-matrix
scoring for Q2/Q3's full-corpus retrieval path. Same
`score_candidates`/`get_embedding`/`embeddings`/`id_to_row` surface as
`LSAIndex`, selected via `fit_indices(..., semantic_backend="lsa"|"sbert")`
and persisted/loaded like everything else (Revision 5's mechanism
unchanged, files renamed `lsa.pkl` → `semantic.pkl` since it's
backend-dependent now).

**Bugs caught and fixed:** (1) a loaded `SentenceTransformer` and a
`faiss.Index` aren't reliably picklable -- added `__getstate__`/
`__setstate__` to exclude both from the pickle and rebuild the FAISS index
from `self.embeddings` on load (cheap) rather than let this fail the way
the lambda-preprocessor issue did in Revision 5. (2) Encoding the combined
train+val+test article universe twice (once implicitly via the fit corpus,
once via `embed()` for history lookups) would have doubled SBERT's actual
expensive step -- `embed()` on both `LSAIndex` and `SBERTIndex` now takes
`article_ids` and reuses already-fit embeddings for any id already in the
corpus, only encoding the genuinely novel ones.

**Result:** at large scale, SBERT reversed the Q4 picture again -- now
**decisively** beating BM25F on every accuracy metric (AUC 0.5860 vs.
0.5579, non-overlapping CIs, holds in both cold/warm slices), a much larger
and cleaner margin than entity-fused LSA's earlier narrow win. But Q2/Q3's
full-corpus recall@K flipped the *other* way -- BM25F now wins at every K,
every slice (e.g. K=200: 3.09% vs. 1.97%). Not a contradiction: Q2/Q3 asks
"does the one exact clicked article land in the top-K out of the entire
104K-article catalog" (SBERT's embeddings cluster topically-similar
articles together, which crowds out the one exact match for this
needle-in-haystack framing); Q4 asks "rerank this specific small candidate
list well" (the task Codabench actually scores, and where SBERT's semantic
understanding pays off). Recommendation followed: submit the SBERT
prediction file, on the strength of the Q4 result, not Q2/Q3's.

---

## Revision 7 — EB-NeRD (Q1-Q5), ebnerd_small train/val + ebnerd_testset predictions

**Before:** MIND only. `pipeline/schema.py`'s unified schema, and every
retrieval/eval module (`retrieval/*.py`), were already dataset-agnostic by
design -- nothing in them assumes MIND specifically, only the *loaders*
(`pipeline/mind.py`) and the *scripts'* hardcoded `data/.../mind/` paths
did. That design paid off directly here.

**Change:** `pipeline/ebnerd.py` (new) maps `ebnerd_small`'s
`articles.parquet` + `{train,validation}/{behaviors,history}.parquet` into
the same unified schema MIND uses -- `title`←title, `abstract`←`subtitle`,
`category`←`category_str`. Unlike MIND, EB-NeRD's `history.parquet` already
carries per-click timestamps directly (no need to derive click history from
positive-labelled impressions), and behaviors don't embed history inline
(joined by `user_id` instead). `train`/`validation` are natively temporally
disjoint (train ends 2023-05-25 07:00, validation starts the same instant)
-- same "use the dataset's own files as the split" principle as MIND.
Verified end-to-end against the real `ebnerd_small` data (477,534
interactions, split-boundary check passed, leakage check passed) before
handing off.

`build_pipeline.py`, `scripts/{build_indices,evaluate_retrieval,
evaluate_ranking,tune_bm25}.py` all gained a `--dataset {mind,ebnerd}` flag
that only changes path prefixes (`data/processed/<dataset>/`, `.../models/
<dataset>/`, `.../reports/<dataset>_*.json`) -- zero logic changes, exactly
because the retrieval/eval code never assumed MIND's schema specifically.

Two EB-NeRD-specific decisions, not just path routing:
- **No entity fusion.** EB-NeRD's `ner_clusters`/`entity_groups` are named-
  entity surface strings ("Willy Strube"), not Wikidata IDs with a matching
  pretrained embedding table the way MIND's TransE vectors are -- there is
  no analogous file to fuse. `entity_vectors` stays `None` throughout.
- **Multilingual SBERT model.** EB-NeRD is Danish; `all-MiniLM-L6-v2` (the
  MIND default) is effectively English-only, so using it on Danish text
  would be a real quality bug, not a suboptimal default.
  `scripts/build_indices.py` auto-selects
  `paraphrase-multilingual-MiniLM-L12-v2` for `--dataset ebnerd` unless
  `--sbert_model` is given explicitly.

**Q5 got its own script** (`scripts/generate_predictions_ebnerd.py`), not a
branch inside the MIND one -- the raw format genuinely differs enough to
warrant it: parquet instead of TSV (chunked via `pyarrow.parquet.
ParquetFile.iter_batches`, since pandas has no `chunksize` for parquet, the
parquet analogue of MIND's CSV-chunksize streaming), and history that must
be joined from a *separate* `history.parquet` by `user_id` rather than read
inline. The unlabeled test set (`ebnerd_testset`) is also ~5.7x larger than
MIND's (13.5M vs. 2.37M impressions) and is deliberately *not* folded into
`build_pipeline.py`'s own `interactions.parquet` the way MIND's test split
is (harmless there, but nothing downstream actually reads that portion of
MIND's back out either -- Q5 always streams the raw test file directly) --
EB-NeRD's per-impression history join would make that inclusion
meaningfully more expensive for no benefit.

**Bug caught and fixed before handoff:** an early version of the article
loader unioned `ner_clusters` (actual entity-mention strings) with
`entity_groups` (their parallel per-mention *type* labels, e.g. "PER"/
"ORG") into one `entities` list, producing rows like `["Willy Strube",
"PER"]` -- a name and a category label sitting in the same field. Currently
inert (nothing consumes `entities` for EB-NeRD, no fusion path exists), but
wrong enough to fix rather than leave for a future feature to trip over.
Fixed to keep only the entity-mention strings.

**Not yet run:** `scripts/build_indices.py --dataset ebnerd`,
`evaluate_retrieval.py`/`evaluate_ranking.py --dataset ebnerd`, and
`generate_predictions_ebnerd.py` -- implemented and (for the Q1 loader)
verified against real data, but the full Q2-Q5 sweep hasn't been executed
yet. k1/b tuned for MIND (`k1=3.0, b=1.0`) should not be assumed to
transfer -- EB-NeRD's candidate lists are much shorter on average (~11.5
vs. MIND's ~37) and history much longer (~292 vs. ~30-40), a different
enough shape that re-running `scripts/tune_bm25.py --dataset ebnerd` before
trusting a k1/b choice is the right call, not an optional nicety.

---

## Currently open / not yet done

- `data/reports/mind_retrieval_eval.json` (Q2/Q3 full-corpus recall@K)
  still holds **Revision 2** numbers (large-scale, pre-Revision-3
  algorithm) — the Revision 3+4+5 combined re-run was interrupted and not
  yet redone.
- No isolated measurement of Revision 3's algorithm changes alone (without
  Revision 4's tuning) at large scale — see the caveat in Revision 3.
- `README.md` has not been updated to reflect Revisions 2–5 yet.
- Q6 (a standalone ≤4-page design note document, distinct from this
  changelog) and EB-NeRD are still not implemented — see `README.md`'s own
  "Not yet done" list.
