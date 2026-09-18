"""A2 Q2: per-impression Assignment-1 retrieval scores (BM25F, semantic,
fusion), reusable as re-ranker input features -- "use Assignment 1's
candidate generator" from the Q2 spec.

This is the exact same per-impression scoring logic scripts/evaluate_ranking.py
already runs (query construction -> score_candidates -> fuse_scores),
extracted into one function so it isn't duplicated a third time now that
the re-ranker needs it too. Deliberately lives in `retrieval/`, not
`pipeline/features_*.py`: these are Assignment-1 retrieval scores, not new
A2 behavioural features, and are dataset-agnostic once given a fitted
BM25Index/semantic index (LSAIndex/SBERTIndex) -- unlike the genuinely
dataset-specific Q1 feature sets, there's nothing MIND/EB-NeRD-specific
about how a lexical/semantic/fusion score is computed.
"""

from retrieval.fusion import fuse_scores
from retrieval.lsa import mean_pool_user_vector
from retrieval.text_utils import recency_weights, weighted_query_entities, weighted_query_terms


def compute_retrieval_scores(history_ids, candidate_ids, bm25, semantic, text_lookup, entity_lookup,
                               recent_n=20, recency_decay=0.85, fusion_alpha=0.7):
    """Returns (bm25_scores, semantic_scores, fusion_scores), each an array
    aligned with `candidate_ids` -- identical semantics to what
    scripts/evaluate_ranking.py computes and reports per impression, just
    factored out so scripts/train_reranker.py can reuse it verbatim rather
    than re-deriving query construction independently (and risking it
    silently drifting out of sync with what Q4's own numbers reflect)."""
    q_weights = weighted_query_terms(history_ids, text_lookup, recent_n=recent_n, decay=recency_decay)
    q_entities = weighted_query_entities(history_ids, entity_lookup, recent_n=recent_n, decay=recency_decay)
    bm25_scores = bm25.score_candidates(q_weights, candidate_ids, query_entity_weights=q_entities)

    recent = history_ids[-recent_n:] if len(history_ids) else []
    embs = [semantic.get_embedding(a) for a in recent]
    w = recency_weights(len(recent), recency_decay) if len(recent) else None
    user_vec = mean_pool_user_vector(embs, weights=w)
    semantic_scores = semantic.score_candidates(user_vec, candidate_ids)

    fusion_scores = fuse_scores(bm25_scores, semantic_scores, alpha=fusion_alpha)
    return bm25_scores, semantic_scores, fusion_scores
