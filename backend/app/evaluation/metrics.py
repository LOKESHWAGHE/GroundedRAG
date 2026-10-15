"""
Retrieval and groundedness metrics used by evaluate.py. Each function takes
plain Python data (lists of ids, verdict strings) rather than ORM objects or
pipeline dataclasses, so metrics can be computed the same way whether the
evaluation is run against the real pipeline or against fixture data in tests.
"""

import math


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of relevant chunks that appear anywhere in the top k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of the top k retrieved chunks that are actually relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """1 / rank of the first relevant chunk; 0.0 if none of the retrieved chunks are relevant."""
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(per_query_retrieved: list[list[int]], per_query_relevant: list[set[int]]) -> float:
    if not per_query_retrieved:
        return 0.0
    scores = [reciprocal_rank(r, rel) for r, rel in zip(per_query_retrieved, per_query_relevant)]
    return sum(scores) / len(scores)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def citation_accuracy(verdicts_per_query: list[list[str]]) -> float:
    """
    Fraction of all claims, across all evaluated queries, whose verdict is
    SUPPORTED. This is the headline "is the system's citation behavior
    trustworthy" number — distinct from groundedness_score, which gives
    partial credit for PARTIALLY_SUPPORTED claims.
    """
    all_verdicts = [v for query_verdicts in verdicts_per_query for v in query_verdicts]
    if not all_verdicts:
        return 0.0
    return sum(1 for v in all_verdicts if v == "SUPPORTED") / len(all_verdicts)


def unsupported_claim_rate(verdicts_per_query: list[list[str]]) -> float:
    all_verdicts = [v for query_verdicts in verdicts_per_query for v in query_verdicts]
    if not all_verdicts:
        return 0.0
    return sum(1 for v in all_verdicts if v == "UNSUPPORTED") / len(all_verdicts)


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
