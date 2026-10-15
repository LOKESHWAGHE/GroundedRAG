from app.evaluation.metrics import (
    citation_accuracy,
    mean,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    stddev,
    unsupported_claim_rate,
)


def test_recall_at_k_all_relevant_found():
    assert recall_at_k([1, 2, 3], {1, 2}, k=3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k([1, 5, 6], {1, 2}, k=3) == 0.5


def test_recall_at_k_respects_k_cutoff():
    # relevant doc 2 is retrieved but only at rank 4, outside k=3
    assert recall_at_k([5, 6, 7, 2], {2}, k=3) == 0.0
    assert recall_at_k([5, 6, 7, 2], {2}, k=4) == 1.0


def test_recall_at_k_no_relevant_docs_is_zero_not_divide_by_zero():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_precision_at_k():
    assert precision_at_k([1, 2, 3], {1, 2}, k=3) == 2 / 3


def test_precision_at_k_empty_retrieved_is_zero():
    assert precision_at_k([], {1}, k=5) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([1, 2, 3], {1}) == 1.0


def test_reciprocal_rank_third_position():
    assert abs(reciprocal_rank([5, 6, 1], {1}) - 1 / 3) < 1e-9


def test_reciprocal_rank_no_match_is_zero():
    assert reciprocal_rank([5, 6, 7], {1}) == 0.0


def test_mean_reciprocal_rank_averages_across_queries():
    retrieved = [[1, 2], [5, 6, 3]]
    relevant = [{1}, {3}]
    # query 1: rank 1 -> RR=1.0 ; query 2: rank 3 -> RR=1/3
    assert abs(mean_reciprocal_rank(retrieved, relevant) - (1.0 + 1 / 3) / 2) < 1e-9


def test_mean_reciprocal_rank_empty_input_is_zero():
    assert mean_reciprocal_rank([], []) == 0.0


def test_mean_of_values():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0


def test_stddev_of_constant_values_is_zero():
    assert stddev([5.0, 5.0, 5.0]) == 0.0


def test_stddev_single_value_is_zero():
    assert stddev([5.0]) == 0.0


def test_citation_accuracy_counts_supported_fraction():
    verdicts = [["SUPPORTED", "SUPPORTED", "UNSUPPORTED"], ["SUPPORTED"]]
    assert citation_accuracy(verdicts) == 3 / 4


def test_citation_accuracy_empty_is_zero():
    assert citation_accuracy([]) == 0.0
    assert citation_accuracy([[]]) == 0.0


def test_unsupported_claim_rate():
    verdicts = [["SUPPORTED", "UNSUPPORTED"], ["UNSUPPORTED", "UNSUPPORTED"]]
    assert unsupported_claim_rate(verdicts) == 3 / 4
