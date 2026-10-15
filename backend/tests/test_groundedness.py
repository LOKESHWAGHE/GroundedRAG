from app.services.verification.citation_verification import ClaimVerdict, SupportingEvidence, Verdict
from app.services.verification.claim_extraction import Claim
from app.services.verification.groundedness import compute_groundedness, has_unsupported_claims


def verdict(verdict_enum, cited=True):
    claim = Claim(claim_id="c", text="x", source_markers=frozenset({1}) if cited else frozenset())
    ev = SupportingEvidence(1, "d", "f.pdf", 1, "q", 0.9, 0.0) if verdict_enum != Verdict.UNSUPPORTED else None
    return ClaimVerdict(claim=claim, verdict=verdict_enum, support_score=0.9, evidence=ev)


def test_matches_the_spec_worked_example():
    verdicts = [verdict(Verdict.SUPPORTED) for _ in range(7)] + [verdict(Verdict.PARTIALLY_SUPPORTED)]
    report = compute_groundedness(verdicts)
    assert report.total_claims == 8
    assert report.supported == 7
    assert report.partially_supported == 1
    assert report.unsupported == 0
    assert abs(report.groundedness_score - 0.9375) < 1e-9


def test_all_supported_is_perfect_score():
    report = compute_groundedness([verdict(Verdict.SUPPORTED) for _ in range(5)])
    assert report.groundedness_score == 1.0


def test_all_unsupported_is_zero_score():
    report = compute_groundedness([verdict(Verdict.UNSUPPORTED) for _ in range(3)])
    assert report.groundedness_score == 0.0


def test_partial_gets_half_credit():
    report = compute_groundedness([verdict(Verdict.PARTIALLY_SUPPORTED)])
    assert report.groundedness_score == 0.5


def test_no_claims_yields_none_not_a_divide_by_zero():
    report = compute_groundedness([])
    assert report.groundedness_score is None
    assert report.citation_coverage is None
    assert report.total_claims == 0


def test_citation_coverage_counts_claims_with_a_marker():
    verdicts = [verdict(Verdict.SUPPORTED, cited=True), verdict(Verdict.SUPPORTED, cited=False)]
    report = compute_groundedness(verdicts)
    assert report.citation_coverage == 0.5


def test_has_unsupported_claims_true_and_false_cases():
    assert has_unsupported_claims([verdict(Verdict.SUPPORTED), verdict(Verdict.UNSUPPORTED)]) is True
    assert has_unsupported_claims([verdict(Verdict.SUPPORTED), verdict(Verdict.PARTIALLY_SUPPORTED)]) is False
    assert has_unsupported_claims([]) is False
