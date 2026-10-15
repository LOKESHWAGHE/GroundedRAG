"""
Aggregates per-claim verdicts into response-level evaluation metrics.

groundedness = (supported + 0.5 * partially_supported) / total_claims

This gives partial credit for partially-supported claims rather than the
simpler `supported / total` ratio the project brief sketches as a starting
point — a response with mostly-SUPPORTED and a few PARTIALLY_SUPPORTED
claims is meaningfully better than one with the same number of fully
UNSUPPORTED claims, and a strict supported-only ratio would score them
identically. These are evaluation metrics, not a guarantee of factual
correctness — stated explicitly in the API response, not just here.

citation_coverage = fraction of claims that had at least one citation marker
attached by the model (whether or not that citation held up under
verification) — a measure of whether the model is behaving like a
citation-grounded system at all, separate from whether its citations were
accurate.
"""

from dataclasses import dataclass

from app.services.verification.citation_verification import ClaimVerdict, Verdict


@dataclass(frozen=True)
class GroundednessReport:
    total_claims: int
    supported: int
    partially_supported: int
    unsupported: int
    groundedness_score: float | None  # None when there were no claims to score (e.g. insufficient-evidence answer)
    citation_coverage: float | None


def compute_groundedness(verdicts: list[ClaimVerdict]) -> GroundednessReport:
    total = len(verdicts)
    if total == 0:
        return GroundednessReport(0, 0, 0, 0, groundedness_score=None, citation_coverage=None)

    supported = sum(1 for v in verdicts if v.verdict == Verdict.SUPPORTED)
    partial = sum(1 for v in verdicts if v.verdict == Verdict.PARTIALLY_SUPPORTED)
    unsupported = total - supported - partial
    cited = sum(1 for v in verdicts if v.claim.source_markers)

    return GroundednessReport(
        total_claims=total,
        supported=supported,
        partially_supported=partial,
        unsupported=unsupported,
        groundedness_score=(supported + 0.5 * partial) / total,
        citation_coverage=cited / total,
    )


def has_unsupported_claims(verdicts: list[ClaimVerdict]) -> bool:
    return any(v.verdict == Verdict.UNSUPPORTED for v in verdicts)
