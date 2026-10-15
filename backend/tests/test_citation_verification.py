from app.services.generation.context_selector import ContextChunk
from app.services.verification.citation_verification import Verdict, verify_claim, verify_claims
from app.services.verification.claim_extraction import Claim


def chunk(marker, chunk_id, text, doc_id="d1", filename="handbook.pdf", page=1):
    return ContextChunk(
        marker=marker, chunk_id=chunk_id, document_id=doc_id, filename=filename,
        page_number=page, section_title=None, text=text,
    )


def scripted_nli(score_table):
    def nli_fn(pairs):
        return [score_table[(premise, hyp)] for premise, hyp in pairs]
    return nli_fn


def test_high_entailment_yields_supported():
    ctx = {1: chunk(1, 101, "Product X supports MySQL and PostgreSQL.")}
    claim = Claim(claim_id="c1", text="Product X supports PostgreSQL.", source_markers=frozenset({1}))
    nli_fn = scripted_nli({
        ("Product X supports MySQL and PostgreSQL.", "Product X supports PostgreSQL."):
            {"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04}
    })
    result = verify_claim(claim, ctx, nli_fn=nli_fn)
    assert result.verdict == Verdict.SUPPORTED
    assert result.support_score == 0.95
    assert result.evidence.chunk_id == 101


def test_unrelated_evidence_yields_unsupported():
    ctx = {1: chunk(1, 101, "Product X supports MySQL and PostgreSQL.")}
    claim = Claim(claim_id="c1", text="Product X supports MongoDB.", source_markers=frozenset({1}))
    nli_fn = scripted_nli({
        ("Product X supports MySQL and PostgreSQL.", "Product X supports MongoDB."):
            {"entailment": 0.05, "contradiction": 0.10, "neutral": 0.85}
    })
    result = verify_claim(claim, ctx, nli_fn=nli_fn)
    assert result.verdict == Verdict.UNSUPPORTED


def test_explicit_contradiction_overrides_middling_entailment():
    ctx = {1: chunk(1, 101, "Product X does not support MongoDB.")}
    claim = Claim(claim_id="c1", text="Product X supports MongoDB.", source_markers=frozenset({1}))
    nli_fn = scripted_nli({
        ("Product X does not support MongoDB.", "Product X supports MongoDB."):
            {"entailment": 0.40, "contradiction": 0.55, "neutral": 0.05}
    })
    result = verify_claim(claim, ctx, nli_fn=nli_fn)
    assert result.verdict == Verdict.UNSUPPORTED


def test_mid_range_entailment_yields_partially_supported():
    ctx = {1: chunk(1, 101, "Some evidence.")}
    claim = Claim(claim_id="c1", text="A related claim.", source_markers=frozenset({1}))
    nli_fn = scripted_nli({
        ("Some evidence.", "A related claim."): {"entailment": 0.50, "contradiction": 0.10, "neutral": 0.40}
    })
    result = verify_claim(claim, ctx, nli_fn=nli_fn)
    assert result.verdict == Verdict.PARTIALLY_SUPPORTED


def test_claim_with_no_citation_is_checked_against_all_context():
    ctx = {
        1: chunk(1, 101, "Irrelevant passage."),
        2: chunk(2, 102, "Product X supports PostgreSQL."),
    }
    claim = Claim(claim_id="c1", text="Product X supports PostgreSQL.", source_markers=frozenset())
    nli_fn = scripted_nli({
        ("Irrelevant passage.", "Product X supports PostgreSQL."):
            {"entailment": 0.02, "contradiction": 0.02, "neutral": 0.96},
        ("Product X supports PostgreSQL.", "Product X supports PostgreSQL."):
            {"entailment": 0.98, "contradiction": 0.0, "neutral": 0.02},
    })
    result = verify_claim(claim, ctx, nli_fn=nli_fn)
    assert result.verdict == Verdict.SUPPORTED
    assert result.evidence.chunk_id == 102


def test_claim_citing_a_marker_not_in_context_has_no_evidence():
    ctx = {1: chunk(1, 101, "Some text.")}
    claim = Claim(claim_id="c1", text="A claim.", source_markers=frozenset({99}))
    result = verify_claim(claim, ctx, nli_fn=lambda pairs: [])
    assert result.verdict == Verdict.UNSUPPORTED
    assert result.evidence is None
    assert result.support_score == 0.0


def test_empty_context_has_no_evidence():
    claim = Claim(claim_id="c1", text="A claim.", source_markers=frozenset())
    result = verify_claim(claim, {}, nli_fn=lambda pairs: [])
    assert result.verdict == Verdict.UNSUPPORTED
    assert result.evidence is None


def test_verify_claims_processes_each_claim_independently():
    ctx = {1: chunk(1, 101, "Evidence text.")}
    claims = [
        Claim(claim_id="c1", text="Claim A.", source_markers=frozenset({1})),
        Claim(claim_id="c2", text="Claim B.", source_markers=frozenset({1})),
    ]
    nli_fn = scripted_nli({
        ("Evidence text.", "Claim A."): {"entailment": 0.9, "contradiction": 0.0, "neutral": 0.1},
        ("Evidence text.", "Claim B."): {"entailment": 0.1, "contradiction": 0.0, "neutral": 0.9},
    })
    results = verify_claims(claims, ctx, nli_fn=nli_fn)
    assert results[0].verdict == Verdict.SUPPORTED
    assert results[1].verdict == Verdict.UNSUPPORTED
