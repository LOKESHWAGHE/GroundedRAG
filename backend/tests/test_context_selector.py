from dataclasses import dataclass

from app.services.generation.context_selector import build_context


@dataclass
class FakeReranked:
    chunk_id: int
    text: str


def meta(doc_id="d1", filename="a.pdf", page=1, section=None):
    return {"document_id": doc_id, "filename": filename, "page_number": page, "section_title": section}


def test_markers_are_1_indexed_in_order():
    reranked = [FakeReranked(1, "alpha"), FakeReranked(2, "beta")]
    chunk_meta = {1: meta(), 2: meta()}
    ctx = build_context(reranked, chunk_meta, max_words=1000)
    assert [c.marker for c in ctx.chunks] == [1, 2]
    assert [c.chunk_id for c in ctx.chunks] == [1, 2]


def test_prompt_text_includes_filename_page_and_marker():
    reranked = [FakeReranked(1, "the answer text")]
    chunk_meta = {1: meta(filename="handbook.pdf", page=12)}
    ctx = build_context(reranked, chunk_meta, max_words=1000)
    assert "[1]" in ctx.prompt_text
    assert "handbook.pdf" in ctx.prompt_text
    assert "page 12" in ctx.prompt_text
    assert "the answer text" in ctx.prompt_text


def test_section_title_included_when_present():
    reranked = [FakeReranked(1, "text")]
    chunk_meta = {1: meta(section="Refund Policy")}
    ctx = build_context(reranked, chunk_meta, max_words=1000)
    assert "Refund Policy" in ctx.prompt_text


def test_budget_truncates_but_always_keeps_at_least_one_chunk():
    long_text = " ".join(f"word{i}" for i in range(500))
    reranked = [FakeReranked(1, long_text), FakeReranked(2, "short")]
    chunk_meta = {1: meta(), 2: meta()}
    ctx = build_context(reranked, chunk_meta, max_words=10)
    assert len(ctx.chunks) == 1
    assert ctx.truncated is True


def test_no_truncation_when_everything_fits():
    reranked = [FakeReranked(1, "short one"), FakeReranked(2, "short two")]
    chunk_meta = {1: meta(), 2: meta()}
    ctx = build_context(reranked, chunk_meta, max_words=1000)
    assert ctx.truncated is False
    assert len(ctx.chunks) == 2


def test_empty_reranked_list_yields_empty_context():
    ctx = build_context([], {}, max_words=1000)
    assert ctx.chunks == []
    assert ctx.prompt_text == ""
    assert ctx.truncated is False
