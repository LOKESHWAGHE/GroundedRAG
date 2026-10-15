"""
Turns the reranked chunks into the numbered context block sent to the LLM.
Each chunk gets a stable [n] marker that the model is instructed to cite by —
those markers are later resolved back to real chunk/document/page metadata,
and (Phase 10) independently checked rather than trusted at face value.

A word budget (CONTEXT_MAX_WORDS) is enforced here, not left to the prompt
template, because RERANK_TOP_M chunks can still add up to more than a
reasonable prompt size once a few of them are long.
"""

from dataclasses import dataclass

from app.core.config import get_settings
from app.services.chunking.tokenizer import count_words
from app.services.reranking.service import RerankedResult


@dataclass(frozen=True)
class ContextChunk:
    marker: int  # 1-indexed citation number shown to the LLM and the user, e.g. "[2]"
    chunk_id: int
    document_id: str
    filename: str
    page_number: int
    section_title: str | None
    text: str


@dataclass(frozen=True)
class ContextBundle:
    chunks: list[ContextChunk]
    prompt_text: str
    truncated: bool  # True if the word budget cut off candidates that were available


def build_context(
    reranked: list[RerankedResult],
    chunk_meta: dict[int, dict],
    max_words: int | None = None,
) -> ContextBundle:
    """
    `chunk_meta[chunk_id]` must provide: document_id, filename, page_number,
    section_title. Kept as a plain dict (rather than importing the ORM Chunk/
    Document models) so this stays testable without a database.
    """
    max_words = max_words or get_settings().CONTEXT_MAX_WORDS

    selected: list[ContextChunk] = []
    used_words = 0
    truncated = False
    for i, r in enumerate(reranked, start=1):
        meta = chunk_meta[r.chunk_id]
        words = count_words(r.text)
        if selected and used_words + words > max_words:
            truncated = True
            break
        selected.append(
            ContextChunk(
                marker=i,
                chunk_id=r.chunk_id,
                document_id=str(meta["document_id"]),
                filename=meta["filename"],
                page_number=meta["page_number"],
                section_title=meta.get("section_title"),
                text=r.text,
            )
        )
        used_words += words

    blocks = []
    for c in selected:
        header = f"[{c.marker}] {c.filename}, page {c.page_number}"
        if c.section_title:
            header += f", section \"{c.section_title}\""
        blocks.append(f"{header}\n{c.text}")
    prompt_text = "\n\n".join(blocks)

    return ContextBundle(chunks=selected, prompt_text=prompt_text, truncated=truncated)
