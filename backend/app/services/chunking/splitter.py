"""
Structure-aware chunking.

Why not fixed-size splitting: cutting text at a fixed word count regardless of
structure routinely slices a sentence (or a table row) in half, which both
hurts embedding quality (the chunk is no longer a coherent unit of meaning)
and makes citations point at a fragment that doesn't read sensibly on its own.

Strategy ("structure", the default):
  1. Split on blank lines into paragraphs; a markdown heading line ("#"..."######")
     starts a new section and is tracked as `section_title` for every chunk under it.
  2. Pack consecutive paragraphs into a chunk up to CHUNK_SIZE_TOKENS words.
  3. A paragraph longer than CHUNK_SIZE_TOKENS on its own is split on sentence
     boundaries (falls back to a hard word split if a single "sentence" is
     still too long, e.g. unbroken text with no punctuation).
  4. Consecutive chunks overlap by CHUNK_OVERLAP_TOKENS words of trailing
     context, so a fact split across a chunk boundary is still retrievable
     from whichever side the query's embedding lands closer to.

Strategy "fixed" (kept only as an evaluation baseline — Experiment 5 compares
the two): ignores structure, slides a fixed-size word window over the raw text.
"""

import re
from dataclasses import dataclass, field

from app.services.chunking.tokenizer import count_words, split_words

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class RawChunk:
    text: str
    section_title: str | None
    token_count: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "token_count", count_words(self.text))


def _split_into_paragraphs(page_text: str) -> list[tuple[str, str | None]]:
    """Return (paragraph_text, current_section_title) pairs, tracking heading context."""
    current_section: str | None = None
    out: list[tuple[str, str | None]] = []
    for block in re.split(r"\n\s*\n", page_text):
        block = block.strip()
        if not block:
            continue
        heading = _HEADING_RE.match(block)
        if heading:
            current_section = heading.group(2).strip()
            continue  # a heading line becomes context, not its own chunk
        out.append((block, current_section))
    return out


def _split_long_paragraph(text: str, max_words: int) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces, buf, buf_words = [], [], 0
    for sent in sentences:
        w = count_words(sent)
        if w > max_words:  # a single "sentence" is still too long: hard-split on words
            if buf:
                pieces.append(" ".join(buf))
                buf, buf_words = [], 0
            words = split_words(sent)
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i : i + max_words]))
            continue
        if buf_words + w > max_words and buf:
            pieces.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(sent)
        buf_words += w
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def chunk_page_structure_aware(
    page_text: str, chunk_size: int, overlap: int, min_tokens: int
) -> list[RawChunk]:
    paragraphs = _split_into_paragraphs(page_text)
    units: list[tuple[str, str | None]] = []
    for para_text, section in paragraphs:
        if count_words(para_text) > chunk_size:
            units.extend((piece, section) for piece in _split_long_paragraph(para_text, chunk_size))
        else:
            units.append((para_text, section))

    chunks: list[RawChunk] = []
    buf_units: list[tuple[str, str | None]] = []
    buf_words = 0

    def flush() -> str | None:
        nonlocal buf_units, buf_words
        if not buf_units:
            return None
        text = " ".join(u[0] for u in buf_units)
        section = next((u[1] for u in buf_units if u[1]), None)
        if count_words(text) >= min_tokens:
            chunks.append(RawChunk(text=text, section_title=section))
        carried = "" if overlap <= 0 else _tail_words(text, overlap)
        buf_units = [(carried, section)] if carried else []
        buf_words = count_words(carried)
        return text

    for unit_text, section in units:
        w = count_words(unit_text)
        if buf_words + w > chunk_size and buf_units:
            flush()
        buf_units.append((unit_text, section))
        buf_words += w
    flush()
    return chunks


def _tail_words(text: str, n: int) -> str:
    words = split_words(text)
    return " ".join(words[-n:]) if len(words) > n else ""


def chunk_page_fixed(page_text: str, chunk_size: int, overlap: int, min_tokens: int) -> list[RawChunk]:
    """Baseline strategy: a plain sliding word window, ignoring paragraph/heading structure."""
    words = split_words(page_text)
    if not words:
        return []
    stride = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), stride):
        piece = words[start : start + chunk_size]
        if len(piece) >= min_tokens:
            chunks.append(RawChunk(text=" ".join(piece), section_title=None))
        if start + chunk_size >= len(words):
            break
    return chunks
