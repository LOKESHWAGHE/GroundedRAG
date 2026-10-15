"""
Splits a chunk's text into sentence-based windows for NLI scoring.

Why not feed the whole chunk to the NLI model as one premise: a ~180-word
chunk often contains several unrelated sentences, and NLI entailment
degrades when the premise is longer and noisier than the claim — a single
supporting sentence can get diluted by the rest of the chunk. Windowing lets
verification find the specific sentence(s) that actually support (or
contradict) a claim, which is also what the debug panel and citation UI show
as the "supporting text" for a claim.
"""

import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class EvidenceWindow:
    text: str
    window_index: int  # position within the chunk, for the debug panel


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def build_evidence_windows(chunk_text: str, max_windows: int, window_size_sentences: int = 2) -> list[EvidenceWindow]:
    """
    Slides a `window_size_sentences`-sentence window over the chunk. If that
    would produce more than `max_windows`, evenly samples down to that many
    so a long chunk still costs a bounded number of NLI calls.
    """
    sentences = _split_sentences(chunk_text)
    if not sentences:
        return []
    if len(sentences) <= window_size_sentences:
        return [EvidenceWindow(text=" ".join(sentences), window_index=0)]

    all_windows = [
        " ".join(sentences[i : i + window_size_sentences]) for i in range(0, len(sentences), window_size_sentences)
    ]
    if len(all_windows) <= max_windows:
        return [EvidenceWindow(text=w, window_index=i) for i, w in enumerate(all_windows)]

    # Evenly sample `max_windows` indices across the full window list, always including the first and last.
    step = (len(all_windows) - 1) / (max_windows - 1) if max_windows > 1 else 0
    indices = sorted({round(i * step) for i in range(max_windows)})
    return [EvidenceWindow(text=all_windows[i], window_index=i) for i in indices]
