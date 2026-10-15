"""
Splits the generated answer into (sentence, citation_markers) pairs.

The answer text looks like: "Product X supports PostgreSQL. [1] It also
supports MySQL. [2][3]" — this keeps each sentence paired with whichever
marker(s) immediately followed it, since that's the LLM's claimed evidence
for that specific sentence (still unverified at this point — Phase 10 checks
it against the actual chunk text).
"""

import re
from dataclasses import dataclass

_UNIT_RE = re.compile(r"(.+?[.!?])(\s*(?:\[\d+\])+)?(?=\s+|$)")
_MARKER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class SentenceUnit:
    text: str  # marker-free
    markers: frozenset[int]


def split_sentences_with_markers(answer_text: str) -> list[SentenceUnit]:
    text = answer_text.strip()
    if not text:
        return []

    units: list[SentenceUnit] = []
    pos = 0
    for match in _UNIT_RE.finditer(text):
        if match.start() != pos:  # guards against the regex silently skipping malformed text
            break
        sentence, marker_group = match.group(1).strip(), match.group(2) or ""
        markers = frozenset(int(m) for m in _MARKER_RE.findall(marker_group))
        if sentence:
            units.append(SentenceUnit(text=sentence, markers=markers))
        pos = match.end()

    remainder = text[pos:].strip()
    if remainder:  # trailing text with no terminal punctuation — still a claim worth checking
        units.append(SentenceUnit(text=remainder, markers=frozenset()))
    return units
