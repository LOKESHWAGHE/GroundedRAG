"""
Parses the raw LLM completion into a structured result: the answer text, the
set of citation markers ("[1]", "[2][4]") actually used, and whether the
model declared the evidence insufficient. Pure string parsing — no network,
no model — so it's fully unit-testable.
"""

import re
from dataclasses import dataclass

_MARKER_RE = re.compile(r"\[(\d+)\]")
_INSUFFICIENT_PREFIX = "INSUFFICIENT_EVIDENCE:"


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str
    cited_markers: set[int]  # markers the model actually referenced, e.g. {1, 3}
    insufficient_evidence: bool
    insufficient_reason: str | None = None


def parse_llm_response(raw_text: str) -> ParsedAnswer:
    text = raw_text.strip()

    if text.startswith(_INSUFFICIENT_PREFIX):
        reason = text[len(_INSUFFICIENT_PREFIX) :].strip() or "Not specified by the model."
        return ParsedAnswer(answer=text, cited_markers=set(), insufficient_evidence=True, insufficient_reason=reason)

    markers = {int(m) for m in _MARKER_RE.findall(text)}
    return ParsedAnswer(answer=text, cited_markers=markers, insufficient_evidence=False)
