"""
Text cleaning applied to every page after parsing.

Goal: remove extraction noise that would hurt retrieval, WITHOUT destroying
structure the chunker needs later (paragraph breaks, markdown headings, list
indentation). So we deliberately do NOT re-flow lines into paragraphs here.
"""

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # keeps \t and \n
_HYPHEN_LINEBREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")  # "inter-\nnational"
_INLINE_SPACES = re.compile(r"(?<=\S)[ \t]+(?=\S)")  # runs of spaces *inside* a line
_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    # NFKC folds ligatures (ﬁ -> fi), non-breaking spaces, full-width forms.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")  # soft hyphens
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_LINEBREAK.sub("", text)  # trade-off: "well-\nknown" becomes "wellknown"
    text = _INLINE_SPACES.sub(" ", text)  # leading indentation is preserved (markdown lists/code)
    text = _TRAILING_SPACES.sub("", text)
    text = _EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()
