"""
Word-based tokenizer used only for chunk sizing (not for the embedding model's
own tokenizer). Whitespace-splitting is a cheap, dependency-free proxy for
token count that is consistent between chunking and the CHUNK_SIZE_TOKENS /
CONTEXT_MAX_WORDS budgets used elsewhere.
"""

import re

_WORD_RE = re.compile(r"\S+")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def split_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)
