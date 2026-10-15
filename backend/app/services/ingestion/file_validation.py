"""
Upload validation. Everything a client sends is untrusted, so we check:
  1. the filename (sanitised; it is only ever used for display, never as a disk path)
  2. the extension against an allow-list
  3. the size (enforced while reading, so a huge upload can't fill memory)
  4. the actual bytes ("magic numbers") against the claimed extension
"""

import re
import unicodedata
from pathlib import PurePath
from typing import BinaryIO

_UNSAFE_CHARS = re.compile(r"[^\w.\- ]", re.UNICODE)
_MAX_FILENAME_LEN = 255


class FileValidationError(Exception):
    """Raised for any rejected upload. `status_code` maps directly to the HTTP response."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def sanitize_filename(raw: str | None) -> str:
    """Strip directories and unsafe characters. `../../etc/passwd` -> `passwd`."""
    if not raw or not raw.strip():
        raise FileValidationError("Missing filename.")
    name = PurePath(raw.replace("\\", "/")).name  # drops any directory components
    name = unicodedata.normalize("NFKC", name)
    name = _UNSAFE_CHARS.sub("_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        raise FileValidationError("Filename is empty after sanitisation.")
    if len(name) > _MAX_FILENAME_LEN:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: _MAX_FILENAME_LEN - len(ext) - 1] + dot + ext) if dot else name[:_MAX_FILENAME_LEN]
    return name


def validate_extension(filename: str, allowed: list[str]) -> str:
    """Return the lower-cased extension (with dot) or raise 415."""
    ext = PurePath(filename).suffix.lower()
    if ext not in allowed:
        raise FileValidationError(
            f"Unsupported file type '{ext or '(none)'}'. Allowed: {', '.join(sorted(allowed))}.",
            status_code=415,
        )
    return ext


def read_upload_limited(stream: BinaryIO, max_bytes: int) -> bytes:
    """Read at most max_bytes; reading one extra byte is enough to know it's too big."""
    data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FileValidationError(
            f"File too large. Maximum allowed size is {max_bytes // (1024 * 1024)} MB.",
            status_code=413,
        )
    return data


def validate_magic_bytes(ext: str, data: bytes) -> None:
    """Reject files whose content doesn't match their extension (e.g. an .exe renamed to .pdf)."""
    if not data:
        raise FileValidationError("Uploaded file is empty.")
    if ext == ".pdf":
        ok = b"%PDF-" in data[:1024]  # the PDF spec allows the header within the first 1024 bytes
    elif ext == ".docx":
        ok = data.startswith(b"PK\x03\x04")  # DOCX is a ZIP container
    else:  # .txt / .md must be text: NUL bytes indicate binary content
        ok = b"\x00" not in data[:8192]
    if not ok:
        raise FileValidationError(f"File content does not look like a valid {ext} file.")
