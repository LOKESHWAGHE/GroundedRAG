import io

import pytest

from app.services.ingestion.file_validation import (
    FileValidationError,
    read_upload_limited,
    sanitize_filename,
    validate_extension,
    validate_magic_bytes,
)

ALLOWED = [".pdf", ".txt", ".md", ".docx"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\notes.txt", "notes.txt"),
        ("my<script>.md", "my_script_.md"),
        ("  spaced   name .txt", "spaced name .txt"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "...", "/"])
def test_sanitize_rejects_empty_names(raw):
    with pytest.raises(FileValidationError):
        sanitize_filename(raw)


def test_extension_allowlist():
    assert validate_extension("A.PDF", ALLOWED) == ".pdf"
    with pytest.raises(FileValidationError) as exc:
        validate_extension("virus.exe", ALLOWED)
    assert exc.value.status_code == 415


def test_size_limit_enforced_while_reading():
    assert read_upload_limited(io.BytesIO(b"abc"), 3) == b"abc"
    with pytest.raises(FileValidationError) as exc:
        read_upload_limited(io.BytesIO(b"abcd"), 3)
    assert exc.value.status_code == 413


def test_magic_bytes_accept_valid_and_reject_mismatch():
    validate_magic_bytes(".pdf", b"%PDF-1.7 ...")
    validate_magic_bytes(".docx", b"PK\x03\x04rest")
    validate_magic_bytes(".txt", "héllo".encode())
    for ext, data in [(".pdf", b"not a pdf"), (".docx", b"not a zip"), (".txt", b"bin\x00ary")]:
        with pytest.raises(FileValidationError):
            validate_magic_bytes(ext, data)


def test_empty_file_rejected():
    with pytest.raises(FileValidationError):
        validate_magic_bytes(".txt", b"")
