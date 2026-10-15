"""
Format-specific text extraction. Each parser returns a list of ParsedPage
(1-indexed page numbers) so citations can point at a page later.

  PDF   -> one ParsedPage per PDF page
  TXT   -> single page
  MD    -> single page (raw markdown kept, so headings survive for Phase 3)
  DOCX  -> single page (Word has no fixed pages); headings converted to '#' markdown
"""

from dataclasses import dataclass
from pathlib import Path


class ParseError(Exception):
    """The file was accepted but its text could not be extracted."""


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


def parse_document(path: Path, ext: str) -> list[ParsedPage]:
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext in (".txt", ".md"):
        return _parse_text(path)
    if ext == ".docx":
        return _parse_docx(path)
    raise ParseError(f"No parser for extension '{ext}'.")



def _parse_text(path: Path) -> list[ParsedPage]:
    try:
        text = path.read_bytes().decode("utf-8-sig")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise ParseError("File is not valid UTF-8 text.") from exc
    return [ParsedPage(page_number=1, text=text)]



def _parse_pdf(path: Path) -> list[ParsedPage]:
    from pypdf import PasswordType, PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            raise ParseError("PDF is password-protected.")
        return [
            ParsedPage(page_number=i, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages, start=1)
        ]
    except ParseError:
        raise
    except Exception as exc:  # pypdf raises many exception types on malformed input
        raise ParseError(f"Could not read PDF: {exc}") from exc


def _heading_level(style_name: str) -> int:
    if style_name == "Title":
        return 1
    if style_name.startswith("Heading "):
        suffix = style_name.split(" ", 1)[1]
        if suffix.isdigit():
            return min(int(suffix), 6)
    return 0


def _parse_docx(path: Path) -> list[ParsedPage]:
    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = DocxDocument(str(path))
        blocks: list[str] = []
        for child in doc.element.body.iterchildren():  # body order: paragraphs and tables interleaved
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                para = Paragraph(child, doc)
                text = para.text.strip()
                if not text:
                    continue
                style = para.style.name if para.style is not None else ""
                level = _heading_level(style or "")
                blocks.append(f"{'#' * level} {text}" if level else text)
            elif tag == "tbl":
                table = Table(child, doc)
                rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
                blocks.append("\n".join(rows))
    except Exception as exc:  # python-docx raises varied errors on corrupt archives
        raise ParseError(f"Could not read DOCX: {exc}") from exc
    return [ParsedPage(page_number=1, text="\n\n".join(blocks))]
