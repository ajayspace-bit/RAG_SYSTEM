import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.config import get_assets_path


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass
class PdfChunk:
    chunk_id: str
    text: str
    source: str
    page: int


def find_pdf() -> Path:
    """Find the configured PDF directory's first PDF in a stable order."""
    pdfs = sorted(get_assets_path().glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF file found in {get_assets_path()}")
    return pdfs[0]


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract non-empty text from every PDF page, retaining 1-based pages."""
    reader = PdfReader(str(pdf_path))
    return [
        (page_number, text.strip())
        for page_number, page in enumerate(reader.pages, start=1)
        if (text := (page.extract_text() or "").strip())
    ]


def chunk_pages(pages: list[tuple[int, str]], source: str) -> list[PdfChunk]:
    """Split page text into overlapping character chunks."""
    chunks: list[PdfChunk] = []
    for page, text in pages:
        start = 0
        page_chunk_number = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    PdfChunk(
                        chunk_id=f"{source}-p{page}-c{page_chunk_number}",
                        text=chunk_text,
                        source=source,
                        page=page,
                    )
                )
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
            page_chunk_number += 1
    return chunks


def pdf_fingerprint(pdf_path: Path) -> str:
    """Create a content hash used to detect changes between indexing runs."""
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()