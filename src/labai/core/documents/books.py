from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from labai.core.documents import SourceKind


@dataclass(frozen=True)
class BookSection:
    section_id: str
    order: int
    title: str
    content: str
    provenance: str


@dataclass(frozen=True)
class Book:
    book_id: str
    collection: str
    title: str
    authors: tuple[str, ...]
    language: str | None
    publisher: str | None
    publication_date: str | None
    source_identifier: str | None
    source_format: str
    source_kind: SourceKind
    original_source_path: Path
    sections: tuple[BookSection, ...]
    content_hash: str
    ingested_at: datetime


def render_book_markdown(book: Book) -> str:
    parts = [f"# {book.title}"]

    if book.authors:
        parts.append(f"**Authors:** {', '.join(book.authors)}")

    for section in book.sections:
        parts.extend((f"## {section.title}", section.content))

    return "\n\n".join(parts) + "\n"
