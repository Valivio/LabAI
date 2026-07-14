from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labai.core.documents import Document, SourceKind
from labai.core.documents.books import Book, render_book_markdown

WORKFLOW_VERSION = "1"
BOOK_WORKFLOW_VERSION = "epub-1"

_SUPPORTED_SOURCE_FORMATS = frozenset({"md", "txt"})


def _validate_path_component(value: str, name: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{name} must be a single safe path component")


def _metadata(document: Document) -> dict[str, str]:
    return {
        "collection": document.collection,
        "content_hash": document.content_hash,
        "document_id": document.document_id,
        "ingested_at": document.ingested_at.isoformat(),
        "original_source_filename": document.source_path.name,
        "source_format": document.source_format,
        "source_kind": document.source_kind.value,
        "workflow_version": WORKFLOW_VERSION,
    }


def _validate_existing_entry(
    document_directory: Path,
    source_filename: str,
    source_bytes: bytes,
    working_bytes: bytes,
    expected_metadata: dict[str, object],
) -> None:
    source_path = document_directory / "source" / source_filename
    working_path = document_directory / "working" / "content.md"
    metadata_path = document_directory / "metadata" / "source.json"

    if not all(path.is_file() for path in (source_path, working_path, metadata_path)):
        raise ValueError("Existing repository entry is incomplete")

    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stable_metadata = {
        key: value for key, value in expected_metadata.items() if key != "ingested_at"
    }
    if (
        set(stored_metadata) != set(expected_metadata)
        or any(stored_metadata.get(key) != value for key, value in stable_metadata.items())
        or source_path.read_bytes() != source_bytes
        or working_path.read_bytes() != working_bytes
    ):
        raise ValueError("Existing repository entry does not match document")


def store_document(document: Document, repository_root: str | Path) -> Path:
    if not isinstance(document.source_kind, SourceKind):
        raise ValueError(f"Unsupported source kind: {document.source_kind!r}")
    if document.source_format not in _SUPPORTED_SOURCE_FORMATS:
        raise ValueError(f"Unsupported source format: {document.source_format}")

    _validate_path_component(document.collection, "Collection")
    _validate_path_component(document.document_id, "Document ID")

    source_bytes = document.source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != document.content_hash:
        raise ValueError("Original source no longer matches the acquired document")

    working_bytes = document.content.encode("utf-8")
    source_filename = document.source_path.name
    expected_metadata = _metadata(document)
    document_directory = (
        Path(repository_root).expanduser().resolve(strict=False)
        / document.collection
        / document.document_id
    )

    if document_directory.exists():
        _validate_existing_entry(
            document_directory,
            source_filename,
            source_bytes,
            working_bytes,
            expected_metadata,
        )
        return document_directory

    source_directory = document_directory / "source"
    working_directory = document_directory / "working"
    metadata_directory = document_directory / "metadata"
    source_directory.mkdir(parents=True)
    working_directory.mkdir()
    metadata_directory.mkdir()

    (source_directory / source_filename).write_bytes(source_bytes)
    (working_directory / "content.md").write_bytes(working_bytes)
    (metadata_directory / "source.json").write_text(
        json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return document_directory


def _book_metadata(book: Book) -> dict[str, object]:
    return {
        "authors": list(book.authors),
        "collection": book.collection,
        "content_hash": book.content_hash,
        "document_id": book.book_id,
        "ingested_at": book.ingested_at.isoformat(),
        "language": book.language,
        "original_source_filename": book.original_source_path.name,
        "publication_date": book.publication_date,
        "publisher": book.publisher,
        "section_count": len(book.sections),
        "source_format": book.source_format,
        "source_identifier": book.source_identifier,
        "source_kind": book.source_kind.value,
        "title": book.title,
        "workflow_version": BOOK_WORKFLOW_VERSION,
    }


def store_book(book: Book, repository_root: str | Path) -> Path:
    if book.source_kind is not SourceKind.BOOK:
        raise ValueError("Book repository requires SourceKind.BOOK")
    if book.source_format != "epub":
        raise ValueError(f"Unsupported book source format: {book.source_format}")

    _validate_path_component(book.collection, "Collection")
    _validate_path_component(book.book_id, "Book ID")

    source_bytes = book.original_source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != book.content_hash:
        raise ValueError("Original source no longer matches the acquired book")

    working_bytes = render_book_markdown(book).encode("utf-8")
    expected_metadata = _book_metadata(book)
    book_directory = (
        Path(repository_root).expanduser().resolve(strict=False)
        / book.collection
        / book.book_id
    )

    if book_directory.exists():
        _validate_existing_entry(
            book_directory,
            "original.epub",
            source_bytes,
            working_bytes,
            expected_metadata,
        )
        return book_directory

    source_directory = book_directory / "source"
    working_directory = book_directory / "working"
    metadata_directory = book_directory / "metadata"
    source_directory.mkdir(parents=True)
    working_directory.mkdir()
    metadata_directory.mkdir()

    (source_directory / "original.epub").write_bytes(source_bytes)
    (working_directory / "content.md").write_bytes(working_bytes)
    (metadata_directory / "source.json").write_text(
        json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return book_directory
