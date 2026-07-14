from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SOURCE_TYPES = {
    ".md": "md",
    ".txt": "txt",
}


@dataclass(frozen=True)
class Document:
    document_id: str
    collection: str
    title: str
    source_path: Path
    source_type: str
    content: str
    content_hash: str
    ingested_at: datetime


def _document_id(collection: str, source_path: Path, content_hash: str) -> str:
    identity = json.dumps(
        [collection, str(source_path), content_hash],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def ingest_document(source_path: str | Path, collection: str) -> Document:
    if not collection.strip():
        raise ValueError("Collection name must not be empty")

    normalized_path = Path(source_path).expanduser().resolve(strict=False)
    if not normalized_path.is_file():
        raise FileNotFoundError(f"Document file not found: {normalized_path}")

    extension = normalized_path.suffix.lower()
    if extension not in _SOURCE_TYPES:
        raise ValueError(f"Unsupported document type: {normalized_path.suffix}")

    source_bytes = normalized_path.read_bytes()
    content = source_bytes.decode("utf-8")
    if not content.strip():
        raise ValueError("Document content must not be empty")

    content_hash = hashlib.sha256(source_bytes).hexdigest()

    return Document(
        document_id=_document_id(collection, normalized_path, content_hash),
        collection=collection,
        title=normalized_path.stem,
        source_path=normalized_path,
        source_type=_SOURCE_TYPES[extension],
        content=content,
        content_hash=content_hash,
        ingested_at=datetime.now(timezone.utc),
    )
