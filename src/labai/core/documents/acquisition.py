from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from labai.core.documents import (
    AcquisitionRequest,
    Document,
    SourceKind,
    ingest_document,
)
from labai.core.documents.repository import store_document


@dataclass(frozen=True)
class AcquisitionResult:
    document: Document
    repository_path: Path


def acquire_document(
    request: AcquisitionRequest,
    repository_root: str | Path,
) -> AcquisitionResult:
    if request.source_kind is not SourceKind.DOCUMENT:
        raise ValueError("Document acquisition requires SourceKind.DOCUMENT")

    document = ingest_document(request)
    repository_path = store_document(document, repository_root)
    return AcquisitionResult(
        document=document,
        repository_path=repository_path,
    )
