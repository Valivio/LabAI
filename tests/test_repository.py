from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from labai.core.documents import AcquisitionRequest, SourceKind, ingest_document
from labai.core.documents.repository import WORKFLOW_VERSION, store_document


class DocumentRepositoryTests(unittest.TestCase):
    def acquire(
        self,
        source_path: Path,
        *,
        collection: str = "marketing",
        source_kind: SourceKind = SourceKind.DOCUMENT,
    ):
        return ingest_document(
            AcquisitionRequest(
                source_path=source_path,
                collection=collection,
                source_kind=source_kind,
            )
        )

    def test_stores_original_working_content_and_metadata(self) -> None:
        source_bytes = b"# Synthetic source\r\n\r\nWorking text.\r\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "source.md"
            repository_root = temporary_path / "repository"
            source_path.write_bytes(source_bytes)
            document = self.acquire(source_path, source_kind=SourceKind.BOOK)

            document_directory = store_document(document, repository_root)

            stored_source = document_directory / "source" / "source.md"
            working_content = document_directory / "working" / "content.md"
            metadata_path = document_directory / "metadata" / "source.json"

            self.assertEqual(source_path.read_bytes(), source_bytes)
            self.assertEqual(stored_source.read_bytes(), source_bytes)
            self.assertEqual(
                working_content.read_bytes(),
                document.content.encode("utf-8"),
            )
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                {
                    "collection": "marketing",
                    "content_hash": document.content_hash,
                    "document_id": document.document_id,
                    "ingested_at": document.ingested_at.isoformat(),
                    "original_source_filename": "source.md",
                    "source_format": "md",
                    "source_kind": "book",
                    "workflow_version": WORKFLOW_VERSION,
                },
            )

        self.assertEqual(
            document_directory,
            repository_root.resolve() / "marketing" / document.document_id,
        )

    def test_same_unchanged_source_is_stored_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "notes.txt"
            repository_root = temporary_path / "repository"
            source_path.write_text("Synthetic notes", encoding="utf-8")

            first_document = self.acquire(source_path)
            first_directory = store_document(first_document, repository_root)
            first_metadata = (first_directory / "metadata" / "source.json").read_bytes()

            second_document = self.acquire(source_path)
            second_directory = store_document(second_document, repository_root)

            self.assertEqual(first_document.document_id, second_document.document_id)
            self.assertEqual(first_directory, second_directory)
            self.assertEqual(
                (second_directory / "metadata" / "source.json").read_bytes(),
                first_metadata,
            )
            self.assertEqual(
                [path.name for path in (repository_root / "marketing").iterdir()],
                [first_document.document_id],
            )

    def test_repository_copy_survives_original_source_removal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "notes.txt"
            repository_root = temporary_path / "repository"
            source_bytes = b"Synthetic notes"
            source_path.write_bytes(source_bytes)
            document = self.acquire(source_path)
            document_directory = store_document(document, repository_root)

            source_path.unlink()

            self.assertEqual(
                (document_directory / "source" / "notes.txt").read_bytes(),
                source_bytes,
            )

    def test_rejects_source_changed_after_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "notes.txt"
            repository_root = temporary_path / "repository"
            source_path.write_text("Version one", encoding="utf-8")
            document = self.acquire(source_path)
            source_path.write_text("Version two", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Original source"):
                store_document(document, repository_root)

            self.assertFalse(repository_root.exists())

    def test_rejects_unsupported_source_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "notes.txt"
            source_path.write_text("Synthetic notes", encoding="utf-8")
            document = replace(self.acquire(source_path), source_format="pdf")

            with self.assertRaisesRegex(ValueError, "Unsupported source format"):
                store_document(document, temporary_path / "repository")

    def test_rejects_collection_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "notes.txt"
            source_path.write_text("Synthetic notes", encoding="utf-8")
            document = self.acquire(source_path, collection="../outside")

            with self.assertRaisesRegex(ValueError, "Collection"):
                store_document(document, temporary_path / "repository")


if __name__ == "__main__":
    unittest.main()
