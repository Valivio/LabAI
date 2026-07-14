from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from labai.core.documents import AcquisitionRequest, Document, SourceKind
from labai.core.documents.acquisition import AcquisitionResult, acquire_document


class DocumentAcquisitionWorkflowTests(unittest.TestCase):
    def request(
        self,
        source_path: Path,
        *,
        collection: str = "marketing",
        source_kind: SourceKind = SourceKind.DOCUMENT,
    ) -> AcquisitionRequest:
        return AcquisitionRequest(
            source_path=source_path,
            collection=collection,
            source_kind=source_kind,
        )

    def test_acquires_txt_document_into_repository(self) -> None:
        source_bytes = b"Synthetic document.\r\nSecond line.\r\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "notes.txt"
            repository_root = temporary_path / "repository"
            source_path.write_bytes(source_bytes)

            result = acquire_document(self.request(source_path), repository_root)

            self.assertIsInstance(result, AcquisitionResult)
            self.assertIsInstance(result.document, Document)
            self.assertEqual(result.document.source_kind, SourceKind.DOCUMENT)
            self.assertEqual(result.document.source_format, "txt")
            self.assertEqual(
                result.repository_path,
                repository_root.resolve()
                / "marketing"
                / result.document.document_id,
            )
            self.assertEqual(source_path.read_bytes(), source_bytes)
            self.assertEqual(
                (result.repository_path / "source" / "notes.txt").read_bytes(),
                source_bytes,
            )
            self.assertEqual(
                (result.repository_path / "working" / "content.md").read_bytes(),
                result.document.content.encode("utf-8"),
            )

            metadata = json.loads(
                (result.repository_path / "metadata" / "source.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["document_id"], result.document.document_id)
            self.assertEqual(metadata["source_kind"], "document")
            self.assertEqual(metadata["source_format"], "txt")

    def test_acquires_markdown_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "brief.md"
            source_path.write_text("# Synthetic brief\n", encoding="utf-8")

            result = acquire_document(
                self.request(source_path),
                temporary_path / "repository",
            )

            self.assertEqual(result.document.source_format, "md")
            self.assertEqual(
                (result.repository_path / "working" / "content.md").read_text(
                    encoding="utf-8"
                ),
                "# Synthetic brief\n",
            )

    def test_repeated_acquisition_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "notes.txt"
            repository_root = temporary_path / "repository"
            source_path.write_text("Synthetic notes", encoding="utf-8")
            request = self.request(source_path)

            first = acquire_document(request, repository_root)
            first_metadata = (
                first.repository_path / "metadata" / "source.json"
            ).read_bytes()
            second = acquire_document(request, repository_root)

            self.assertEqual(first.document.document_id, second.document.document_id)
            self.assertEqual(first.repository_path, second.repository_path)
            self.assertEqual(
                (second.repository_path / "metadata" / "source.json").read_bytes(),
                first_metadata,
            )
            self.assertEqual(
                [path.name for path in (repository_root / "marketing").iterdir()],
                [first.document.document_id],
            )

    def test_acquisition_result_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "notes.txt"
            source_path.write_text("Synthetic notes", encoding="utf-8")
            result = acquire_document(
                self.request(source_path),
                temporary_path / "repository",
            )

        with self.assertRaises(FrozenInstanceError):
            result.repository_path = Path("/synthetic")  # type: ignore[misc]

    def test_rejects_book_before_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            request = self.request(
                temporary_path / "missing-book.txt",
                source_kind=SourceKind.BOOK,
            )

            with self.assertRaisesRegex(ValueError, "SourceKind.DOCUMENT"):
                acquire_document(request, temporary_path / "repository")

    def test_propagates_ingestion_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "unsupported.pdf"
            source_path.write_text("Synthetic content", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported document type"):
                acquire_document(
                    self.request(source_path),
                    temporary_path / "repository",
                )

    def test_propagates_repository_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "notes.txt"
            source_path.write_text("Synthetic notes", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Collection"):
                acquire_document(
                    self.request(source_path, collection="../outside"),
                    temporary_path / "repository",
                )


if __name__ == "__main__":
    unittest.main()
