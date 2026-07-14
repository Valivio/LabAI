from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from labai.core.documents import (
    AcquisitionRequest,
    Document,
    SourceKind,
    ingest_document,
)


class DocumentIngestionTests(unittest.TestCase):
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

    def test_ingests_text_document_with_provenance(self) -> None:
        content = b"Synthetic content.\r\nSecond line.\r\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "campaign-notes.txt"
            source_path.write_bytes(content)

            document = ingest_document(self.request(source_path))

            self.assertEqual(source_path.read_bytes(), content)

        self.assertIsInstance(document, Document)
        self.assertEqual(document.collection, "marketing")
        self.assertEqual(document.title, "campaign-notes")
        self.assertEqual(document.source_path, source_path.resolve())
        self.assertEqual(document.source_kind, SourceKind.DOCUMENT)
        self.assertEqual(document.source_format, "txt")
        self.assertEqual(document.content, content.decode("utf-8"))
        self.assertEqual(
            document.content_hash,
            hashlib.sha256(content).hexdigest(),
        )
        self.assertEqual(document.ingested_at.utcoffset(), timedelta(0))

    def test_source_kind_and_format_are_independent(self) -> None:
        combinations = (
            (SourceKind.BOOK, ".txt", "txt"),
            (SourceKind.BOOK, ".md", "md"),
            (SourceKind.DOCUMENT, ".txt", "txt"),
            (SourceKind.DOCUMENT, ".md", "md"),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)

            for source_kind, extension, expected_format in combinations:
                with self.subTest(source_kind=source_kind, extension=extension):
                    source_path = directory / f"source-{source_kind.value}{extension}"
                    source_path.write_text("Synthetic content", encoding="utf-8")

                    document = ingest_document(
                        self.request(source_path, source_kind=source_kind)
                    )

                    self.assertEqual(document.source_kind, source_kind)
                    self.assertEqual(document.source_format, expected_format)

    def test_document_id_is_deterministic_and_uses_all_identity_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_path = directory / "first.txt"
            second_path = directory / "second.txt"
            first_path.write_text("Version one", encoding="utf-8")
            second_path.write_text("Version one", encoding="utf-8")

            first = ingest_document(self.request(first_path))
            repeated = ingest_document(self.request(first_path))
            other_collection = ingest_document(
                self.request(first_path, collection="sales")
            )
            other_path = ingest_document(self.request(second_path))
            other_kind = ingest_document(
                self.request(first_path, source_kind=SourceKind.BOOK)
            )

            first_path.write_text("Version two", encoding="utf-8")
            other_content = ingest_document(self.request(first_path))

        self.assertEqual(first.document_id, repeated.document_id)
        self.assertNotEqual(first.document_id, other_collection.document_id)
        self.assertNotEqual(first.document_id, other_path.document_id)
        self.assertNotEqual(first.document_id, other_kind.document_id)
        self.assertNotEqual(first.document_id, other_content.document_id)

    def test_document_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "notes.txt"
            source_path.write_text("Synthetic notes", encoding="utf-8")
            document = ingest_document(self.request(source_path))

        with self.assertRaises(FrozenInstanceError):
            document.title = "Changed"  # type: ignore[misc]

    def test_acquisition_request_is_immutable(self) -> None:
        request = AcquisitionRequest(
            source_path="/synthetic/notes.txt",
            collection="marketing",
            source_kind=SourceKind.DOCUMENT,
        )

        with self.assertRaises(FrozenInstanceError):
            request.collection = "sales"  # type: ignore[misc]

    def test_rejects_unsupported_source_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported source kind"):
            AcquisitionRequest(
                source_path="/synthetic/notes.txt",
                collection="marketing",
                source_kind="podcast",  # type: ignore[arg-type]
            )

    def test_rejects_empty_collection_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "notes.txt"
            source_path.write_text("Synthetic notes", encoding="utf-8")

            for collection in ("", "   "):
                with self.subTest(collection=collection):
                    with self.assertRaisesRegex(ValueError, "Collection name"):
                        ingest_document(self.request(source_path, collection=collection))

    def test_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "missing.txt"

            with self.assertRaisesRegex(FileNotFoundError, "Document file not found"):
                ingest_document(self.request(source_path))

    def test_rejects_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "notes.pdf"
            source_path.write_text("Synthetic notes", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported document type"):
                ingest_document(self.request(source_path))

    def test_rejects_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "invalid.txt"
            source_path.write_bytes(b"\xff\xfe")

            with self.assertRaises(UnicodeDecodeError):
                ingest_document(self.request(source_path))

    def test_propagates_unreadable_file_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "unreadable.txt"
            source_path.write_text("Synthetic notes", encoding="utf-8")

            with (
                patch.object(
                    Path,
                    "read_bytes",
                    side_effect=PermissionError("synthetic permission error"),
                ),
                self.assertRaises(PermissionError),
            ):
                ingest_document(self.request(source_path))

    def test_rejects_empty_or_whitespace_only_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "empty.txt"

            for content in ("", " \n\t"):
                with self.subTest(content=content):
                    source_path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Document content"):
                        ingest_document(self.request(source_path))


if __name__ == "__main__":
    unittest.main()
