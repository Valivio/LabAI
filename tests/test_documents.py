from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from labai.core.documents import Document, ingest_document


class DocumentIngestionTests(unittest.TestCase):
    def test_ingests_text_document_with_provenance(self) -> None:
        content = b"Synthetic content.\r\nSecond line.\r\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "campaign-notes.txt"
            source_path.write_bytes(content)

            document = ingest_document(source_path, "marketing")

            self.assertEqual(source_path.read_bytes(), content)

        self.assertIsInstance(document, Document)
        self.assertEqual(document.collection, "marketing")
        self.assertEqual(document.title, "campaign-notes")
        self.assertEqual(document.source_path, source_path.resolve())
        self.assertEqual(document.source_type, "txt")
        self.assertEqual(document.content, content.decode("utf-8"))
        self.assertEqual(
            document.content_hash,
            hashlib.sha256(content).hexdigest(),
        )
        self.assertEqual(document.ingested_at.utcoffset(), timedelta(0))

    def test_ingests_markdown_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "brief.md"
            source_path.write_text("# Synthetic brief\n", encoding="utf-8")

            document = ingest_document(source_path, "marketing")

        self.assertEqual(document.source_type, "md")
        self.assertEqual(document.title, "brief")

    def test_document_id_is_deterministic_and_uses_all_identity_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_path = directory / "first.txt"
            second_path = directory / "second.txt"
            first_path.write_text("Version one", encoding="utf-8")
            second_path.write_text("Version one", encoding="utf-8")

            first = ingest_document(first_path, "marketing")
            repeated = ingest_document(first_path, "marketing")
            other_collection = ingest_document(first_path, "sales")
            other_path = ingest_document(second_path, "marketing")

            first_path.write_text("Version two", encoding="utf-8")
            other_content = ingest_document(first_path, "marketing")

        self.assertEqual(first.document_id, repeated.document_id)
        self.assertNotEqual(first.document_id, other_collection.document_id)
        self.assertNotEqual(first.document_id, other_path.document_id)
        self.assertNotEqual(first.document_id, other_content.document_id)

    def test_document_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "notes.txt"
            source_path.write_text("Synthetic notes", encoding="utf-8")
            document = ingest_document(source_path, "marketing")

        with self.assertRaises(FrozenInstanceError):
            document.title = "Changed"  # type: ignore[misc]

    def test_rejects_empty_collection_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "notes.txt"
            source_path.write_text("Synthetic notes", encoding="utf-8")

            for collection in ("", "   "):
                with self.subTest(collection=collection):
                    with self.assertRaisesRegex(ValueError, "Collection name"):
                        ingest_document(source_path, collection)

    def test_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "missing.txt"

            with self.assertRaisesRegex(FileNotFoundError, "Document file not found"):
                ingest_document(source_path, "marketing")

    def test_rejects_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "notes.pdf"
            source_path.write_text("Synthetic notes", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported document type"):
                ingest_document(source_path, "marketing")

    def test_rejects_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "invalid.txt"
            source_path.write_bytes(b"\xff\xfe")

            with self.assertRaises(UnicodeDecodeError):
                ingest_document(source_path, "marketing")

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
                ingest_document(source_path, "marketing")

    def test_rejects_empty_or_whitespace_only_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "empty.txt"

            for content in ("", " \n\t"):
                with self.subTest(content=content):
                    source_path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Document content"):
                        ingest_document(source_path, "marketing")


if __name__ == "__main__":
    unittest.main()
