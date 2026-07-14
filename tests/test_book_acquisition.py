from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path

from labai.core.documents import AcquisitionRequest, SourceKind
from labai.core.documents.book_acquisition import (
    BookAcquisitionResult,
    acquire_book,
)
from labai.core.documents.books import Book, BookSection
from labai.core.documents.repository import BOOK_WORKFLOW_VERSION


class EpubBookAcquisitionTests(unittest.TestCase):
    def request(
        self,
        source_path: Path,
        *,
        collection: str = "library",
        source_kind: SourceKind = SourceKind.BOOK,
    ) -> AcquisitionRequest:
        return AcquisitionRequest(
            source_path=source_path,
            collection=collection,
            source_kind=source_kind,
        )

    def create_epub(
        self,
        source_path: Path,
        *,
        readable: bool = True,
        include_optional_metadata: bool = True,
        epub_version: str = "3.0",
    ) -> bytes:
        optional_metadata = ""
        if include_optional_metadata:
            optional_metadata = """
                <dc:creator>Ada Example</dc:creator>
                <dc:creator>Lin Example</dc:creator>
                <dc:publisher>Synthetic Press</dc:publisher>
                <dc:date>2026-07-14</dc:date>
            """

        if readable:
            first_content = (
                "<h1>First Section</h1>"
                "<p>First synthetic paragraph.</p>"
                "<h2>Detail</h2><p>Structured detail.</p>"
            )
            second_content = (
                "<h1>Second Section</h1><p>Second synthetic paragraph.</p>"
            )
        else:
            first_content = "<h1>Empty Section</h1><p>   </p>"
            second_content = "<h1>Another Empty Section</h1>"

        container_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="EPUB/content.opf"
                          media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>
        """
        if epub_version == "2.0":
            navigation_manifest = (
                '<item id="ncx" href="toc.ncx" '
                'media-type="application/x-dtbncx+xml"/>'
            )
            spine_attributes = ' toc="ncx"'
            navigation_spine = ""
        else:
            navigation_manifest = """
                <item id="nav" href="nav.xhtml"
                      media-type="application/xhtml+xml" properties="nav"/>
            """
            spine_attributes = ""
            navigation_spine = '<itemref idref="nav"/>'

        package_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="{epub_version}"
                     unique-identifier="book-id">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:identifier id="book-id">urn:synthetic:book-001</dc:identifier>
                <dc:title>Synthetic Field Guide</dc:title>
                <dc:language>en</dc:language>
                {optional_metadata}
              </metadata>
              <manifest>
                {navigation_manifest}
                <item id="cover" href="cover.xhtml"
                      media-type="application/xhtml+xml"/>
                <item id="chapter-1" href="text/first.xhtml"
                      media-type="application/xhtml+xml"/>
                <item id="chapter-2" href="text/second.xhtml"
                      media-type="application/xhtml+xml"/>
              </manifest>
              <spine{spine_attributes}>
                {navigation_spine}
                <itemref idref="cover"/>
                <itemref idref="chapter-1"/>
                <itemref idref="chapter-2"/>
              </spine>
              <guide>
                <reference type="cover" title="Cover" href="cover.xhtml"/>
              </guide>
            </package>
        """

        def xhtml(body: str) -> str:
            return (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>Synthetic</title></head><body>{body}</body></html>"
            )

        with zipfile.ZipFile(source_path, "w") as epub_archive:
            epub_archive.writestr(
                "mimetype",
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            epub_archive.writestr("META-INF/container.xml", container_xml)
            epub_archive.writestr("EPUB/content.opf", package_xml)
            # Archive order deliberately differs from the spine reading order.
            epub_archive.writestr(
                "EPUB/text/second.xhtml",
                xhtml(second_content),
            )
            epub_archive.writestr(
                "EPUB/cover.xhtml",
                xhtml("<h1>Cover</h1><p>Excluded cover text.</p>"),
            )
            epub_archive.writestr(
                "EPUB/nav.xhtml",
                xhtml("<nav><p>Navigation</p></nav>"),
            )
            if epub_version == "2.0":
                epub_archive.writestr(
                    "EPUB/toc.ncx",
                    """<?xml version="1.0" encoding="UTF-8"?>
                        <ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">
                          <head/><docTitle><text>Synthetic Field Guide</text></docTitle>
                          <navMap/>
                        </ncx>
                    """,
                )
            epub_archive.writestr(
                "EPUB/text/first.xhtml",
                xhtml(first_content),
            )

        return source_path.read_bytes()

    def test_acquires_epub2_in_spine_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "synthetic-epub2.epub"
            self.create_epub(source_path, epub_version="2.0")

            result = acquire_book(
                self.request(source_path),
                temporary_path / "repository",
            )

        self.assertEqual(
            [section.title for section in result.book.sections],
            ["First Section", "Second Section"],
        )
        self.assertEqual(
            [section.provenance for section in result.book.sections],
            ["text/first.xhtml", "text/second.xhtml"],
        )

    def test_acquires_epub_book_with_structure_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "synthetic.epub"
            repository_root = temporary_path / "repository"
            source_bytes = self.create_epub(source_path)

            result = acquire_book(self.request(source_path), repository_root)

            self.assertIsInstance(result, BookAcquisitionResult)
            self.assertIsInstance(result.book, Book)
            self.assertEqual(result.book.title, "Synthetic Field Guide")
            self.assertEqual(result.book.authors, ("Ada Example", "Lin Example"))
            self.assertEqual(result.book.language, "en")
            self.assertEqual(result.book.publisher, "Synthetic Press")
            self.assertEqual(result.book.publication_date, "2026-07-14")
            self.assertEqual(result.book.source_identifier, "urn:synthetic:book-001")
            self.assertEqual(result.book.source_format, "epub")
            self.assertEqual(result.book.source_kind, SourceKind.BOOK)
            self.assertEqual(result.book.original_source_path, source_path.resolve())
            self.assertEqual(
                [section.title for section in result.book.sections],
                ["First Section", "Second Section"],
            )
            self.assertEqual(
                [section.order for section in result.book.sections],
                [1, 2],
            )
            self.assertTrue(
                all(isinstance(section, BookSection) for section in result.book.sections)
            )
            self.assertEqual(
                [section.provenance for section in result.book.sections],
                ["text/first.xhtml", "text/second.xhtml"],
            )
            self.assertIn("### Detail", result.book.sections[0].content)

            self.assertEqual(source_path.read_bytes(), source_bytes)
            self.assertEqual(
                (result.repository_path / "source" / "original.epub").read_bytes(),
                source_bytes,
            )

            working_markdown = (
                result.repository_path / "working" / "content.md"
            ).read_text(encoding="utf-8")
            self.assertIn("# Synthetic Field Guide", working_markdown)
            self.assertIn("**Authors:** Ada Example, Lin Example", working_markdown)
            self.assertLess(
                working_markdown.index("## First Section"),
                working_markdown.index("## Second Section"),
            )
            self.assertNotIn("Excluded cover text", working_markdown)
            self.assertNotIn("Navigation", working_markdown)

            metadata = json.loads(
                (result.repository_path / "metadata" / "source.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["document_id"], result.book.book_id)
            self.assertEqual(metadata["authors"], ["Ada Example", "Lin Example"])
            self.assertEqual(metadata["section_count"], 2)
            self.assertEqual(metadata["source_kind"], "book")
            self.assertEqual(metadata["source_format"], "epub")
            self.assertEqual(metadata["workflow_version"], BOOK_WORKFLOW_VERSION)

    def test_missing_optional_metadata_does_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "minimal.epub"
            self.create_epub(source_path, include_optional_metadata=False)

            result = acquire_book(
                self.request(source_path),
                temporary_path / "repository",
            )

        self.assertEqual(result.book.authors, ())
        self.assertIsNone(result.book.publisher)
        self.assertIsNone(result.book.publication_date)

    def test_repeated_acquisition_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "synthetic.epub"
            repository_root = temporary_path / "repository"
            self.create_epub(source_path)
            request = self.request(source_path)

            first = acquire_book(request, repository_root)
            first_working = (
                first.repository_path / "working" / "content.md"
            ).read_bytes()
            first_metadata = (
                first.repository_path / "metadata" / "source.json"
            ).read_bytes()
            second = acquire_book(request, repository_root)

            self.assertEqual(first.book.book_id, second.book.book_id)
            self.assertEqual(first.book.sections, second.book.sections)
            self.assertEqual(first.repository_path, second.repository_path)
            self.assertEqual(
                (second.repository_path / "working" / "content.md").read_bytes(),
                first_working,
            )
            self.assertEqual(
                (second.repository_path / "metadata" / "source.json").read_bytes(),
                first_metadata,
            )
            self.assertEqual(
                [path.name for path in (repository_root / "library").iterdir()],
                [first.book.book_id],
            )

    def test_book_models_and_result_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "synthetic.epub"
            self.create_epub(source_path)
            result = acquire_book(
                self.request(source_path),
                temporary_path / "repository",
            )

        with self.assertRaises(FrozenInstanceError):
            result.book.title = "Changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.book.sections[0].title = "Changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.repository_path = Path("/synthetic")  # type: ignore[misc]

    def test_rejects_document_source_kind_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            request = self.request(
                temporary_path / "missing.epub",
                source_kind=SourceKind.DOCUMENT,
            )

            with self.assertRaisesRegex(ValueError, "SourceKind.BOOK"):
                acquire_book(request, temporary_path / "repository")

    def test_rejects_missing_non_file_and_wrong_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            repository_root = temporary_path / "repository"

            with self.assertRaisesRegex(FileNotFoundError, "Book file not found"):
                acquire_book(
                    self.request(temporary_path / "missing.epub"),
                    repository_root,
                )

            directory_path = temporary_path / "directory.epub"
            directory_path.mkdir()
            with self.assertRaisesRegex(ValueError, "not a file"):
                acquire_book(self.request(directory_path), repository_root)

            wrong_extension = temporary_path / "book.txt"
            wrong_extension.write_text("not an epub", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported book type"):
                acquire_book(self.request(wrong_extension), repository_root)

    def test_rejects_corrupt_and_invalid_epub(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            repository_root = temporary_path / "repository"
            corrupt_path = temporary_path / "corrupt.epub"
            corrupt_path.write_bytes(b"not a zip archive")

            with self.assertRaisesRegex(ValueError, "Invalid or corrupt EPUB"):
                acquire_book(self.request(corrupt_path), repository_root)

            invalid_path = temporary_path / "invalid.epub"
            with zipfile.ZipFile(invalid_path, "w") as invalid_epub:
                invalid_epub.writestr("mimetype", "application/epub+zip")

            with self.assertRaisesRegex(ValueError, "Invalid or corrupt EPUB"):
                acquire_book(self.request(invalid_path), repository_root)

    def test_rejects_epub_without_readable_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "empty.epub"
            self.create_epub(source_path, readable=False)

            with self.assertRaisesRegex(ValueError, "no readable textual content"):
                acquire_book(
                    self.request(source_path),
                    temporary_path / "repository",
                )

    def test_rejects_unsafe_repository_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "synthetic.epub"
            self.create_epub(source_path)

            with self.assertRaisesRegex(ValueError, "Collection"):
                acquire_book(
                    self.request(source_path, collection="../outside"),
                    temporary_path / "repository",
                )

    def test_incomplete_repository_entry_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "synthetic.epub"
            repository_root = temporary_path / "repository"
            self.create_epub(source_path)
            request = self.request(source_path)
            first = acquire_book(request, repository_root)
            (first.repository_path / "metadata" / "source.json").unlink()

            with self.assertRaisesRegex(ValueError, "incomplete"):
                acquire_book(request, repository_root)

    def test_conflicting_repository_entry_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "synthetic.epub"
            repository_root = temporary_path / "repository"
            self.create_epub(source_path)
            request = self.request(source_path)
            first = acquire_book(request, repository_root)
            metadata_path = first.repository_path / "metadata" / "source.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["title"] = "Conflicting title"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match"):
                acquire_book(request, repository_root)


if __name__ == "__main__":
    unittest.main()
