from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from epub_utils import Document as EpubDocument

from labai.core.documents import AcquisitionRequest, SourceKind
from labai.core.documents.books import Book, BookSection
from labai.core.documents.repository import store_book


@dataclass(frozen=True)
class BookAcquisitionResult:
    book: Book
    repository_path: Path


class _ReadableHTMLParser(HTMLParser):
    _BLOCK_TAGS = frozenset(
        {"blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "pre"}
    )
    _IGNORED_TAGS = frozenset({"head", "nav", "script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._current_tag: str | None = None
        self._current_text: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        tag = tag.lower()
        if self._ignored_depth:
            self._ignored_depth += 1
        elif tag in self._IGNORED_TAGS:
            self._flush()
            self._ignored_depth = 1
        elif tag in self._BLOCK_TAGS:
            self._flush()
            self._current_tag = tag
        elif tag == "br" and self._current_tag is not None:
            self._current_text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == self._current_tag:
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and self._current_tag is not None:
            self._current_text.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._current_tag is not None:
            text = " ".join("".join(self._current_text).split())
            if text:
                self.blocks.append((self._current_tag, text))
        self._current_tag = None
        self._current_text = []


def _metadata_values(epub_document: EpubDocument, name: str) -> tuple[str, ...]:
    value = getattr(epub_document.package.metadata, name, None)
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list):
        values = tuple(value)
    else:
        values = ()
    return tuple(item.strip() for item in values if item and item.strip())


def _first_metadata(epub_document: EpubDocument, name: str) -> str | None:
    return next(iter(_metadata_values(epub_document, name)), None)


def _book_id(
    collection: str,
    source_path: Path,
    content_hash: str,
) -> str:
    identity = json.dumps(
        [collection, str(source_path), SourceKind.BOOK.value, content_hash],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _section_id(book_id: str, order: int, provenance: str) -> str:
    identity = json.dumps(
        [book_id, order, provenance],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _cover_resources(epub_document: EpubDocument) -> set[str]:
    resources: set[str] = set()
    package_root = ElementTree.fromstring(epub_document.package.xml_content)
    for element in package_root.iter():
        if element.tag.rsplit("}", 1)[-1] != "reference":
            continue
        if element.get("type") == "cover" and element.get("href"):
            resources.add(element.get("href", "").split("#", 1)[0])
    return resources


def _extract_section(
    xml_content: str,
    fallback_title: str,
) -> tuple[str, str] | None:
    parser = _ReadableHTMLParser()
    parser.feed(xml_content)
    parser.close()

    first_heading = next(
        (text for tag, text in parser.blocks if tag.startswith("h")),
        None,
    )
    title = first_heading or fallback_title
    content_parts: list[str] = []
    skipped_title = False

    for tag, text in parser.blocks:
        if tag.startswith("h") and not skipped_title and text == first_heading:
            skipped_title = True
            continue
        if tag.startswith("h"):
            content_parts.append(f"### {text}")
        else:
            content_parts.append(text)

    if not content_parts:
        return None
    return title, "\n\n".join(content_parts)


def _extract_sections(
    epub_document: EpubDocument,
    book_id: str,
) -> tuple[BookSection, ...]:
    cover_resources = _cover_resources(epub_document)
    extracted: list[tuple[str, str, str]] = []

    for item_reference in epub_document.package.spine.itemrefs:
        item_id = item_reference["idref"]
        item = epub_document.package.manifest.find_by_id(item_id)
        if item is None:
            raise ValueError(f"EPUB spine references missing item: {item_id}")
        if item["media_type"] not in {"application/xhtml+xml", "text/html"}:
            raise ValueError(f"EPUB spine item is not readable XHTML: {item_id}")
        if "nav" in item["properties"] or item["href"] in cover_resources:
            continue

        content = epub_document.find_content_by_id(item_id)
        section = _extract_section(content.xml_content, Path(item["href"]).stem)
        if section is not None:
            title, content = section
            extracted.append((title, content, item["href"]))

    return tuple(
        BookSection(
            section_id=_section_id(book_id, order, provenance),
            order=order,
            title=title,
            content=content,
            provenance=provenance,
        )
        for order, (title, content, provenance) in enumerate(extracted, start=1)
    )


def _ingest_epub_book(request: AcquisitionRequest) -> Book:
    if not request.collection.strip():
        raise ValueError("Collection name must not be empty")

    source_path = Path(request.source_path).expanduser().resolve(strict=False)
    if not source_path.exists():
        raise FileNotFoundError(f"Book file not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Book source is not a file: {source_path}")
    if source_path.suffix.lower() != ".epub":
        raise ValueError(f"Unsupported book type: {source_path.suffix}")

    source_bytes = source_path.read_bytes()
    content_hash = hashlib.sha256(source_bytes).hexdigest()
    book_id = _book_id(request.collection, source_path, content_hash)

    try:
        epub_document = EpubDocument(source_path)
        title = _first_metadata(epub_document, "title") or source_path.stem
        sections = _extract_sections(epub_document, book_id)
        authors = _metadata_values(epub_document, "creator")
        language = _first_metadata(epub_document, "language")
        publisher = _first_metadata(epub_document, "publisher")
        publication_date = _first_metadata(epub_document, "date")
        source_identifier = _first_metadata(epub_document, "identifier")
    except Exception as error:
        raise ValueError(f"Invalid or corrupt EPUB: {source_path}") from error

    if not sections:
        raise ValueError("EPUB contains no readable textual content")

    return Book(
        book_id=book_id,
        collection=request.collection,
        title=title,
        authors=authors,
        language=language,
        publisher=publisher,
        publication_date=publication_date,
        source_identifier=source_identifier,
        source_format="epub",
        source_kind=SourceKind.BOOK,
        original_source_path=source_path,
        sections=sections,
        content_hash=content_hash,
        ingested_at=datetime.now(timezone.utc),
    )


def acquire_book(
    request: AcquisitionRequest,
    repository_root: str | Path,
) -> BookAcquisitionResult:
    if request.source_kind is not SourceKind.BOOK:
        raise ValueError("Book acquisition requires SourceKind.BOOK")

    book = _ingest_epub_book(request)
    repository_path = store_book(book, repository_root)
    return BookAcquisitionResult(book=book, repository_path=repository_path)
