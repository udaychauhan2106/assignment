from app.ingestion import split_into_sections
from app.schema import KnowledgeDocument


def make_document(content: str, metadata: dict | None = None) -> KnowledgeDocument:
    return KnowledgeDocument(
        source="policy.md",
        content=content,
        metadata=metadata or {},
    )


def test_splits_normal_headings_and_preserves_source_and_metadata() -> None:
    metadata = {"status": "active", "authority": "official"}

    chunks = split_into_sections(
        make_document("# Returns\n\nReturn details.\n\n## Exceptions\n\nSome exceptions.", metadata)
    )

    assert [(chunk.heading, chunk.content) for chunk in chunks] == [
        ("Returns", "Return details."),
        ("Exceptions", "Some exceptions."),
    ]
    assert all(chunk.source == "policy.md" for chunk in chunks)
    assert all(chunk.metadata == metadata for chunk in chunks)


def test_preserves_multiple_heading_levels_and_preamble() -> None:
    document = make_document(
        "Introductory context.\n\n# Title\n\nTop level.\n\n### Detail\n\nDetails.\n\n###### Note\n\nNote."
    )

    chunks = split_into_sections(document)

    assert [(chunk.heading, chunk.content) for chunk in chunks] == [
        ("", "Introductory context."),
        ("Title", "Top level."),
        ("Detail", "Details."),
        ("Note", "Note."),
    ]


def test_document_without_headings_returns_whole_document() -> None:
    content = "Text without Markdown headings.\n\nIt stays together."

    chunks = split_into_sections(make_document(content))

    assert len(chunks) == 1
    assert chunks[0].content == content
    assert chunks[0].heading == ""


def test_ignores_empty_sections() -> None:
    chunks = split_into_sections(
        make_document("# Empty\n\n## Filled\n\nUseful content.\n\n### Also empty")
    )

    assert [(chunk.heading, chunk.content) for chunk in chunks] == [
        ("Filled", "Useful content."),
    ]


def test_does_not_require_metadata_fields_or_mutate_metadata() -> None:
    metadata = {"title": "Returns"}
    document = make_document("# Section\n\nBody.", metadata)

    chunks = split_into_sections(document)

    assert chunks[0].metadata == {"title": "Returns"}
    assert chunks[0].metadata is not document.metadata