from pathlib import Path
import re
from typing import Any

import yaml

from app.schema import KnowledgeChunk, KnowledgeDocument

FRONTMATTER_PATTERN=re.compile(
    r"^---\s*\n(.+?)\n---\s*\n?", re.DOTALL
)

def parse_markdown_file(path: Path) -> KnowledgeDocument:
    """
    Parse a markdown file and return a KnowledgeDocument object.
    """
    text=path.read_text(encoding="utf-8")
    match=FRONTMATTER_PATTERN.match(text)

    if match:
        frontmatter_text=match.group(1)
        content=text[match.end():]
        metadata=yaml.safe_load(frontmatter_text) or {}

    else:
        content=text
        metadata={}

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Frontmatter in {path.name} must be a YAML object"
        )

    expected_fields=[
        "document_id",
        "title",
        "status",
        "effective_date",
        "last_reviewed",
        "audience",
        "policy_authority",
        "customer_answering",
        "supersedes",
        "superseded_date",
        "superseded_by",
    ]

    normalized_metadata: dict[str,Any] = {
        field: metadata.get(field) 
        for field in expected_fields
    }

    normalized_metadata["source"]=path.name

    return KnowledgeDocument(
        source=path.name,
        content=content.strip(),
        metadata=normalized_metadata,
    )

def load_knowledge_base(directory:str|Path) -> list[KnowledgeDocument]:
    """
    load every MArkdown document from the knowledge-base directory.
    """
    directory=Path(directory)
    documents:list[KnowledgeDocument]=[]

    for path in sorted(directory.glob("*.md")):
        documents.append(parse_markdown_file(path))

    return documents


HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def split_into_sections(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    """Split a document into non-empty Markdown heading sections."""
    matches = list(HEADING_PATTERN.finditer(document.content))

    if not matches:
        return [
            KnowledgeChunk(
                content=document.content,
                source=document.source,
                heading="",
                metadata=dict(document.metadata),
            )
        ]

    chunks: list[KnowledgeChunk] = []

    def add_chunk(content: str, heading: str) -> None:
        content = content.strip()
        if content:
            chunks.append(
                KnowledgeChunk(
                    content=content,
                    source=document.source,
                    heading=heading,
                    metadata=dict(document.metadata),
                )
            )

    add_chunk(document.content[: matches[0].start()], "")

    for index, match in enumerate(matches):
        section_start = match.end()
        section_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(document.content)
        )
        heading = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
        add_chunk(document.content[section_start:section_end], heading)

    return chunks