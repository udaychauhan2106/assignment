from pathlib import Path

from app.ingestion import load_knowledge_base, split_into_sections


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    documents = load_knowledge_base(repository_root / "knowledge-base")
    chunks = [chunk for document in documents for chunk in split_into_sections(document)]

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    for chunk in chunks:
        preview = " ".join(chunk.content.split())[:100]
        print(f"- {chunk.source} | {chunk.heading or '[preamble]'} | {preview!r}")


if __name__ == "__main__":
    main()