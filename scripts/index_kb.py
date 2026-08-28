from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import load_knowledge_base, split_into_sections
from app.retrieval import DEFAULT_COLLECTION, index_documents


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    documents = load_knowledge_base(repository_root / "knowledge-base")
    chunks = [chunk for document in documents for chunk in split_into_sections(document)]
    indexed = index_documents(documents)
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks indexed: {indexed} (prepared: {len(chunks)})")
    print(f"Collection: {DEFAULT_COLLECTION}")


if __name__ == "__main__":
    main()