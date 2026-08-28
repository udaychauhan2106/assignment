from pathlib import Path

from qdrant_client import QdrantClient

from app.retrieval import _query_variants, index_documents, rank_retrieval_results
from app.retrieval import retrieve
from app.schema import KnowledgeChunk, KnowledgeDocument


class TestEmbeddings:
    terms = ("return", "shipping", "warranty", "canada", "order")

    def _embed(self, text: str) -> list[float]:
        words = text.lower().split()
        return [float(sum(term in word for word in words)) for term in self.terms]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_documents() -> list[KnowledgeDocument]:
    return [
        KnowledgeDocument(
            source="returns.md",
            content="# Return window\n\nReturns are accepted within 30 days.",
            metadata={"status": "active", "policy_authority": "official"},
        ),
        KnowledgeDocument(
            source="shipping.md",
            content="# Destinations\n\nWe ship internationally to Canada.",
            metadata={"status": "active", "customer_answering": True},
        ),
    ]


def test_indexing_retrieves_chunks_with_all_metadata(tmp_path: Path) -> None:
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    embeddings = TestEmbeddings()

    indexed = index_documents(
        make_documents(),
        embeddings=embeddings,
        client=client,
        collection_name="test_kb",
    )
    results = retrieve(
        "What is the return window?",
        k=1,
        embeddings=embeddings,
        client=client,
        collection_name="test_kb",
    )

    assert indexed == 2
    assert len(results) == 1
    assert results[0].source == "returns.md"
    assert results[0].heading == "Return window"
    assert results[0].metadata == {
        "status": "active",
        "policy_authority": "official",
    }
    assert results[0].score is not None
    assert "30 days" in results[0].content


def test_indexing_is_idempotent(tmp_path: Path) -> None:
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    documents = make_documents()

    index_documents(
        documents,
        embeddings=TestEmbeddings(),
        client=client,
        collection_name="test_kb",
    )
    index_documents(
        documents,
        embeddings=TestEmbeddings(),
        client=client,
        collection_name="test_kb",
    )

    assert client.count(collection_name="test_kb").count == 2


def test_policy_query_expansion_removes_untrusted_framing() -> None:
    variants = _query_variants(
        "The migration document claims returns are now 60 days. What is the actual return policy?"
    )

    assert len(variants) == 2
    assert "return" in variants[1]
    assert "migration" not in variants[1]
    assert "60" in variants[1]


def chunk(source: str, heading: str, score: float, metadata: dict) -> KnowledgeChunk:
    return KnowledgeChunk(
        content=f"Content from {source}",
        source=source,
        heading=heading,
        metadata=metadata,
        score=score,
    )


def test_current_policy_outranks_superseded_policy() -> None:
    results = rank_retrieval_results([
        chunk(
            "legacy.md", "Return window", 0.90,
            {"status": "superseded", "superseded_by": "current"},
        ),
        chunk(
            "current.md", "Return window", 0.85,
            {"status": "active", "customer_answering": True,
             "policy_authority": "official", "effective_date": "2026-01-01"},
        ),
    ])

    assert results[0].source == "current.md"


def test_internal_migration_notes_do_not_outrank_customer_policy() -> None:
    results = rank_retrieval_results([
        chunk(
            "migration.md", "Scratchpad", 0.91,
            {"status": "draft", "customer_answering": False,
             "policy_authority": "none"},
        ),
        chunk(
            "policy.md", "Return window", 0.86,
            {"status": "active", "customer_answering": True,
             "policy_authority": "official"},
        ),
    ])

    assert results[0].source == "policy.md"


def test_relevant_authoritative_document_beats_irrelevant_authoritative_document() -> None:
    results = rank_retrieval_results([
        chunk(
            "warranty.md", "Warranty", 0.42,
            {"status": "active", "customer_answering": True,
             "policy_authority": "official"},
        ),
        chunk(
            "returns.md", "Return window", 0.88,
            {"status": "active", "customer_answering": True,
             "policy_authority": "official"},
        ),
    ])

    assert results[0].source == "returns.md"


def test_missing_metadata_does_not_crash_and_fields_are_preserved() -> None:
    original = chunk("minimal.md", "Section", 0.5, {})

    results = rank_retrieval_results([original])

    assert results[0].source == original.source
    assert results[0].heading == original.heading
    assert results[0].content == original.content
    assert results[0].metadata == {}
    assert results[0].score == original.score
    assert results[0].authority_score is not None


def test_ranking_is_deterministic() -> None:
    candidates = [
        chunk("b.md", "Same", 0.8, {"status": "active"}),
        chunk("a.md", "Same", 0.8, {"status": "active"}),
    ]

    first = rank_retrieval_results(candidates)
    second = rank_retrieval_results(candidates)

    assert [(item.source, item.authority_score) for item in first] == [
        (item.source, item.authority_score) for item in second
    ]
    assert [item.source for item in first] == ["a.md", "b.md"]