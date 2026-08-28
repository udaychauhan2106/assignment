import json
import os
import re
from datetime import date
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.ingestion import split_into_sections
from app.schema import KnowledgeChunk, KnowledgeDocument


DEFAULT_COLLECTION = "aster_row_knowledge"
DEFAULT_QDRANT_PATH = ".qdrant"
AUTHORITY_WEIGHT = 0.08


def _client() -> QdrantClient:
    return QdrantClient(path=os.getenv("QDRANT_PATH", DEFAULT_QDRANT_PATH))


def _embeddings() -> GoogleGenerativeAIEmbeddings:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini embeddings")
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        api_key=api_key,
    )


def _payload_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML values such as dates to values Qdrant can serialize."""
    return json.loads(json.dumps(metadata, default=str))


def _chunk_id(chunk: KnowledgeChunk) -> str:
    identity = f"{chunk.source}\n{chunk.heading}\n{chunk.content}"
    return str(uuid5(NAMESPACE_URL, identity))


def _query_variants(query: str) -> list[str]:
    """Add a clean policy-intent query when untrusted framing obscures it."""
    lowered = query.lower()
    if not any(word in lowered for word in ("migration", "note", "document", "claims", "ignore")):
        return [query]
    policy_terms = ("return", "refund", "cancel", "shipping", "warranty", "damaged", "delivery")
    if not any(term in lowered for term in policy_terms):
        return [query]
    cleaned = re.sub(
        r"\b(migration|note|notes|document|documents|claims?|ignore|real|newer|use|that|approve|approval|give|everyone|says?|said|should|i|me|my|the|and|to)\b",
        " ",
        lowered,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .?!")
    return [query, cleaned] if cleaned and cleaned != lowered else [query]


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {
        "true", "yes", "1", "official", "current", "active"
    }


def _date_value(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def authority_score(chunk: KnowledgeChunk) -> float:
    """Return an explainable authority score from 0 (weak) to 1 (strong)."""
    metadata = chunk.metadata or {}
    score = 0.0

    customer_answering = metadata.get("customer_answering")
    if _is_true(customer_answering):
        score += 0.40
    elif customer_answering is False or str(customer_answering).strip().lower() in {
        "false", "no", "0"
    }:
        score -= 0.20

    status = str(metadata.get("status") or "").strip().lower()
    if status in {"active", "current"}:
        score += 0.25
    elif status in {"superseded", "legacy", "archived"}:
        score -= 0.25
    elif status in {"draft", "inactive"}:
        score -= 0.15

    policy_authority = metadata.get("policy_authority")
    if _is_true(policy_authority):
        score += 0.20
    elif str(policy_authority or "").strip().lower() in {"none", "unofficial"}:
        score -= 0.10

    if metadata.get("superseded_by") or metadata.get("superseded_date"):
        score -= 0.25

    # A valid effective date makes otherwise comparable current content preferable.
    if _date_value(metadata.get("effective_date")):
        score += 0.05

    return max(0.0, min(1.0, score))


def rank_retrieval_results(results: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    """Rerank semantic candidates without removing less authoritative sources."""
    ranked: list[KnowledgeChunk] = []
    for result in results:
        ranked.append(
            KnowledgeChunk(
                content=result.content,
                source=result.source,
                heading=result.heading,
                metadata=dict(result.metadata),
                score=result.score,
                authority_score=authority_score(result),
            )
        )

    def sort_key(result: KnowledgeChunk) -> tuple[float, float, float, str, str]:
        similarity = result.score if result.score is not None else 0.0
        authority = result.authority_score or 0.0
        return (
            -(similarity + AUTHORITY_WEIGHT * authority),
            -authority,
            -similarity,
            result.source,
            result.heading,
        )

    return sorted(ranked, key=sort_key)


def index_documents(
    documents: list[KnowledgeDocument],
    *,
    embeddings: Any | None = None,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
) -> int:
    """Embed and index document sections, recreating the named collection."""
    chunks = [
        chunk
        for document in documents
        for chunk in split_into_sections(document)
    ]
    if not chunks:
        raise ValueError("Cannot index an empty document set")

    embeddings = embeddings or _embeddings()
    client = client or _client()
    collection_name = collection_name or os.getenv(
        "QDRANT_COLLECTION", DEFAULT_COLLECTION
    )
    vectors = embeddings.embed_documents([chunk.content for chunk in chunks])
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=_chunk_id(chunk),
            vector=vector,
            payload={
                "content": chunk.content,
                "source": chunk.source,
                "heading": chunk.heading,
                "metadata": _payload_metadata(chunk.metadata),
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(chunks)


def retrieve(
    query: str,
    k: int = 5,
    *,
    embeddings: Any | None = None,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
) -> list[KnowledgeChunk]:
    """Return the top semantic matches, including their similarity scores."""
    if k < 1:
        raise ValueError("k must be at least 1")

    embeddings = embeddings or _embeddings()
    client = client or _client()
    collection_name = collection_name or os.getenv(
        "QDRANT_COLLECTION", DEFAULT_COLLECTION
    )
    points_by_id = {}
    for variant in _query_variants(query):
        response = client.query_points(
            collection_name=collection_name,
            query=embeddings.embed_query(variant),
            limit=k,
            with_payload=True,
        )
        for point in response.points:
            existing = points_by_id.get(point.id)
            if existing is None or point.score > existing.score:
                points_by_id[point.id] = point
    points = sorted(points_by_id.values(), key=lambda point: (-point.score, str(point.id)))[:k]
    return [
        KnowledgeChunk(
            content=point.payload["content"],
            source=point.payload["source"],
            heading=point.payload["heading"],
            metadata=point.payload["metadata"],
            score=point.score,
        )
        for point in points
    ]