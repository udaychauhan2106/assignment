import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import _customer_safe
from app.retrieval import rank_retrieval_results, retrieve


def _print_chunk(rank: int, chunk) -> None:
    print(f"Rank: {rank}")
    print(f"Source: {chunk.source}")
    print(f"Heading: {chunk.heading or '[preamble]'}")
    print(f"Similarity score: {chunk.score:.4f}")
    print(f"Authority/reranking score: {chunk.authority_score:.4f}")
    print(f"Status: {chunk.metadata.get('status')}")
    print(f"Customer answering: {chunk.metadata.get('customer_answering')}")
    print(f"Policy authority: {chunk.metadata.get('policy_authority')}")
    print("Content:")
    print(chunk.content)
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval and reranking without calling an LLM"
    )
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument(
        "--generation-context",
        action="store_true",
        help="Show the exact customer-safe chunks RAG would supply to the model",
    )
    args = parser.parse_args()

    results = rank_retrieval_results(retrieve(args.query, args.k))
    if args.generation_context:
        results = [chunk for chunk in results if _customer_safe(chunk)]
        print("Chunks supplied to generation model (no model call made)")
    else:
        print("Retrieved and reranked candidates (no model call made)")
    print(f"Query: {args.query}")
    print(f"Count: {len(results)}")
    print("=" * 80)
    for rank, chunk in enumerate(results, start=1):
        _print_chunk(rank, chunk)


if __name__ == "__main__":
    main()