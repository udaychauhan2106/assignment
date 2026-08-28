import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import rank_retrieval_results, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the indexed Aster & Row KB")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    results = rank_retrieval_results(retrieve(args.query, args.k))
    for rank, chunk in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"Score: {chunk.score:.4f}")
        print(f"Authority/reranking score: {chunk.authority_score:.4f}")
        print(f"Source: {chunk.source}")
        print(f"Heading: {chunk.heading or '[preamble]'}")
        print(f"Status: {chunk.metadata.get('status')}")
        print(f"Customer answering: {chunk.metadata.get('customer_answering')}")
        print(f"Policy authority: {chunk.metadata.get('policy_authority')}")
        print("Content:")
        print(chunk.content)
        print()


if __name__ == "__main__":
    main()