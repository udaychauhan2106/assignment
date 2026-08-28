import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Aster & Row knowledge base")
    parser.add_argument("question")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    response = answer_question(args.question, debug=args.debug)

    print(f"QUESTION\n{args.question}\n")
    print(f"ANSWER\n{response.answer}\n")
    print("SOURCES")
    for source in response.sources:
        print(f"- {source.source} | {source.heading}")
    print(f"\nHANDOFF\n{response.handoff}")
    if response.retrieved_chunks is not None:
        print("\nRETRIEVED CHUNKS SUPPLIED TO MODEL")
        for index, chunk in enumerate(response.retrieved_chunks, start=1):
            print(f"{index}. {chunk.source} | {chunk.heading}")
            print(chunk.content[:200].replace("\n", " "))


if __name__ == "__main__":
    main()