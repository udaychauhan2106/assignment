from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orders import lookup_order


def main() -> None:
    examples = [
        "ORD-1007",
        " ord-1007 ",
        "ORD-9999",
        None,
        "1007",
    ]
    for value in examples:
        result = lookup_order(value)
        print(f"Input: {value!r} -> state={result.state}, found={result.found}")
        print(result.model_dump(exclude_none=True))
        print()


if __name__ == "__main__":
    main()