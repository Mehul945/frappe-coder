from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl, stable_hash, write_jsonl


def row_content(row: dict) -> str:
    if "messages" in row:
        return "\n".join(message.get("content", "") for message in row["messages"])
    if "text" in row:
        return row["text"]
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def dedupe(paths: list[Path]):
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            content_hash = stable_hash(row_content(row))
            if content_hash in seen:
                continue
            seen.add(content_hash)
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate JSONL training rows by normalized content hash.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    missing = [path for path in args.inputs if not path.exists()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Missing input JSONL file(s):\n"
            f"{missing_list}\n\n"
            "Run the extraction steps first, for example:\n"
            "  python scripts/clone_repos.py\n"
            "  python scripts/extract_code.py\n"
            "  python scripts/extract_docs.py\n"
            "  python scripts/build_task_dataset.py"
        )

    count = write_jsonl(args.output, dedupe(args.inputs))
    print(f"Wrote {count} deduplicated rows to {args.output}")


if __name__ == "__main__":
    main()
