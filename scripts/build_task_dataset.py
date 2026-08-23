from __future__ import annotations

import argparse
from pathlib import Path

from common import PROCESSED_DIR, read_jsonl, stable_hash, write_jsonl


TASK_TEMPLATES = {
    "doctype_controller": "Explain how this Frappe DocType controller should be extended safely, including validation and tests.",
    "doctype_json": "Identify the important fields and relationships in this Frappe DocType JSON.",
    "hooks": "Explain what this hooks.py configuration changes in the Frappe app lifecycle.",
    "migration_patch": "Explain what this Frappe migration patch changes and what should be tested.",
    "test": "Summarize the behavior covered by this test and name the production code it protects.",
    "documentation": "Turn this documentation into practical implementation guidance for a Frappe developer.",
}


def build_rows(inputs: list[Path], max_chars: int):
    for input_path in inputs:
        for row in read_jsonl(input_path):
            meta = row.get("meta", {})
            kind = meta.get("type", "source")
            prompt = TASK_TEMPLATES.get(kind)
            if not prompt:
                continue
            context = row.get("text", "")[:max_chars]
            user = (
                f"{prompt}\n\n"
                f"Repository: {meta.get('repo', 'unknown')}\n"
                f"Path: {meta.get('path', 'unknown')}\n\n"
                f"Context:\n{context}"
            )
            assistant = (
                "Review the file in its Frappe context, preserve framework conventions, "
                "call out permission or migration risks, and propose focused tests before changing behavior."
            )
            yield {
                "id": stable_hash(f"task:{row.get('id', '')}:{prompt}"),
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "meta": {**meta, "type": "task", "source_type": kind},
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build starter instruction/task examples from extracted corpus rows.")
    parser.add_argument("inputs", nargs="*", type=Path, default=[PROCESSED_DIR / "source.jsonl", PROCESSED_DIR / "docs.jsonl"])
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "tasks.jsonl")
    parser.add_argument("--max-chars", type=int, default=12000)
    args = parser.parse_args()

    count = write_jsonl(args.output, build_rows(args.inputs, args.max_chars))
    print(f"Wrote {count} task examples to {args.output}")


if __name__ == "__main__":
    main()

