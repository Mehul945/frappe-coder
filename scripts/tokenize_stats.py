from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from transformers import AutoTokenizer

from common import read_jsonl


console = Console()


def text_for(row: dict, tokenizer) -> str:
    if "messages" in row:
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    return row.get("text", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate token counts and context-fit stats for JSONL datasets.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = 0
    tokens = 0
    over_limit = 0
    max_tokens = 0

    for row in read_jsonl(args.input):
        length = len(tokenizer(text_for(row, tokenizer), add_special_tokens=False).input_ids)
        rows += 1
        tokens += length
        max_tokens = max(max_tokens, length)
        over_limit += int(length > args.max_seq_length)

    console.print(
        {
            "rows": rows,
            "tokens": tokens,
            "avg_tokens": round(tokens / max(rows, 1), 2),
            "max_tokens": max_tokens,
            "over_limit": over_limit,
            "over_limit_pct": round(100 * over_limit / max(rows, 1), 2),
        }
    )


if __name__ == "__main__":
    main()

