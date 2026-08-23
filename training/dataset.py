from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def format_row(row: dict[str, Any], tokenizer) -> str:
    if "messages" in row:
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    if "text" in row:
        return row["text"]
    raise ValueError("Training row must contain either messages or text.")


def chunk_text(text: str, tokenizer, max_tokens: int, min_chunk_tokens: int) -> list[str]:
    """Split one example by tokens and avoid leaving a tiny final chunk."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= min_chunk_tokens <= (max_tokens + 1) // 2:
        raise ValueError("min_chunk_tokens must be between 0 and half of max_tokens")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]

    chunk_count = (len(token_ids) + max_tokens - 1) // max_tokens
    base_size, remainder = divmod(len(token_ids), chunk_count)
    sizes = [base_size + (index < remainder) for index in range(chunk_count)]

    # Rebalancing makes every chunk nearly equal instead of leaving a tiny tail.
    assert min(sizes) >= min_chunk_tokens

    chunks: list[str] = []
    offset = 0
    for size in sizes:
        chunks.append(tokenizer.decode(token_ids[offset : offset + size], skip_special_tokens=False))
        offset += size
    return chunks


def build_text_chunks(
    rows: list[dict[str, Any]],
    tokenizer,
    max_tokens: int,
    min_chunk_tokens: int,
) -> list[str]:
    chunks: list[str] = []
    for row in rows:
        chunks.extend(chunk_text(format_row(row, tokenizer), tokenizer, max_tokens, min_chunk_tokens))
    return chunks
