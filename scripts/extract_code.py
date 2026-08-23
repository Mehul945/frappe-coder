from __future__ import annotations

import argparse
from pathlib import Path

from common import PROCESSED_DIR, RAW_DIR, classify_file, domain_for, ensure_dirs, iter_repo_files, language_for, read_text, stable_hash, write_jsonl


def build_rows(raw_dir: Path):
    for repo_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        repo = repo_dir.name
        for path in iter_repo_files(repo_dir):
            text = read_text(path)
            if not text:
                continue
            relative = path.relative_to(repo_dir)
            kind = classify_file(relative, text)
            if kind == "documentation":
                continue
            yield {
                "id": stable_hash(f"{repo}:{relative}:{text}"),
                "text": format_source_example(repo, relative, text),
                "meta": {
                    "repo": repo,
                    "path": str(relative).replace("\\", "/"),
                    "domain": domain_for(repo, relative),
                    "language": language_for(path),
                    "type": kind,
                    "bytes": path.stat().st_size,
                },
            }


def format_source_example(repo: str, relative: Path, text: str) -> str:
    return (
        f"Repository: {repo}\n"
        f"Path: {str(relative).replace('\\', '/')}\n\n"
        "```text\n"
        f"{text}\n"
        "```"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract metadata-aware source examples.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "source.jsonl")
    args = parser.parse_args()

    ensure_dirs()
    count = write_jsonl(args.output, build_rows(args.raw_dir))
    print(f"Wrote {count} source examples to {args.output}")


if __name__ == "__main__":
    main()

