from __future__ import annotations

import argparse
from pathlib import Path

from common import PROCESSED_DIR, RAW_DIR, domain_for, ensure_dirs, iter_repo_files, read_text, stable_hash, write_jsonl


DOC_NAMES = {"readme.md", "contributing.md", "license.md"}


def is_doc(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    return path.suffix.lower() == ".md" or "/docs/" in normalized or path.name.lower() in DOC_NAMES


def build_rows(raw_dir: Path):
    for repo_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        repo = repo_dir.name
        for path in iter_repo_files(repo_dir):
            relative = path.relative_to(repo_dir)
            if not is_doc(relative):
                continue
            text = read_text(path)
            if len(text) < 120:
                continue
            normalized_path = str(relative).replace("\\", "/")
            yield {
                "id": stable_hash(f"docs:{repo}:{relative}:{text}"),
                "text": (
                    f"Repository: {repo}\n"
                    f"Documentation path: {normalized_path}\n\n"
                    f"{text}"
                ),
                "meta": {
                    "repo": repo,
                    "path": normalized_path,
                    "domain": domain_for(repo, relative),
                    "language": "markdown",
                    "type": "documentation",
                    "bytes": path.stat().st_size,
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract documentation examples.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "docs.jsonl")
    args = parser.parse_args()

    ensure_dirs()
    count = write_jsonl(args.output, build_rows(args.raw_dir))
    print(f"Wrote {count} documentation examples to {args.output}")


if __name__ == "__main__":
    main()
