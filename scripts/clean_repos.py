from __future__ import annotations

import argparse
from pathlib import Path

from common import RAW_DIR, SKIP_DIRS, SKIP_SUFFIXES


def candidates(raw_dir: Path):
    for path in raw_dir.rglob("*"):
        relative_parts = set(path.relative_to(raw_dir).parts)
        if path.is_dir() and path.name in SKIP_DIRS:
            yield path
            continue
        if path.is_file() and (path.name.lower() in {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}):
            yield path
            continue
        if path.is_file() and any(path.name.lower().endswith(suffix) for suffix in SKIP_SUFFIXES):
            yield path
            continue
        if path.is_file() and relative_parts & {"dist", "coverage", "logs"}:
            yield path


def remove_path(path: Path) -> None:
    if path.is_dir():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove generated or irrelevant files from cloned raw repositories.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--apply", action="store_true", help="Actually delete files. Without this flag, only prints candidates.")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    found = sorted(set(path.resolve() for path in candidates(raw_dir)), key=lambda value: (len(value.parts), str(value)))

    for path in found:
        if raw_dir not in path.parents and path != raw_dir:
            raise RuntimeError(f"Refusing to touch path outside raw_dir: {path}")
        print(path)

    if args.apply:
        for path in sorted(found, key=lambda value: len(value.parts), reverse=True):
            if path.exists():
                remove_path(path)
        print(f"Removed {len(found)} paths")
    else:
        print(f"Found {len(found)} cleanup candidates. Re-run with --apply to delete them.")


if __name__ == "__main__":
    main()

