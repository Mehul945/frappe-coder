from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import CORE_REPOS, RAW_DIR, ensure_dirs


def run(command: list[str], cwd: Path | None = None) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone or update Frappe ecosystem repositories.")
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    parser.add_argument("--branch", default=None, help="Optional branch/tag to checkout after clone.")
    parser.add_argument("--depth", default="1", help="Git clone depth. Use 0 for full history.")
    parser.add_argument("--repo", action="append", choices=sorted(CORE_REPOS), help="Clone one repo; repeatable.")
    args = parser.parse_args()

    ensure_dirs()
    args.dest.mkdir(parents=True, exist_ok=True)
    repos = args.repo or list(CORE_REPOS)

    for name in repos:
        target = args.dest / name
        url = CORE_REPOS[name]
        if target.exists():
            run(["git", "fetch", "--all", "--tags"], cwd=target)
        else:
            command = ["git", "clone"]
            if args.depth != "0":
                command += ["--depth", args.depth]
            if args.branch:
                command += ["--branch", args.branch]
            command += [url, str(target)]
            run(command)
        if args.branch and target.exists():
            run(["git", "checkout", args.branch], cwd=target)


if __name__ == "__main__":
    main()

