from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from common import CORE_REPOS, PROCESSED_DIR, RAW_DIR, stable_hash, write_jsonl


def github_api(url: str, token: str | None) -> dict | list:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "frappe-coder-dataset-builder",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def repo_slug(remote_url: str) -> str:
    parsed = urlparse(remote_url)
    if parsed.netloc == "github.com":
        return parsed.path.strip("/").removesuffix(".git")
    if remote_url.startswith("git@github.com:"):
        return remote_url.split(":", 1)[1].removesuffix(".git")
    raise ValueError(f"Cannot derive GitHub repo slug from {remote_url}")


def local_diff(repo_dir: Path, base_ref: str, head_ref: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--find-renames", f"{base_ref}...{head_ref}"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def build_rows(raw_dir: Path, limit: int, token: str | None):
    for repo_name in CORE_REPOS:
        repo_dir = raw_dir / repo_name
        if not repo_dir.exists():
            continue
        slug = repo_slug(CORE_REPOS[repo_name])
        pulls = github_api(f"https://api.github.com/repos/{slug}/pulls?state=closed&per_page={limit}", token)
        for pull in pulls:
            if not pull.get("merged_at"):
                continue
            number = pull["number"]
            base_sha = pull["base"]["sha"]
            merge_sha = pull.get("merge_commit_sha") or pull["head"]["sha"]
            try:
                subprocess.run(["git", "fetch", "origin", base_sha, merge_sha], cwd=repo_dir, check=True)
                diff = local_diff(repo_dir, base_sha, merge_sha)
            except (subprocess.CalledProcessError, FileNotFoundError):
                diff = ""
            if not diff:
                continue
            issue_text = pull.get("body") or pull.get("title") or ""
            user = (
                f"Fix this {repo_name} issue or pull request.\n\n"
                f"Title: {pull.get('title', '')}\n"
                f"Description:\n{issue_text[:6000]}"
            )
            assistant = (
                "The relevant implementation is represented by this patch:\n\n"
                "```diff\n"
                f"{diff[:24000]}\n"
                "```"
            )
            yield {
                "id": stable_hash(f"pr:{repo_name}:{number}:{merge_sha}"),
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "meta": {
                    "repo": repo_name,
                    "path": f"pulls/{number}",
                    "domain": repo_name,
                    "language": "diff",
                    "type": "pull_request",
                    "url": pull.get("html_url"),
                    "merged_at": pull.get("merged_at"),
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PR problem-to-patch examples from local repos and GitHub metadata.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "prs.jsonl")
    parser.add_argument("--limit", type=int, default=25, help="Closed PRs to inspect per repo.")
    parser.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN"))
    args = parser.parse_args()

    count = write_jsonl(args.output, build_rows(args.raw_dir, args.limit, args.github_token))
    print(f"Wrote {count} PR examples to {args.output}")


if __name__ == "__main__":
    main()

