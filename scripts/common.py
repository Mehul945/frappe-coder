from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

CORE_REPOS = {
    "frappe": "https://github.com/frappe/frappe.git",
    "erpnext": "https://github.com/frappe/erpnext.git",
    "hrms": "https://github.com/frappe/hrms.git",
}

KEEP_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".vue",
    ".json",
    ".md",
    ".html",
    ".css",
    ".sql",
    ".yml",
    ".yaml",
    ".txt",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "archived",
    "coverage",
    "dist",
    "env",
    "logs",
    "node_modules",
    "public",
    "sites",
    "venv",
}

SKIP_SUFFIXES = {
    ".lock",
    ".map",
    ".min.css",
    ".min.js",
    ".pyc",
    ".pyo",
}

MAX_FILE_BYTES = 512_000


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, EVAL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def iter_repo_files(repo_dir: Path) -> Iterable[Path]:
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(repo_dir).parts)
        if parts & SKIP_DIRS:
            continue
        if path.suffix.lower() not in KEEP_EXTENSIONS:
            continue
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in SKIP_SUFFIXES):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        if is_probably_binary(path):
            continue
        yield path


def is_probably_binary(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    if mime and not (mime.startswith("text/") or mime in {"application/json", "application/sql"}):
        return True
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\x00" in chunk


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def language_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".css": "css",
        ".html": "html",
        ".js": "javascript",
        ".json": "json",
        ".md": "markdown",
        ".py": "python",
        ".sql": "sql",
        ".ts": "typescript",
        ".vue": "vue",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(suffix, suffix.lstrip(".") or "text")


def classify_file(relative_path: Path, text: str) -> str:
    path = str(relative_path).replace("\\", "/").lower()
    name = relative_path.name.lower()
    if "doctype" in path and name.endswith(".py"):
        return "doctype_controller"
    if "doctype" in path and name.endswith(".json"):
        return "doctype_json"
    if "report" in path and name.endswith(".json"):
        return "report_json"
    if "print_format" in path:
        return "print_format"
    if name == "hooks.py":
        return "hooks"
    if "patches" in path:
        return "migration_patch"
    if "test" in name or "/test" in path:
        return "test"
    if name.endswith(".md"):
        return "documentation"
    if "workspace" in path and name.endswith(".json"):
        return "workspace"
    if "property_setter" in path:
        return "property_setter"
    if "custom_field" in path:
        return "custom_field"
    if text.startswith("import frappe") or "\nimport frappe" in text:
        return "frappe_python"
    return "source"


def domain_for(repo: str, relative_path: Path) -> str:
    path = str(relative_path).replace("\\", "/").lower()
    if repo == "hrms" or any(key in path for key in ["payroll", "leave", "attendance", "employee"]):
        return "hrms"
    if any(key in path for key in ["accounts", "stock", "selling", "buying", "manufacturing"]):
        return "erpnext"
    if any(key in path for key in ["desk", "website", "permissions", "model", "database"]):
        return "frappe_framework"
    return repo

