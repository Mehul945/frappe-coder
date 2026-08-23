#!/usr/bin/env bash
set -Eeuo pipefail

# Run from any directory. Override the interpreter with PYTHON_BIN=/path/to/python.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="training/config.yaml"
SKIP_CLONE=0
FRESH=0

usage() {
  cat <<'EOF'
Usage: ./run_training.sh [--config PATH] [--skip-clone] [--fresh]

Build the FrappeCoder dataset, report token statistics, and train the model.
By default, training resumes from the newest checkpoint in the configured output_dir.

  --config PATH  Use a different training YAML file
  --skip-clone   Use repositories already present in data/raw
  --fresh        Do not resume from an existing checkpoint
  -h, --help     Show this help
EOF
}

while (($#)); do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a path" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --skip-clone)
      SKIP_CLONE=1
      shift
      ;;
    --fresh)
      FRESH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 127; }

on_error() {
  local exit_code=$?
  echo "Pipeline failed at line $1. Fix the error and rerun this command; saved checkpoints remain resumable." >&2
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

if ((SKIP_CLONE == 0)); then
  "$PYTHON_BIN" scripts/clone_repos.py
fi

"$PYTHON_BIN" scripts/extract_code.py
"$PYTHON_BIN" scripts/extract_docs.py
"$PYTHON_BIN" scripts/build_task_dataset.py
TRAIN_FILE="$($PYTHON_BIN -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["train_file"])' "$CONFIG")"
"$PYTHON_BIN" scripts/deduplicate.py \
  data/processed/source.jsonl \
  data/processed/docs.jsonl \
  data/processed/tasks.jsonl \
  --output "$TRAIN_FILE"

# Read these values from the selected config so stats match the actual training run.
MODEL_NAME="$($PYTHON_BIN -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["model_name"])' "$CONFIG")"
MAX_SEQ_LENGTH="$($PYTHON_BIN -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("max_seq_length", 4096))' "$CONFIG")"
"$PYTHON_BIN" scripts/tokenize_stats.py "$TRAIN_FILE" \
  --model "$MODEL_NAME" \
  --max-seq-length "$MAX_SEQ_LENGTH"

TRAIN_ARGS=(--config "$CONFIG")
if ((FRESH == 1)); then
  TRAIN_ARGS+=(--resume none)
fi
"$PYTHON_BIN" training/train.py "${TRAIN_ARGS[@]}"

echo "Pipeline complete. Model adapter and checkpoints are in the configured output_dir."
