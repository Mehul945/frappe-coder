# FrappeCoder

FrappeCoder is a focused QLoRA fine-tuning pipeline for adapting `Qwen/Qwen3-4B-Base` to Frappe, ERPNext, and HRMS development.

The first milestone is deliberately small:

- Build a metadata-aware corpus from `frappe`, `erpnext`, and `hrms`.
- Create a held-out Frappe benchmark before training.
- Run a tiny pilot on CPU/small GPU to validate the pipeline.
- Spend GPU time only after dataset loading, tokenization, checkpointing, and inference all work.

## Layout

```text
data/
  raw/          cloned source repositories
  processed/    source/docs/tasks/train JSONL files
  eval/         held-out benchmark and model answers
scripts/        dataset collection and preparation
training/       QLoRA training code
evaluation/     benchmark runner and seed prompts
inference/      local adapter inference
```

## Quick Start

Install dependencies in a Lightning Studio or local Python environment:

```bash
pip install -r requirements.txt
```

Run the entire pipeline—repository sync, extraction, dataset creation, token
statistics, and resumable training—with one command:

```bash
chmod +x run_training.sh
./run_training.sh
```

On later runs, the same command resumes the newest checkpoint automatically. Use
`./run_training.sh --skip-clone` to avoid syncing source repositories, or
`./run_training.sh --fresh` to intentionally start training without a checkpoint.

Clone the core repositories:

```bash
python scripts/clone_repos.py
```

Extract metadata-aware examples:

```bash
python scripts/extract_code.py
python scripts/extract_docs.py
python scripts/build_task_dataset.py
python scripts/deduplicate.py data/processed/source.jsonl data/processed/docs.jsonl data/processed/tasks.jsonl --output data/processed/train.jsonl
python scripts/tokenize_stats.py data/processed/train.jsonl --model Qwen/Qwen3-4B-Base
```

Run the pilot training:

```bash
python training/train.py --config training/config.yaml
```

Long examples are split into balanced, token-sized chunks before training; short
examples are still efficiently combined by packing. Training automatically resumes
from the newest valid `checkpoint-*` directory under `output_dir`. To select a
checkpoint explicitly, use `--resume checkpoints/frappecoder-4b-pilot/checkpoint-300`;
set `resume_from_checkpoint: false` in the config to force a fresh run.

Run evaluation:

```bash
python evaluation/benchmark.py --model Qwen/Qwen3-4B-Base --tasks data/eval/frappe_benchmark.jsonl --output data/eval/baseline_answers.jsonl
```

Run adapter inference after training:

```bash
python inference/chat.py --model Qwen/Qwen3-4B-Base --adapter checkpoints/frappecoder-4b
```

## Data Format

All processed files are JSONL. Training rows may be either plain completion examples:

```json
{"text": "..." , "meta": {"repo": "erpnext", "path": "...", "type": "source"}}
```

or chat examples:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "meta": {"type": "task"}}
```

The trainer accepts both.

## GPU Budget Defaults

The default config targets the first 4-hour pilot:

- Model: `Qwen/Qwen3-4B-Base`
- Method: QLoRA, 4-bit NF4
- Context: 4096 for pilot, increase to 8192 for the main run
- LoRA: `r=32`, `alpha=64`, `dropout=0.05`
- Optimizer: `paged_adamw_8bit`

For the main run, copy `training/config.yaml`, set `max_seq_length: 8192`, increase `max_steps` or set epochs, and save checkpoints frequently.
