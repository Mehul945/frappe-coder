from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import re

import yaml

from dataset import build_text_chunks, load_jsonl


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    output_dir = Path(output_dir)
    checkpoints: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return None
    for path in output_dir.iterdir():
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match and (path / "trainer_state.json").is_file():
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume_checkpoint(value, output_dir: str | Path) -> Path | None:
    if value in (None, False, "none", "false"):
        return None
    if value in (True, "auto", "latest"):
        return latest_checkpoint(output_dir)
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA train FrappeCoder.")
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        help="Resume from the latest checkpoint, or from an explicit checkpoint path.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=dtype_map.get(config.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    rows = load_jsonl(config["train_file"])
    max_seq_length = config.get("max_seq_length", 4096)
    chunk_size = config.get("chunk_size", max_seq_length)
    if chunk_size > max_seq_length:
        raise ValueError("chunk_size cannot exceed max_seq_length")
    min_chunk_tokens = config.get("min_chunk_tokens", max(1, chunk_size // 4))
    texts = build_text_chunks(rows, tokenizer, chunk_size, min_chunk_tokens)
    dataset = Dataset.from_dict({"text": texts})
    print(f"Prepared {len(texts)} training chunks from {len(rows)} rows (up to {chunk_size} tokens each).")

    peft_config = LoraConfig(
        r=config.get("lora_r", 32),
        lora_alpha=config.get("lora_alpha", 64),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.get("lora_target_modules"),
    )

    requested_max_steps = config.get("max_steps", -1)
    warmup_ratio = config.get("warmup_ratio", 0.05)
    warmup_steps = config.get("warmup_steps")
    if warmup_steps is None and requested_max_steps and requested_max_steps > 0:
        warmup_steps = max(1, int(requested_max_steps * warmup_ratio))

    training_kwargs = {
        "output_dir": config["output_dir"],
        "num_train_epochs": config.get("num_train_epochs", 1),
        "max_steps": requested_max_steps,
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 16),
        "learning_rate": config.get("learning_rate", 1e-4),
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "logging_steps": config.get("logging_steps", 10),
        "save_steps": config.get("save_steps", 100),
        "save_total_limit": config.get("save_total_limit", 5),
        "bf16": config.get("bf16", True),
        "gradient_checkpointing": config.get("gradient_checkpointing", True),
        "optim": config.get("optim", "paged_adamw_8bit"),
        "report_to": config.get("report_to", "none"),
    }
    training_args = TrainingArguments(**filter_supported_kwargs(TrainingArguments.__init__, training_kwargs))

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": dataset,
        "dataset_text_field": "text",
        "max_seq_length": max_seq_length,
        "packing": config.get("packing", True),
        "peft_config": peft_config,
        "args": training_args,
    }
    trainer = SFTTrainer(**filter_supported_kwargs(SFTTrainer.__init__, trainer_kwargs))
    resume_value = args.resume if args.resume is not None else config.get("resume_from_checkpoint", "auto")
    resume_checkpoint = resolve_resume_checkpoint(resume_value, config["output_dir"])
    if resume_checkpoint:
        print(f"Resuming training from {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
