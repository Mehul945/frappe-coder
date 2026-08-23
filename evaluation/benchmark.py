from __future__ import annotations

import argparse
import json
from pathlib import Path



def load_tasks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".json":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def prompt_messages(task: dict) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are FrappeCoder, a careful assistant for Frappe, ERPNext, and HRMS development.",
        },
        {"role": "user", "content": task["prompt"]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark prompts and save model answers.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tasks", type=Path, default=Path("evaluation/prompts.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/answers.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for task in load_tasks(args.tasks):
            rendered = tokenizer.apply_chat_template(prompt_messages(task), tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
            handle.write(json.dumps({**task, "answer": answer}, ensure_ascii=False) + "\n")
            print(f"Answered {task['id']}")


if __name__ == "__main__":
    main()


