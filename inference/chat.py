from __future__ import annotations

import argparse



def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with base model or trained LoRA adapter.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
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

    messages = [
        {
            "role": "system",
            "content": "You are FrappeCoder, a careful assistant for Frappe, ERPNext, and HRMS development.",
        }
    ]
    print("FrappeCoder chat. Press Ctrl+C to exit.")
    while True:
        user = input("\nUser: ").strip()
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()


