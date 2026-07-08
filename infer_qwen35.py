"""
Run local text inference with a fine-tuned Qwen model.

Examples:
    python infer_qwen35.py \
        --model_path ./qwen35_finetuned/merged_16bit \
        --prompt "解释什么是机器学习"

    python infer_qwen35.py \
        --model_path ./qwen35_finetuned/lora \
        --base_model Qwen/Qwen3.5-4B \
        --prompt "解释什么是机器学习"
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen fine-tuned model inference")
    parser.add_argument(
        "--model_path",
        default="./qwen35_finetuned/merged_16bit",
        help="Path to merged model directory, or LoRA adapter directory.",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Base model id/path. Required when --model_path points to a LoRA adapter.",
    )
    parser.add_argument("--prompt", required=True, help="User prompt.")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen thinking mode when the tokenizer chat template supports it.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 uses greedy decoding, which is safer on MPS.",
    )
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Model dtype.",
    )
    return parser.parse_args()


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested, device):
    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp32":
        return torch.float32
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float32
    return torch.float32


def build_messages(args):
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    return messages


def build_prompt(tokenizer, messages, enable_thinking=False):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        try:
            return tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking")
            return tokenizer.apply_chat_template(messages, **template_kwargs)

    parts = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def load_model_and_tokenizer(args, device, dtype):
    if args.base_model:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=dtype,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return model, tokenizer


def main():
    args = parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    print(f"Loading model: {args.model_path}")
    if args.base_model:
        print(f"Loading base model: {args.base_model}")
    print(f"Device: {device}, dtype: {dtype}")

    model, tokenizer = load_model_and_tokenizer(args, device, dtype)
    prompt_text = build_prompt(tokenizer, build_messages(args), args.enable_thinking)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    do_sample = args.temperature > 0
    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "remove_invalid_values": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
