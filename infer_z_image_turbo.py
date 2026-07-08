"""
Z-Image-Turbo local inference script.

Example:
    python infer_z_image_turbo.py \
        --model_path /path/to/Z-Image-Turbo \
        --prompt "一只橘猫坐在赛博朋克城市的窗边，电影感光影" \
        --output outputs/z_image_turbo.png
"""

import argparse
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with a local Z-Image-Turbo model."
    )
    parser.add_argument(
        "--model_path",
        default=os.environ.get("Z_IMAGE_TURBO_MODEL", "Tongyi-MAI/Z-Image-Turbo"),
        help=(
            "Local model directory or Hugging Face model id. "
            "You can also set Z_IMAGE_TURBO_MODEL."
        ),
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for generation.")
    parser.add_argument(
        "--negative_prompt",
        default=None,
        help="Optional negative prompt. Turbo usually works best without CFG.",
    )
    parser.add_argument("--output", default="outputs/z_image_turbo.png")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--steps",
        type=int,
        default=9,
        help="Inference steps. Z-Image-Turbo commonly uses 9, resulting in 8 DiT forwards.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance. Use 0.0 for Z-Image-Turbo.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Model dtype. auto uses bf16 on CUDA, fp16 on MPS, fp32 on CPU.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load files already present locally.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force Hugging Face/diffusers offline mode.",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable model CPU offload to reduce VRAM usage on CUDA.",
    )
    parser.add_argument(
        "--attention_backend",
        default=None,
        help='Optional transformer attention backend, for example "flash".',
    )
    parser.add_argument(
        "--compile_transformer",
        action="store_true",
        help="Compile the transformer for faster later runs. First run is slower.",
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
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_pipeline(model_path, dtype, local_files_only):
    try:
        from diffusers import ZImagePipeline
    except ImportError as exc:
        raise RuntimeError(
            "diffusers is not installed. Install Z-Image support with:\n"
            "  pip install git+https://github.com/huggingface/diffusers.git -U"
        ) from exc

    try:
        return ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            local_files_only=local_files_only,
        )
    except (AttributeError, ValueError, OSError):
        from diffusers import DiffusionPipeline

        return DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            local_files_only=local_files_only,
        )


def make_generator(device, seed):
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except RuntimeError:
        return torch.Generator(device="cpu").manual_seed(seed)


def save_images(images, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if len(images) == 1:
        images[0].save(output)
        return [output]

    suffix = output.suffix or ".png"
    stem = output.with_suffix("")
    paths = []
    for index, image in enumerate(images, start=1):
        image_path = Path(f"{stem}_{index:02d}{suffix}")
        image.save(image_path)
        paths.append(image_path)
    return paths


def main():
    args = parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)

    print(f"Loading model: {args.model_path}")
    print(f"Device: {device}, dtype: {dtype}")
    pipe = load_pipeline(args.model_path, dtype, args.local_files_only)

    if args.cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if args.attention_backend and hasattr(pipe, "transformer"):
        pipe.transformer.set_attention_backend(args.attention_backend)

    if args.compile_transformer and hasattr(pipe, "transformer"):
        pipe.transformer.compile()

    generator = make_generator(device, args.seed)

    call_kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "num_images_per_prompt": args.num_images,
    }
    if args.negative_prompt:
        call_kwargs["negative_prompt"] = args.negative_prompt

    print("Generating image...")
    with torch.inference_mode():
        result = pipe(**call_kwargs)

    paths = save_images(result.images, args.output)
    for path in paths:
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
