"""
Z-Image Turbo UINT4 - Fast Image Generation on Mac

Uses the quantized uint4 model (only 3.5GB!) for fast inference on Apple Silicon.
Now with LoRA support!
"""

import os
import argparse

# Enable fast-math for MPS
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

from anima_aio import ANIMA_DEFAULTS, ANIMA_MODEL_TYPE, ANIMA_PRESETS, generate_anima_aio, get_anima_preset


def load_pipeline(device="mps"):
    """Load the full-precision Z-Image pipeline."""
    import torch
    from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler

    print("Loading Z-Image-Turbo (full precision)...")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"PyTorch version: {torch.__version__}")

    # Use bfloat16 for better quality
    dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Use Euler with beta sigmas for cleaner images
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_beta_sigmas=True,
    )

    pipe.to(device)

    # Memory optimizations
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        print("VAE slicing enabled")

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
        print("VAE tiling enabled")

    print("Pipeline loaded!")
    return pipe


def generate(
    pipe,
    prompt: str,
    height: int = 512,
    width: int = 512,
    steps: int = 5,
    seed: int = None,
    device: str = "mps",
):
    """Generate an image from a prompt."""
    import torch

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    print(f"Generating with seed {seed}...")

    # Use appropriate generator for device
    if device == "cuda":
        generator = torch.Generator("cuda").manual_seed(seed)
    elif device == "mps":
        generator = torch.Generator("mps").manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

    return image, seed


def main():
    parser = argparse.ArgumentParser(description="Generate images with Z-Image Turbo or Anima AIO")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument(
        "--model",
        type=str,
        default="zimage-full",
        choices=["zimage-full", ANIMA_MODEL_TYPE],
        help="Model/runtime to use (default: zimage-full)",
    )
    parser.add_argument("--height", type=int, default=None, help="Image height (default: 512, Anima: 768)")
    parser.add_argument("--width", type=int, default=None, help="Image width (default: 512)")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps (default: 5, Anima: 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output path")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Anima CFG scale (default: 1.0)")
    parser.add_argument(
        "--anima-preset",
        type=str,
        default="Balanced",
        choices=list(ANIMA_PRESETS.keys()),
        help="Anima preset: Fast=3 steps + cache, Balanced=8 + cache, Quality=16 no cache",
    )

    # LoRA arguments
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA safetensors file")
    parser.add_argument("--lora-strength", type=float, default=1.0, help="LoRA strength (default: 1.0)")

    args = parser.parse_args()

    if args.model == ANIMA_MODEL_TYPE:
        anima_preset = get_anima_preset(args.anima_preset)
        args.height = args.height or ANIMA_DEFAULTS["height"]
        args.width = args.width or ANIMA_DEFAULTS["width"]
        args.steps = args.steps or anima_preset["steps"]
    else:
        args.height = args.height or 512
        args.width = args.width or 512
        args.steps = args.steps or 5

    if args.model == ANIMA_MODEL_TYPE:
        anima_preset = get_anima_preset(args.anima_preset)
        result = generate_anima_aio(
            args.prompt,
            height=args.height,
            width=args.width,
            steps=args.steps,
            seed=-1 if args.seed is None else args.seed,
            cfg_scale=args.cfg_scale,
            cache_mode=anima_preset["cache_mode"],
            output_path=args.output,
        )
        timing = f", gen: {result['generation_time']}" if result.get("generation_time") else ""
        print(
            f"Saved to {result['path']} "
            f"(seed: {result['seed']}, preset: {args.anima_preset}, cache: {result.get('cache_mode')}{timing})"
        )
        return

    import torch

    # Determine device
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    pipe = load_pipeline(device)

    # Load LoRA if specified (using native diffusers support)
    if args.lora:
        if not os.path.exists(args.lora):
            print(f"Error: LoRA file not found: {args.lora}")
            return

        print(f"Loading LoRA: {args.lora} (strength={args.lora_strength})")
        try:
            pipe.load_lora_weights(args.lora, adapter_name="default")
            pipe.set_adapters(["default"], adapter_weights=[args.lora_strength])
            print("LoRA loaded successfully!")
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            return

    image, seed = generate(
        pipe,
        args.prompt,
        args.height,
        args.width,
        args.steps,
        args.seed,
        device,
    )

    image.save(args.output)
    lora_info = f", LoRA: {os.path.basename(args.lora)}" if args.lora else ""
    print(f"Saved to {args.output} (seed: {seed}{lora_info})")


if __name__ == "__main__":
    main()
