"""
Ultra-Fast Image Gen - Command Line Interface

Supports all models available through the Gradio web interface:

  zimage-quant    Z-Image Turbo (quantized uint4, ~3.5 GB, fastest)
  zimage-full     Z-Image Turbo (full precision, LoRA support)
  flux2-4b-int8   FLUX.2-klein-4B (int8 quantized, img2img)
  flux2-4b-sdnq   FLUX.2-klein-4B (4bit SDNQ, img2img)
  flux2-9b-sdnq   FLUX.2-klein-9B (4bit SDNQ, higher quality, img2img)
  anima           Anima Turbo AIO Q4 (Metal runner, baked Turbo LoRA)

Usage examples:
  python generate.py zimage-quant "a red fox in snow" --steps 5
  python generate.py zimage-full "a red fox" --lora my.safetensors --lora-strength 0.8
  python generate.py flux2-4b-sdnq "a red fox" --guidance 3.5 --steps 28
  python generate.py flux2-4b-int8 "edit the fox" --input-images ref.png --guidance 3.5
  python generate.py anima "a red fox" --anima-preset Balanced
"""

import os
import argparse
import importlib

os.environ["PYTORCH_MPS_FAST_MATH"] = "1"


# CC BY-SA 4.0 https://stackoverflow.com/a/78312617
class _LazyLoader:
    """Defers module import until first attribute access."""

    def __init__(self, modname):
        self._modname = modname
        self._mod = None

    def __getattr__(self, attr):
        try:
            return getattr(self._mod, attr)
        except Exception as e:
            if self._mod is None:
                self._mod = importlib.import_module(self._modname)
            else:
                raise e
        return getattr(self._mod, attr)


torch = _LazyLoader("torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_device(requested: str) -> str:
    device = requested
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    return device


def make_generator(seed, device: str):
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    if device == "cuda":
        gen = torch.Generator("cuda").manual_seed(seed)
    elif device == "mps":
        gen = torch.Generator("mps").manual_seed(seed)
    else:
        gen = torch.Generator().manual_seed(seed)
    return gen, seed


def load_input_images(paths, width: int, height: int):
    from PIL import Image

    images = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: image not found, skipping: {path}")
            continue
        img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# Model handlers
# ---------------------------------------------------------------------------


def run_zimage_quant(args):
    from loaders import load_zimage_pipeline

    args.prompt = " ".join(args.prompt)
    device = resolve_device(args.device)
    pipe = load_zimage_pipeline(device, use_full_model=False)
    generator, seed = make_generator(args.seed, device)

    print(f"Generating with seed {seed}...")
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

    image.save(args.output)
    print(f"Saved to {args.output} (seed: {seed})")


def run_zimage_full(args):
    from loaders import load_zimage_pipeline

    args.prompt = " ".join(args.prompt)
    device = resolve_device(args.device)
    pipe = load_zimage_pipeline(device, use_full_model=True)

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

    generator, seed = make_generator(args.seed, device)

    print(f"Generating with seed {seed}...")
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

    image.save(args.output)
    lora_info = f", LoRA: {os.path.basename(args.lora)}" if args.lora else ""
    print(f"Saved to {args.output} (seed: {seed}{lora_info})")


def run_flux2_klein(args, loader_fn):
    args.prompt = " ".join(args.prompt)
    device = resolve_device(args.device)
    pipe = loader_fn(device)

    input_images = []
    if args.input_images:
        input_images = load_input_images(args.input_images[:6], args.width, args.height)
        if (
            input_images
            and hasattr(pipe, "vae")
            and hasattr(pipe.vae, "disable_tiling")
        ):
            pipe.vae.disable_tiling()

    generator, seed = make_generator(args.seed, device)

    print(f"Generating with seed {seed}...")
    with torch.inference_mode():
        if input_images:
            image = pipe(
                prompt=args.prompt,
                image=input_images[0] if len(input_images) == 1 else input_images,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
        else:
            image = pipe(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]

    if input_images and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    image.save(args.output)
    mode = f"img2img ({len(input_images)} ref)" if input_images else "txt2img"
    print(f"Saved to {args.output} (seed: {seed}, mode: {mode})")


def run_anima(args):
    # Anima runs through an external Metal runner (no torch / diffusers).
    from anima_aio import ANIMA_DEFAULTS, generate_anima_aio, get_anima_preset

    args.prompt = " ".join(args.prompt)
    preset = get_anima_preset(args.anima_preset)

    # height/width/steps default to None for this sub-command (see set_defaults
    # in build_parser) so the Anima-specific defaults can be applied here.
    height = args.height if args.height is not None else ANIMA_DEFAULTS["height"]
    width = args.width if args.width is not None else ANIMA_DEFAULTS["width"]
    steps = args.steps if args.steps is not None else preset["steps"]

    result = generate_anima_aio(
        args.prompt,
        height=height,
        width=width,
        steps=steps,
        seed=-1 if args.seed is None else args.seed,
        cfg_scale=args.cfg_scale,
        cache_mode=preset["cache_mode"],
        output_path=args.output,
    )
    timing = f", gen: {result['generation_time']}" if result.get("generation_time") else ""
    print(
        f"Saved to {result['path']} "
        f"(seed: {result['seed']}, preset: {args.anima_preset}, "
        f"cache: {result.get('cache_mode')}{timing})"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    # Parent parser: arguments common to every sub-command
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("prompt", nargs="+", help="Text prompt for image generation (quoting optional)")
    common.add_argument(
        "--height", type=int, default=512, help="Image height in pixels (default: 512)"
    )
    common.add_argument(
        "--width", type=int, default=512, help="Image width in pixels (default: 512)"
    )
    common.add_argument(
        "--seed", type=int, default=None, help="Random seed (random if omitted)"
    )
    common.add_argument(
        "--output", default="output.png", help="Output file path (default: output.png)"
    )
    common.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Compute device (default: mps)",
    )

    # Parent parser: extra arguments shared by FLUX.2-klein sub-commands
    flux_opts = argparse.ArgumentParser(add_help=False)
    flux_opts.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps (default: 28)",
    )
    flux_opts.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale (default: 3.5)",
    )
    flux_opts.add_argument(
        "--input-images",
        nargs="+",
        metavar="PATH",
        default=None,
        help="Input images for image-to-image editing (up to 6 paths)",
    )

    parser = argparse.ArgumentParser(
        description="Command-line image generation with multiple model backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="model", required=True, metavar="MODEL")

    # zimage-quant
    p = sub.add_parser(
        "zimage-quant",
        parents=[common],
        help="Z-Image Turbo quantized uint4 (~3.5 GB, fastest)",
    )
    p.add_argument("--steps", type=int, default=5, help="Inference steps (default: 5)")

    # zimage-full
    p = sub.add_parser(
        "zimage-full",
        parents=[common],
        help="Z-Image Turbo full precision (supports LoRA)",
    )
    p.add_argument("--steps", type=int, default=5, help="Inference steps (default: 5)")
    p.add_argument("--lora", default=None, help="Path to a LoRA .safetensors file")
    p.add_argument(
        "--lora-strength", type=float, default=1.0, help="LoRA strength (default: 1.0)"
    )

    # flux2-4b-int8
    sub.add_parser(
        "flux2-4b-int8",
        parents=[common, flux_opts],
        help="FLUX.2-klein-4B int8 quantized (supports img2img)",
    )

    # flux2-4b-sdnq
    sub.add_parser(
        "flux2-4b-sdnq",
        parents=[common, flux_opts],
        help="FLUX.2-klein-4B 4bit SDNQ (supports img2img)",
    )

    # flux2-9b-sdnq
    sub.add_parser(
        "flux2-9b-sdnq",
        parents=[common, flux_opts],
        help="FLUX.2-klein-9B 4bit SDNQ (higher quality, supports img2img)",
    )

    # anima (external Metal runner, baked Turbo LoRA)
    p = sub.add_parser(
        "anima",
        parents=[common],
        help="Anima Turbo AIO Q4 (Metal runner, baked Turbo LoRA)",
    )
    p.add_argument(
        "--steps", type=int, default=None, help="Inference steps (default: from preset)"
    )
    p.add_argument(
        "--cfg-scale", type=float, default=1.0, help="Anima CFG scale (default: 1.0)"
    )
    p.add_argument(
        "--anima-preset",
        default="Balanced",
        choices=["Fast", "Balanced", "Quality"],  # keys of anima_aio.ANIMA_PRESETS
        help="Fast=3 steps + cache, Balanced=8 + cache, Quality=16 no cache",
    )
    # Anima uses its own height/width defaults (see run_anima), so unset -> None.
    p.set_defaults(height=None, width=None)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.model == "zimage-quant":
        run_zimage_quant(args)

    elif args.model == "zimage-full":
        run_zimage_full(args)

    elif args.model == "flux2-4b-int8":
        from loaders import load_flux2_klein_pipeline

        run_flux2_klein(args, load_flux2_klein_pipeline)

    elif args.model == "flux2-4b-sdnq":
        from loaders import load_flux2_klein_sdnq_pipeline

        run_flux2_klein(args, load_flux2_klein_sdnq_pipeline)

    elif args.model == "flux2-9b-sdnq":
        from loaders import load_flux2_klein_9b_sdnq_pipeline

        run_flux2_klein(args, load_flux2_klein_9b_sdnq_pipeline)

    elif args.model == "anima":
        run_anima(args)


if __name__ == "__main__":
    main()
