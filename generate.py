"""
Ultra-Fast Image Gen - Command Line Interface

Supports all models available through the Gradio web interface:

  zimage-quant    Z-Image Turbo (quantized uint4, ~3.5 GB, fastest)
  zimage-full     Z-Image Turbo (full precision, LoRA support)
  flux2-4b-int8   FLUX.2-klein-4B (int8 quantized, img2img)
  flux2-4b-sdnq   FLUX.2-klein-4B (4bit SDNQ, img2img)
  flux2-9b-sdnq   FLUX.2-klein-9B (4bit SDNQ, higher quality, img2img)
  flux2-4b-uncensored  FLUX.2-klein-4B + uncensored Qwen3 GGUF text encoder
  flux2-4b-uncensored-mflux-hs  MFLUX/MLX q4 + uncensored GGUF TE + HS, fastest 2K path
  flux2-4b-uncensored-sdnq-hs   PyTorch SDNQ + uncensored GGUF TE + MPS/HS 2K path
  anima           Anima Turbo AIO Q4 (Metal runner, baked Turbo LoRA)
  bonsai-ternary  Bonsai Image 4B ternary (MLX, Apple Silicon)

Usage examples:
  python generate.py zimage-quant "a red fox in snow" --steps 5
  python generate.py zimage-full "a red fox" --lora my.safetensors --lora-strength 0.8
  python generate.py flux2-4b-sdnq "a red fox" --guidance 3.5 --steps 28
  python generate.py flux2-4b-int8 "edit the fox" --input-images ref.png --guidance 3.5
  python generate.py flux2-4b-uncensored-mflux-hs "a red fox" --width 2048 --height 2048
  python generate.py flux2-4b-uncensored-sdnq-hs "a red fox" --width 2048 --height 2048
  python generate.py anima "a red fox" --anima-preset Balanced
  python generate.py bonsai-ternary "a red fox" --steps 4
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


def fit_image_to_canvas(image, width: int, height: int):
    from PIL import Image

    image.thumbnail((width, height), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    left = (width - image.width) // 2
    top = (height - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def load_input_images(paths, width: int, height: int):
    from PIL import Image

    images = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: image not found, skipping: {path}")
            continue
        img = fit_image_to_canvas(Image.open(path).convert("RGB"), width, height)
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


def run_flux2_klein_uncensored_sdnq_hs(args):
    from flux2_sdnq_hs import (
        Flux2SdnqHsConfig,
        install_flux2_sdnq_hs_optimizations,
        reset_flux2_sdnq_hs_state,
    )
    from loaders import load_flux2_klein_uncensored_pipeline

    args.prompt = " ".join(args.prompt)
    device = resolve_device(args.device)
    if device != "mps":
        print("Warning: the optimized SDNQ HS path is tuned for MPS; continuing anyway.")

    pipe = load_flux2_klein_uncensored_pipeline(device, quant=args.gguf_quant)
    cfg = Flux2SdnqHsConfig.for_steps(
        args.steps,
        qchunk=args.qchunk,
        hs_stride=args.hs_stride,
        hs_skip_transformer_forwards=args.hs_skip_transformer_forwards,
        hs_max_transformer_forward=args.hs_max_transformer_forward,
        hs_single_start_frac=args.hs_single_start_frac,
        hs_single_end_frac=args.hs_single_end_frac,
        verbose=args.hs_verbose,
    )
    patch_cfg = install_flux2_sdnq_hs_optimizations(pipe, cfg)
    reset_flux2_sdnq_hs_state(pipe)
    generator, seed = make_generator(args.seed, device)
    input_images = []
    if args.input_images:
        input_images = load_input_images(args.input_images[:2], args.width, args.height)

    print(f"Generating with PyTorch SDNQ HS (seed {seed}, patch={patch_cfg})...")
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

    image.save(args.output)
    mode = f"img2img ({len(input_images)} ref)" if input_images else "txt2img"
    print(f"Saved to {args.output} (seed: {seed}, mode: {mode}, backend: pytorch-sdnq-hs)")


def run_flux2_klein_uncensored_mflux_hs(args):
    from mflux_hs_uncensored import generate_mflux_hs_uncensored

    prompt = " ".join(args.prompt)
    if args.width > 2048 or args.height > 2048:
        print("Warning: MFLUX HS preset is validated up to 2048px longest side.")
    seed = args.seed if args.seed is not None else torch.randint(0, 2**32, (1,)).item()

    print(f"Generating with MFLUX/MLX HS (seed {seed})...")
    mflux_model = getattr(args, "mflux_model", None)
    gguf_variant = getattr(args, "gguf_variant", None)
    if args.model == "flux2-9b-uncensored-mflux-hs":
        mflux_model = mflux_model or "flux2-klein-9b"
        gguf_variant = gguf_variant or "9b"
    else:
        mflux_model = mflux_model or "flux2-klein-4b"
        gguf_variant = gguf_variant or "4b"
    result = generate_mflux_hs_uncensored(
        prompt,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=seed,
        output_path=args.output,
        timeout=args.timeout,
        mflux_dir=args.mflux_dir,
        mflux_model=mflux_model,
        gguf_variant=gguf_variant,
        gguf_quant=args.gguf_quant,
        gguf_device=args.gguf_device,
        gguf_repo=getattr(args, "gguf_repo", None),
        gguf_subdir=getattr(args, "gguf_subdir", None),
        gguf_filename=getattr(args, "gguf_filename", None),
        hs_stride=args.hs_stride,
        hs_max_transformer_forward=args.hs_max_transformer_forward,
        hs_skip_transformer_forwards=args.hs_skip_transformer_forwards,
        hs_single_start_frac=args.hs_single_start_frac,
        hs_single_end_frac=args.hs_single_end_frac,
        hs_verbose=args.hs_verbose,
        input_image_paths=args.input_images[:2] if args.input_images else None,
    )
    print(result.log, end="")
    print(
        f"Saved to {result.path} "
        f"(seed: {result.seed}, wall: {result.elapsed_s:.1f}s, backend: mflux-hs)"
    )


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


def run_bonsai(args):
    # Bonsai runs in-process through prism-image-studio's MLX FluxPipeline.
    # Apple Silicon only; ignores --device (always MLX) and --guidance (the
    # 4-step distilled model takes none).
    import secrets

    from bonsai_mflux import load_bonsai_pipeline, generate_bonsai

    args.prompt = " ".join(args.prompt)
    seed = secrets.randbits(31) if args.seed is None else args.seed

    height = args.height
    width = args.width

    pipe = load_bonsai_pipeline()

    print(f"Generating with seed {seed}...")
    image, meta = generate_bonsai(
        pipe,
        args.prompt,
        height=height,
        width=width,
        steps=args.steps,
        seed=seed,
    )

    image.save(args.output)
    print(
        f"Saved to {args.output} "
        f"(seed: {meta['seed']}, {meta['width']}x{meta['height']}, steps: {meta['steps']})"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def make_common() -> argparse.ArgumentParser:
    """Parent parser: arguments common to every sub-command.

    Returns a fresh parser per call: argparse `parents=[...]` shares the
    parent's action OBJECTS with each child, so a sub-command calling
    set_defaults() (anima does, for height/width) would otherwise mutate
    action.default for every sub-command built from the same instance.
    """
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
    return common


def build_parser() -> argparse.ArgumentParser:
    speed_common = argparse.ArgumentParser(add_help=False)
    speed_common.add_argument("prompt", nargs="+", help="Text prompt for image generation (quoting optional)")
    speed_common.add_argument(
        "--height", type=int, default=2048, help="Image height in pixels (default: 2048)"
    )
    speed_common.add_argument(
        "--width", type=int, default=2048, help="Image width in pixels (default: 2048)"
    )
    speed_common.add_argument(
        "--seed", type=int, default=None, help="Random seed (random if omitted)"
    )
    speed_common.add_argument(
        "--output", default="output.png", help="Output file path (default: output.png)"
    )
    speed_common.add_argument(
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

    flux_fast_opts = argparse.ArgumentParser(add_help=False)
    flux_fast_opts.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of inference steps (default: 4)",
    )
    flux_fast_opts.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="Guidance scale (default: 0.0 for distilled klein)",
    )
    flux_fast_opts.add_argument("--gguf-quant", default="q4_k_m", choices=["q4_k_m", "q6_k", "q8_0"])
    flux_fast_opts.add_argument("--gguf-variant", default=None, choices=["4b", "9b"])
    flux_fast_opts.add_argument("--gguf-repo", default=None)
    flux_fast_opts.add_argument("--gguf-subdir", default=None)
    flux_fast_opts.add_argument("--gguf-filename", default=None)
    flux_fast_opts.add_argument("--qchunk", type=int, default=1024, help="MPS attention query chunk")
    flux_fast_opts.add_argument("--hs-stride", type=int, default=2, help="Hidden-state compression stride")
    flux_fast_opts.add_argument(
        "--hs-skip-transformer-forwards",
        type=int,
        default=0,
        help="Denoising transformer forwards to leave exact before compressing",
    )
    flux_fast_opts.add_argument(
        "--hs-max-transformer-forward",
        type=int,
        default=None,
        help="Exclusive denoising forward bound for HS compression (default: steps - 1)",
    )
    flux_fast_opts.add_argument("--hs-single-start-frac", type=float, default=0.0)
    flux_fast_opts.add_argument("--hs-single-end-frac", type=float, default=1.0)
    flux_fast_opts.add_argument("--hs-verbose", action="store_true")
    flux_fast_opts.add_argument(
        "--input-images",
        nargs="+",
        metavar="PATH",
        default=None,
        help="Input images for image-to-image editing (up to 2 paths in optimized app mode)",
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
        parents=[make_common()],
        help="Z-Image Turbo quantized uint4 (~3.5 GB, fastest)",
    )
    p.add_argument("--steps", type=int, default=5, help="Inference steps (default: 5)")

    # zimage-full
    p = sub.add_parser(
        "zimage-full",
        parents=[make_common()],
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
        parents=[make_common(), flux_opts],
        help="FLUX.2-klein-4B int8 quantized (supports img2img)",
    )

    # flux2-4b-sdnq
    sub.add_parser(
        "flux2-4b-sdnq",
        parents=[make_common(), flux_opts],
        help="FLUX.2-klein-4B 4bit SDNQ (supports img2img)",
    )

    # flux2-9b-sdnq
    sub.add_parser(
        "flux2-9b-sdnq",
        parents=[make_common(), flux_opts],
        help="FLUX.2-klein-9B 4bit SDNQ (higher quality, supports img2img)",
    )

    # flux2-4b-uncensored (SDNQ backbone + abliterated Qwen3 GGUF text encoder)
    sub.add_parser(
        "flux2-4b-uncensored",
        parents=[make_common(), flux_opts],
        help="FLUX.2-klein-4B with uncensored Qwen3 text encoder (q4_k_m GGUF, img2img)",
    )

    p = sub.add_parser(
        "flux2-4b-uncensored-sdnq-hs",
        parents=[speed_common, flux_fast_opts],
        help="PyTorch SDNQ uncensored backend with exact MPS chunked attention + HS speed path",
    )

    p = sub.add_parser(
        "flux2-4b-uncensored-mflux-hs",
        parents=[speed_common, flux_fast_opts],
        help="MFLUX/MLX q4 backend with uncensored GGUF TE + HS speed path (validated to 2K)",
    )
    p.add_argument(
        "--mflux-dir",
        default=None,
        help="Patched MFLUX checkout (default: ~/.cache/ultra-fast-image-gen/mflux; see scripts/setup_mflux_hs.sh)",
    )
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--gguf-device", default="mps")
    p.add_argument("--mflux-model", default="flux2-klein-4b")

    p = sub.add_parser(
        "flux2-9b-uncensored-mflux-hs",
        parents=[speed_common, flux_fast_opts],
        help="MFLUX/MLX q4 9B backend with 9B uncensored GGUF TE + HS speed path",
    )
    p.add_argument(
        "--mflux-dir",
        default=None,
        help="Patched MFLUX checkout (default: ~/.cache/ultra-fast-image-gen/mflux; see scripts/setup_mflux_hs.sh)",
    )
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--gguf-device", default="mps")
    p.add_argument("--mflux-model", default="flux2-klein-9b")

    # anima (external Metal runner, baked Turbo LoRA)
    p = sub.add_parser(
        "anima",
        parents=[make_common()],
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

    # bonsai-ternary (MLX, Apple Silicon)
    p = sub.add_parser(
        "bonsai-ternary",
        parents=[make_common()],
        help="Bonsai Image 4B ternary (MLX, Apple Silicon)",
    )
    p.add_argument("--steps", type=int, default=4, help="Inference steps (default: 4)")

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

    elif args.model == "flux2-4b-uncensored":
        from loaders import load_flux2_klein_uncensored_pipeline

        run_flux2_klein(
            args, lambda device: load_flux2_klein_uncensored_pipeline(device, quant="q4_k_m")
        )

    elif args.model == "flux2-4b-uncensored-sdnq-hs":
        run_flux2_klein_uncensored_sdnq_hs(args)

    elif args.model in ("flux2-4b-uncensored-mflux-hs", "flux2-9b-uncensored-mflux-hs"):
        run_flux2_klein_uncensored_mflux_hs(args)

    elif args.model == "anima":
        run_anima(args)

    elif args.model == "bonsai-ternary":
        run_bonsai(args)


if __name__ == "__main__":
    main()
