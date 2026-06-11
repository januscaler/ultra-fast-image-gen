#!/usr/bin/env python3
"""Run the current best MFLUX + HS compression + uncensored GGUF TE path.

This is a reproducibility wrapper around the patched MFLUX checkout (installed
by scripts/setup_mflux_hs.sh). It does not import MFLUX directly; it launches
the MFLUX CLI with the env knobs that produced the clean 2048x2048 result.
"""

from __future__ import annotations

import argparse
import sys

from mflux_hs_uncensored import generate_mflux_hs_uncensored, resolve_mflux_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run patched MFLUX FLUX.2-klein with HS compression and uncensored Qwen GGUF TE."
    )
    parser.add_argument(
        "--mflux-dir",
        default=None,
        help="Patched MFLUX checkout (default: ~/.cache/ultra-fast-image-gen/mflux; see scripts/setup_mflux_hs.sh)",
    )
    parser.add_argument("--prompt", default="a portrait of a cute woman, highly detailed")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--output", default="/tmp/mflux_uncensored_gguf_chat_hs_fixed_2048.png")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--quantize", default="4")
    parser.add_argument("--gguf-quant", default="q4_k_m")
    parser.add_argument("--gguf-device", default="mps")
    parser.add_argument("--hs-stride", type=int, default=2)
    parser.add_argument("--hs-max-transformer-forward", type=int, default=3)
    parser.add_argument("--hs-skip-transformer-forwards", default="0")
    parser.add_argument("--hs-single-start-frac", default="0.0")
    parser.add_argument("--hs-single-end-frac", default="1.0")
    parser.add_argument("--hs-verbose", action="store_true")
    parser.add_argument("--open-preview", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.mflux_dir = str(resolve_mflux_dir(args.mflux_dir))
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    print(
        "RUN patched MFLUX/MLX FLUX.2-klein",
        f"{args.width}x{args.height}",
        f"steps={args.steps}",
        f"seed={args.seed}",
        f"out={args.output}",
        flush=True,
    )
    print(
        "ENV",
        f"uncensored_gguf={args.gguf_quant}",
        f"hs=stride{args.hs_stride}/calls0-{args.hs_max_transformer_forward - 1}/final-exact",
        flush=True,
    )

    try:
        result = generate_mflux_hs_uncensored(
            args.prompt,
            height=args.height,
            width=args.width,
            steps=args.steps,
            seed=args.seed,
            output_path=args.output,
            timeout=args.timeout,
            mflux_dir=args.mflux_dir,
            quantize=args.quantize,
            gguf_quant=args.gguf_quant,
            gguf_device=args.gguf_device,
            hs_stride=args.hs_stride,
            hs_max_transformer_forward=args.hs_max_transformer_forward,
            hs_skip_transformer_forwards=int(args.hs_skip_transformer_forwards),
            hs_single_start_frac=float(args.hs_single_start_frac),
            hs_single_end_frac=float(args.hs_single_end_frac),
            hs_verbose=args.hs_verbose,
        )
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        return 124
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    print(result.log, end="", flush=True)
    print(f"OK elapsed={result.elapsed_s:.1f}s out={result.path}", flush=True)
    if args.open_preview:
        import subprocess

        subprocess.run(["open", "-a", "Preview", args.output], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
