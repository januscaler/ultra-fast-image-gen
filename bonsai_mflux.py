"""Adapter for Bonsai Image 4B via prism-image-studio's MLX FluxPipeline.

Apple Silicon only. The FLUX.2 forward pass and the ternary (2-bit affine)
quant live in prism-image-studio (`backend.pipeline.FluxPipeline`) plus the
mflux-prism fork of mflux; this module is the thin loader + generation glue,
parallel to anima_aio.py.

Scope: ternary only. The 1-bit/binary arm needed a source-built mlx fork whose
Metal kernels don't compile on current macOS, so it's dropped. Ternary uses
standard 2-bit affine quant, which runs on stock pypi `mlx` (prebuilt wheel,
no Xcode) -- see the `mlx` override in pyproject.toml.

    ternary -> bonsai-ternary-mlx -> prism-ml/bonsai-image-ternary-4B-mlx-2bit
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


BONSAI_MODEL_CHOICE = "Bonsai Image 4B (Ternary - MLX)"
BONSAI_MODEL_CHOICES = [BONSAI_MODEL_CHOICE]

BONSAI_BACKEND = "bonsai-ternary-mlx"
BONSAI_REPO_ID = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"

# Bonsai is a 4-step distilled model; guidance is unused (kept at 0 for the UI).
BONSAI_DEFAULTS = {
    "width": 512,
    "height": 512,
    "steps": 4,
    "guidance": 0.0,
}


def bonsai_available() -> bool:
    """True if the optional bonsai extra (mflux) is installed.

    `mflux` only lands via `uv sync --extra bonsai`, so its presence is a
    reliable proxy for the whole MLX stack being there.
    """
    import importlib.util

    return sys.platform == "darwin" and importlib.util.find_spec("mflux") is not None


def bonsai_known_models() -> dict[str, str]:
    """repo_id -> display name, for merging into app.py's KNOWN_MODELS."""
    return {BONSAI_REPO_ID: "Bonsai Image 4B (Ternary MLX)"}


def is_bonsai_model_choice(model_choice: str | None) -> bool:
    return isinstance(model_choice, str) and model_choice.startswith("Bonsai Image 4B")


def _round16(value: int) -> int:
    """Bonsai requires dimensions in [256, 2048] that are multiples of 16."""
    value = int(value)
    value = (value // 16) * 16
    return max(256, min(2048, value))


def load_bonsai_pipeline(*, evict_text_encoder: bool = True):
    """Load the Bonsai ternary MLX pipeline."""
    if sys.platform != "darwin":
        raise RuntimeError("Bonsai Image (MLX) runs on Apple Silicon only.")

    from huggingface_hub import snapshot_download

    try:
        from backend.pipeline import FluxPipeline, PipelineConfig
    except ImportError as e:
        raise RuntimeError(
            "Bonsai support is not installed. Install the optional extra:\n"
            "  uv sync --extra bonsai\n"
            "Apple Silicon + python >=3.11 required."
        ) from e

    print("Loading Bonsai Image 4B (ternary, MLX)...")
    model_root = snapshot_download(BONSAI_REPO_ID, token=HF_TOKEN)

    pipeline = FluxPipeline(
        PipelineConfig(
            backend=BONSAI_BACKEND,
            baked_model_path=str(model_root),
            te_4bit=True,
            evict_text_encoder=evict_text_encoder,
        )
    )
    print("  Bonsai Image 4B (ternary) ready!")
    return pipeline


def generate_bonsai(
    pipeline,
    prompt: str,
    *,
    height: int = BONSAI_DEFAULTS["height"],
    width: int = BONSAI_DEFAULTS["width"],
    steps: int = BONSAI_DEFAULTS["steps"],
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Run one generation. Returns (PIL.Image, metadata dict)."""
    from PIL import Image

    w = _round16(width)
    h = _round16(height)
    seed = int(seed)

    png_bytes = pipeline.generate_png(
        prompt=prompt or "",
        seed=seed,
        steps=int(steps),
        height=h,
        width=w,
    )

    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    image.load()

    return image, {"width": w, "height": h, "steps": int(steps), "seed": seed}
