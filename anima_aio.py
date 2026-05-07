"""Adapter for the local Anima Turbo AIO Metal runner.

The actual model/runtime lives outside this repo under ~/anima-comfyui by
default. This module lets the UIs treat it like a normal selectable model while
delegating generation to the patched sd.cpp path.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


ANIMA_MODEL_CHOICE = "Anima Turbo AIO Q4 (Metal - Fast)"
ANIMA_MODEL_TYPE = "anima-aio-metal"

ANIMA_DEFAULTS = {
    "width": 512,
    "height": 768,
    "steps": 8,
    "guidance": 1.0,
}

ANIMA_MODEL_REPO_ID = "n-Arno/Anima-P3-Turbo-AIO-Q4_K"
ANIMA_MODEL_FILENAME = "Anima-P3-Turbo-AIO-Q4_K.gguf"
DEFAULT_ANIMA_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "ultra-fast-image-gen")


def get_anima_root() -> str:
    return os.path.expanduser(os.environ.get("ULTRA_FAST_ANIMA_ROOT", os.path.join("~", "anima-comfyui")))


def get_anima_runner_path() -> str:
    default_runner = os.path.join(get_anima_root(), "run_anima_aio_metal.sh")
    return os.path.expanduser(os.environ.get("ULTRA_FAST_ANIMA_RUNNER", default_runner))


def get_anima_model_path() -> str:
    default_model = os.path.join(get_anima_root(), "sdcpp", "models", ANIMA_MODEL_FILENAME)
    return os.path.expanduser(os.environ.get("ULTRA_FAST_ANIMA_MODEL", default_model))


def is_anima_model_choice(model_choice: str | None) -> bool:
    return model_choice == ANIMA_MODEL_CHOICE or (
        isinstance(model_choice, str) and model_choice.startswith("Anima Turbo AIO")
    )


def get_anima_status() -> tuple[bool, str]:
    runner = Path(get_anima_runner_path())
    model = Path(get_anima_model_path())
    if not runner.exists():
        return False, f"Anima runner not found: {runner}"
    if not os.access(runner, os.X_OK):
        return False, f"Anima runner is not executable: {runner}"
    if not model.exists():
        return False, f"Anima model not downloaded: {model}"
    return True, f"Ready: {runner}"


def is_anima_model_downloaded() -> bool:
    return Path(get_anima_model_path()).exists()


def download_anima_model() -> str:
    """Download the Anima AIO GGUF into the path expected by the local runner."""
    target = Path(get_anima_model_path())
    if target.exists():
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=ANIMA_MODEL_REPO_ID,
            filename=ANIMA_MODEL_FILENAME,
            local_dir=str(target.parent),
        )
    )

    if downloaded != target and downloaded.exists() and not target.exists():
        shutil.copy2(downloaded, target)

    if not target.exists():
        raise RuntimeError(f"Anima model download finished but file is missing: {target}")

    return str(target)


def ensure_anima_ready(download: bool = True) -> tuple[bool, str]:
    runner = Path(get_anima_runner_path())
    if not runner.exists():
        return False, f"Anima runner not found: {runner}"
    if not os.access(runner, os.X_OK):
        return False, f"Anima runner is not executable: {runner}"

    if not is_anima_model_downloaded():
        if not download:
            return False, f"Anima model not downloaded: {get_anima_model_path()}"
        download_anima_model()

    return get_anima_status()


def format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    return f"{size_bytes / 1024**3:.2f} GB"


def get_anima_storage_entry() -> dict[str, Any] | None:
    model = Path(get_anima_model_path())
    if not model.exists():
        return None

    size = model.stat().st_size
    return {
        "repo_id": ANIMA_MODEL_REPO_ID,
        "display_name": "Anima Turbo AIO Q4 (Metal)",
        "cache_name": ANIMA_MODEL_FILENAME,
        "path": str(model),
        "size": size,
        "size_str": format_bytes(size),
        "external": "anima",
    }


def delete_anima_model() -> bool:
    model = Path(get_anima_model_path())
    if not model.exists():
        return False
    model.unlink()
    return True


def _safe_slug(prompt: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in prompt[:30]).strip("_")
    return slug or "prompt"


def _read_log_tail(log_path: str, max_chars: int = 4000) -> str:
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _parse_first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def generate_anima_aio(
    prompt: str,
    *,
    height: int = ANIMA_DEFAULTS["height"],
    width: int = ANIMA_DEFAULTS["width"],
    steps: int = ANIMA_DEFAULTS["steps"],
    seed: int | None = -1,
    cfg_scale: float = ANIMA_DEFAULTS["guidance"],
    output_dir: str | None = None,
    output_path: str | None = None,
    negative_prompt: str | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    """Generate an image through the local Anima AIO Metal runner."""
    ready, status = ensure_anima_ready(download=True)
    if not ready:
        raise RuntimeError(status)

    prompt = prompt or ""
    seed_value = -1 if seed is None else int(seed)

    if output_path:
        out_path = Path(os.path.expanduser(output_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if output_dir:
            base_dir = Path(os.path.expanduser(output_dir))
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = Path(tempfile.mkdtemp(prefix="ultra_fast_anima_"))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = base_dir / f"anima_aio_{width}x{height}_{steps}s_{stamp}_{_safe_slug(prompt)}.png"

    log_path = out_path.with_suffix(".log")

    env = os.environ.copy()
    env.update(
        {
            "WIDTH": str(int(width)),
            "HEIGHT": str(int(height)),
            "STEPS": str(int(steps)),
            "CFG_SCALE": str(float(cfg_scale)),
            "SEED": str(seed_value),
            "OPEN_OUTPUT": "0",
            "MODEL": get_anima_model_path(),
            "OUT": str(out_path),
            "LOG": str(log_path),
            "PYTHON_CMD": sys.executable,
        }
    )
    if negative_prompt:
        env["NEGATIVE_PROMPT"] = negative_prompt

    result = subprocess.run(
        [get_anima_runner_path(), prompt],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    log_text = result.stdout + result.stderr + _read_log_tail(str(log_path))
    if result.returncode != 0:
        raise RuntimeError(f"Anima generation failed with exit code {result.returncode}\n{log_text[-4000:]}")

    if not out_path.exists():
        raise RuntimeError(f"Anima generation finished but no output image was written: {out_path}\n{log_text[-4000:]}")

    image = Image.open(out_path).convert("RGB")
    image.load()

    resolved_seed = _parse_first(r"generating image:\s+\d+/\d+\s+-\s+seed\s+(\d+)", log_text)
    generation_time = _parse_first(r"generate_image completed in ([0-9.]+s)", log_text)
    wall_time = _parse_first(r"\breal\s+([0-9.]+)", log_text)

    return {
        "image": image,
        "seed": int(resolved_seed) if resolved_seed is not None else seed_value,
        "path": str(out_path),
        "log_path": str(log_path),
        "generation_time": generation_time,
        "wall_time": f"{wall_time}s" if wall_time else None,
        "log": log_text,
    }
