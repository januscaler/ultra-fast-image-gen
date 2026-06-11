"""External MFLUX/MLX backend for the fast uncensored 2K path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import signal
import subprocess
import time

from PIL import Image

# The gated text-encoder download inside the subprocess needs HF_TOKEN; load it
# here so the CLI path gets .env credentials too (app.py loads dotenv itself).
from dotenv import load_dotenv
load_dotenv()

DEFAULT_MFLUX_DIR = Path.home() / ".cache" / "ultra-fast-image-gen" / "mflux"
LEGACY_MFLUX_DIR = Path("/tmp/mflux")
SETUP_HINT = (
    "Run scripts/setup_mflux_hs.sh to install the patched MFLUX runtime "
    "(Launch.command does this automatically), or use the 'FLUX.2-klein-4B "
    "Uncensored SDNQ HS' model which needs no extra setup."
)


def resolve_mflux_dir(mflux_dir: str | None = None) -> Path:
    """Locate the patched MFLUX checkout, raising a setup hint if absent."""
    if mflux_dir:
        candidates = [Path(mflux_dir).expanduser()]
    else:
        candidates = []
        env_dir = os.environ.get("ULTRA_FAST_MFLUX_HS_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        candidates.extend([DEFAULT_MFLUX_DIR, LEGACY_MFLUX_DIR])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    looked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Missing patched MFLUX checkout (looked in: {looked}). {SETUP_HINT}")


@dataclass
class MfluxHsResult:
    image: Image.Image
    path: str
    seed: int
    elapsed_s: float
    log: str


def generate_mflux_hs_uncensored(
    prompt: str,
    *,
    height: int = 2048,
    width: int = 2048,
    steps: int = 4,
    seed: int = 1234,
    output_path: str = "/tmp/mflux_uncensored_gguf_chat_hs_fixed_2048.png",
    timeout: int = 180,
    mflux_dir: str | None = None,
    mflux_model: str = "flux2-klein-4b",
    quantize: str = "4",
    gguf_variant: str = "4b",
    gguf_quant: str = "q4_k_m",
    gguf_device: str = "mps",
    gguf_repo: str | None = None,
    gguf_subdir: str | None = None,
    gguf_filename: str | None = None,
    hs_stride: int = 2,
    hs_max_transformer_forward: int | None = None,
    hs_skip_transformer_forwards: int = 0,
    hs_single_start_frac: float = 0.0,
    hs_single_end_frac: float = 1.0,
    hs_verbose: bool = False,
    input_image_paths: list[str] | None = None,
) -> MfluxHsResult:
    """Generate through the patched MFLUX checkout and return a PIL image."""
    mflux_path = resolve_mflux_dir(mflux_dir)
    if shutil.which("uv") is None:
        raise FileNotFoundError(
            "The 'uv' command is required to run the MFLUX HS lane. "
            "Install it (e.g. 'brew install uv') and try again."
        )

    if hs_max_transformer_forward is None:
        hs_max_transformer_forward = max(0, int(steps) - 1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    command_name = "mflux-generate-flux2-edit" if input_image_paths else "mflux-generate-flux2"
    cmd = [
        "uv",
        "run",
        "--project",
        str(mflux_path),
        "--with",
        "gguf",
        "--with",
        "accelerate",
        "--with",
        "python-dotenv",
        command_name,
        "--model",
        mflux_model,
        "--quantize",
        str(quantize),
        "--prompt",
        prompt,
        "--steps",
        str(int(steps)),
        "--seed",
        str(int(seed)),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--output",
        str(output),
    ]
    if input_image_paths:
        cmd.extend(["--image-paths", *[str(path) for path in input_image_paths[:2]]])

    env = os.environ.copy()
    env.update(
        {
            "MFLUX_DISABLE_COMPILE": "1",
            "MFLUX_UNCENSORED_GGUF_TE": "1",
            "MFLUX_SKIP_STOCK_TEXT_ENCODER": "1",
            "MFLUX_UNCENSORED_GGUF_VARIANT": gguf_variant,
            "MFLUX_UNCENSORED_GGUF_QUANT": gguf_quant,
            "MFLUX_UNCENSORED_GGUF_DEVICE": gguf_device,
            "MFLUX_UNCENSORED_GGUF_REPO_ROOT": str(Path(__file__).resolve().parent),
            "MFLUX_HS_STRIDE": str(int(hs_stride)),
            "MFLUX_HS_SKIP_TRANSFORMER_FORWARDS": str(int(hs_skip_transformer_forwards)),
            "MFLUX_HS_MAX_TRANSFORMER_FORWARD": str(int(hs_max_transformer_forward)),
            "MFLUX_HS_SINGLE_START_FRAC": str(float(hs_single_start_frac)),
            "MFLUX_HS_SINGLE_END_FRAC": str(float(hs_single_end_frac)),
            "MFLUX_HS_VERBOSE": "1" if hs_verbose else "0",
        }
    )
    if gguf_repo:
        env["MFLUX_UNCENSORED_GGUF_REPO"] = gguf_repo
    if gguf_subdir is not None:
        env["MFLUX_UNCENSORED_GGUF_SUBDIR"] = gguf_subdir
    if gguf_filename:
        env["MFLUX_UNCENSORED_GGUF_FILENAME"] = gguf_filename

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )
    try:
        log, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            log, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            log, _ = proc.communicate()
        raise TimeoutError(f"MFLUX generation timed out after {timeout}s\n{log}")

    elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"MFLUX generation failed with code {proc.returncode}\n{log}")

    image = Image.open(output).convert("RGB")
    return MfluxHsResult(image=image, path=str(output), seed=int(seed), elapsed_s=elapsed, log=log)
