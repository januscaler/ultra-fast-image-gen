"""
Generation engine shared by server.py (web UI) — pipeline cache, job queue,
and the background worker that runs generations one at a time.

This replaces the pipeline-management half of the old Gradio app.py. Models
are described in MODELS; the worker maps a model id to the right loader or
external runner and reports per-step progress where the pipeline supports it.
"""

import base64
import inspect
import io
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime

os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

from dotenv import load_dotenv
load_dotenv()

import torch
from PIL import Image

from anima_aio import (
    ANIMA_DEFAULTS,
    ANIMA_PRESETS,
    delete_anima_model,
    generate_anima_aio,
    get_anima_preset,
    get_anima_storage_entry,
)
from loaders import (
    load_zimage_pipeline,
    load_flux2_klein_pipeline,
    load_flux2_klein_sdnq_pipeline,
    load_flux2_klein_9b_sdnq_pipeline,
    load_flux2_klein_uncensored_pipeline,
)

try:
    from bonsai_mflux import (
        bonsai_available,
        bonsai_known_models,
        generate_bonsai,
        load_bonsai_pipeline,
    )
except Exception:  # optional extra not installed
    def bonsai_available():
        return False

    def bonsai_known_models():
        return {}

DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "ultra-fast-image-gen")
JOB_ROOT = os.path.join(tempfile.gettempdir(), "ufig_jobs")

GATED_TE_NOTE = (
    "Uncensored text encoder (~2.5GB GGUF) downloads on first use — the repo is "
    "gated on Hugging Face (instant auto-approval), so accept the terms on the "
    "model page and put HF_TOKEN=... in .env."
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = OrderedDict()


def _register(mid, **info):
    info["id"] = mid
    MODELS[mid] = info


_register(
    "mflux-hs-2k",
    label="FLUX.2-klein-4B Uncensored MFLUX HS",
    tag="2K Fast",
    defaults={"width": 2048, "height": 2048, "steps": 4, "guidance": 0.0},
    img2img=False,
    lora=False,
    progress="indeterminate",
    note=(
        "Fastest 2K lane: ~100s per image, ~7GB RAM, via the patched MFLUX/MLX "
        "runtime (one-time scripts/setup_mflux_hs.sh; Launch.command installs it). "
        + GATED_TE_NOTE
    ),
)
_register(
    "sdnq-hs-2k",
    label="FLUX.2-klein-4B Uncensored SDNQ HS",
    tag="PyTorch 2K",
    defaults={"width": 2048, "height": 2048, "steps": 4, "guidance": 0.0},
    img2img=False,
    lora=False,
    progress="steps",
    note=(
        "Pure PyTorch 2K lane, no extra setup: ~110s per image, low MPS memory. "
        "Shares the klein-4B (4bit SDNQ) base model. " + GATED_TE_NOTE
    ),
)
_register(
    "flux2-4b-sdnq",
    label="FLUX.2-klein-4B (4bit SDNQ)",
    tag="Low VRAM",
    defaults={"width": 512, "height": 512, "steps": 4, "guidance": 3.5},
    img2img=True,
    lora=False,
    progress="steps",
    note="Lowest-memory klein-4B. Supports image editing with up to 6 reference images.",
)
_register(
    "flux2-9b-sdnq",
    label="FLUX.2-klein-9B (4bit SDNQ)",
    tag="Higher Quality",
    defaults={"width": 512, "height": 512, "steps": 4, "guidance": 3.5},
    img2img=True,
    lora=False,
    progress="steps",
    note="Higher-quality 9B klein; needs more memory (~12GB @ 512px). Supports image editing.",
)
_register(
    "flux2-4b-int8",
    label="FLUX.2-klein-4B (Int8)",
    tag="",
    defaults={"width": 512, "height": 512, "steps": 4, "guidance": 3.5},
    img2img=True,
    lora=False,
    progress="steps",
    note="Int8-quantized klein-4B (~16GB). Supports image editing.",
)
_register(
    "flux2-4b-uncensored",
    label="FLUX.2-klein-4B Uncensored",
    tag="q4_k_m TE",
    defaults={"width": 512, "height": 512, "steps": 4, "guidance": 3.5},
    img2img=True,
    lora=False,
    progress="steps",
    note=(
        "Standard klein-4B pipeline with the uncensored text encoder, image editing "
        "included. Shares the klein-4B (4bit SDNQ) base model. " + GATED_TE_NOTE
    ),
)
_register(
    "zimage-quant",
    label="Z-Image Turbo (Quantized)",
    tag="Fastest",
    defaults={"width": 512, "height": 512, "steps": 5, "guidance": 0.0},
    img2img=False,
    lora=False,
    progress="steps",
    note="Fastest small model (~3.5GB download). Text-to-image only, no LoRA.",
)
_register(
    "anima",
    label="Anima Turbo AIO Q4 (Metal)",
    tag="",
    defaults={"width": ANIMA_DEFAULTS["width"], "height": ANIMA_DEFAULTS["height"],
              "steps": ANIMA_DEFAULTS["steps"], "guidance": ANIMA_DEFAULTS["guidance"]},
    img2img=False,
    lora=False,
    progress="indeterminate",
    anima_presets=list(ANIMA_PRESETS.keys()),
    note="External patched sd.cpp Metal runner with the Turbo LoRA baked in. GGUF auto-downloads on first use.",
)
_register(
    "bonsai",
    label="Bonsai Image 4B (Ternary MLX)",
    tag="Optional",
    defaults={"width": 512, "height": 512, "steps": 4, "guidance": 0.0},
    img2img=False,
    lora=False,
    progress="indeterminate",
    note="Ternary klein in-process on MLX, Apple Silicon only. Install with: uv sync --extra bonsai.",
)
_register(
    "zimage-full",
    label="Z-Image Turbo (Full)",
    tag="LoRA",
    defaults={"width": 512, "height": 512, "steps": 5, "guidance": 0.0},
    img2img=False,
    lora=True,
    progress="steps",
    note="Full-precision Z-Image (~24GB). Slower; use when you need LoRA support (path to a local .safetensors).",
)


def mflux_hs_available():
    try:
        from mflux_hs_uncensored import resolve_mflux_dir

        resolve_mflux_dir()
        return True
    except Exception:
        return False


def list_models():
    """Model registry for the API, with availability flags resolved now."""
    out = []
    for mid, m in MODELS.items():
        if mid == "bonsai" and not bonsai_available():
            continue
        entry = {k: v for k, v in m.items()}
        if mid == "mflux-hs-2k":
            entry["setup_required"] = not mflux_hs_available()
        out.append(entry)
    return out


def get_devices():
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

_pipe = None
_pipe_model = None
_pipe_device = None
_lora_path = None


def _free_pipe():
    global _pipe, _pipe_model, _pipe_device, _lora_path
    if _pipe is not None:
        del _pipe
        _pipe = None
    _pipe_model = None
    _pipe_device = None
    _lora_path = None
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_pipe(model_id, device):
    """Load (or reuse) the in-process pipeline for a model id."""
    global _pipe, _pipe_model, _pipe_device
    if _pipe is not None and _pipe_model == model_id and _pipe_device == device:
        return _pipe
    _free_pipe()

    if model_id == "flux2-4b-int8":
        _pipe = load_flux2_klein_pipeline(device)
    elif model_id == "flux2-4b-sdnq":
        _pipe = load_flux2_klein_sdnq_pipeline(device)
    elif model_id == "flux2-9b-sdnq":
        _pipe = load_flux2_klein_9b_sdnq_pipeline(device)
    elif model_id == "flux2-4b-uncensored":
        _pipe = load_flux2_klein_uncensored_pipeline(device, quant="q4_k_m")
    elif model_id == "sdnq-hs-2k":
        from flux2_sdnq_hs import Flux2SdnqHsConfig, install_flux2_sdnq_hs_optimizations

        _pipe = load_flux2_klein_uncensored_pipeline(device, quant="q4_k_m")
        cfg = Flux2SdnqHsConfig.for_steps(
            4, qchunk=1024, hs_stride=2, hs_skip_transformer_forwards=0,
            hs_max_transformer_forward=3, hs_single_start_frac=0.0, hs_single_end_frac=1.0,
        )
        install_flux2_sdnq_hs_optimizations(_pipe, cfg)
    elif model_id == "zimage-quant":
        _pipe = load_zimage_pipeline(device, use_full_model=False)
    elif model_id == "zimage-full":
        _pipe = load_zimage_pipeline(device, use_full_model=True)
    elif model_id == "bonsai":
        _pipe = load_bonsai_pipeline()
    else:
        raise ValueError(f"unknown in-process model: {model_id}")

    _pipe_model = model_id
    _pipe_device = device
    return _pipe


def _apply_lora(pipe, lora_path, strength):
    global _lora_path
    if not lora_path:
        if _lora_path:
            pipe.unload_lora_weights()
            _lora_path = None
        return
    lora_path = os.path.expanduser(lora_path)
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")
    if _lora_path != lora_path:
        if _lora_path:
            pipe.unload_lora_weights()
        pipe.load_lora_weights(lora_path, adapter_name="default")
        _lora_path = lora_path
    pipe.set_adapters(["default"], adapter_weights=[float(strength)])


# ---------------------------------------------------------------------------
# Job queue + worker
# ---------------------------------------------------------------------------

_jobs = {}
_queue = []
_lock = threading.Lock()
_wake = threading.Event()


def submit_job(params):
    """Queue a generation job; returns the job id."""
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",
        "created": time.time(),
        "started": None,
        "finished": None,
        "params": params,
        "progress": None,      # 0..1 float, or None for indeterminate
        "stage": "Queued",
        "images": [],          # [{url, seed, file}]
        "error": None,
    }
    with _lock:
        _jobs[job_id] = job
        _queue.append(job_id)
    _wake.set()
    return job_id


def get_job(job_id):
    return _jobs.get(job_id)


def cancel_job(job_id):
    """Cancel a job if it is still queued. Running jobs can't be aborted."""
    with _lock:
        job = _jobs.get(job_id)
        if job and job["status"] == "queued":
            _queue.remove(job_id)
            job["status"] = "cancelled"
            job["stage"] = "Cancelled"
            return True
    return False


def queue_position(job_id):
    with _lock:
        try:
            return _queue.index(job_id)
        except ValueError:
            return None


def current_model():
    return _pipe_model


def _job_dir(job_id):
    d = os.path.join(JOB_ROOT, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _save_result(job, idx, image, seed, auto_save, output_dir, prompt):
    path = os.path.join(_job_dir(job["id"]), f"{idx}.png")
    image.save(path, "PNG")
    entry = {"url": f"/api/files/{job['id']}/{idx}.png", "seed": seed, "file": path}
    if auto_save:
        out_dir = os.path.expanduser(output_dir or DEFAULT_OUTPUT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in (prompt or "")[:30]).strip("_")
        saved = os.path.join(out_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}_{slug}.png")
        shutil.copyfile(path, saved)
        entry["saved"] = saved
    job["images"].append(entry)


def _decode_images(data_urls):
    images = []
    for du in (data_urls or [])[:6]:
        payload = du.split(",", 1)[1] if "," in du else du
        img = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        images.append(img)
    return images


def _supports_step_callback(pipe):
    try:
        return "callback_on_step_end" in inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False


def _run_diffusers(job, model_id, p):
    device = p.get("device") or get_devices()[0]
    job["stage"] = "Loading model"
    pipe = _get_pipe(model_id, device)

    if model_id == "zimage-full":
        job["stage"] = "Loading LoRA" if p.get("lora_path") else job["stage"]
        _apply_lora(pipe, p.get("lora_path"), p.get("lora_strength", 1.0))

    input_images = _decode_images(p.get("input_images")) if MODELS[model_id]["img2img"] else []
    width, height = int(p["width"]), int(p["height"])
    steps = int(p["steps"])
    guidance = float(p.get("guidance", 0.0))
    count = max(1, min(8, int(p.get("count", 1))))
    base_seed = p.get("seed")
    if base_seed in (None, -1, ""):
        base_seed = torch.randint(0, 2**31, (1,)).item()
    base_seed = int(base_seed)

    if input_images:
        resized = []
        for img in input_images:
            r = img.copy().resize((width, height), Image.LANCZOS)
            resized.append(r.convert("RGB"))
        input_images = resized
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "disable_tiling"):
            pipe.vae.disable_tiling()

    use_callback = _supports_step_callback(pipe)

    try:
        _run_diffusers_loop(job, model_id, p, pipe, input_images, width, height,
                            steps, guidance, count, base_seed, device, use_callback)
    finally:
        # The pipe stays cached across jobs, so tiling must come back on even
        # if a generation raised (it is what keeps large renders in memory).
        if input_images and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_diffusers_loop(job, model_id, p, pipe, input_images, width, height,
                        steps, guidance, count, base_seed, device, use_callback):
    for i in range(count):
        seed = base_seed + i
        job["stage"] = f"Generating {i + 1}/{count}" if count > 1 else "Generating"
        if device in ("mps", "cuda"):
            generator = torch.Generator(device).manual_seed(seed)
        else:
            generator = torch.Generator().manual_seed(seed)

        kwargs = dict(
            prompt=p["prompt"],
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if input_images:
            kwargs["image"] = input_images if len(input_images) > 1 else input_images[0]
        if use_callback:
            def _cb(pipeline, step, timestep, cb_kwargs, _i=i):
                job["progress"] = (_i + (step + 1) / steps) / count
                return cb_kwargs

            kwargs["callback_on_step_end"] = _cb
        else:
            job["progress"] = None

        if model_id == "sdnq-hs-2k":
            from flux2_sdnq_hs import (
                Flux2SdnqHsConfig,
                install_flux2_sdnq_hs_optimizations,
                reset_flux2_sdnq_hs_state,
            )

            cfg = Flux2SdnqHsConfig.for_steps(
                steps, qchunk=1024, hs_stride=2, hs_skip_transformer_forwards=0,
                hs_max_transformer_forward=max(0, steps - 1),
                hs_single_start_frac=0.0, hs_single_end_frac=1.0,
            )
            install_flux2_sdnq_hs_optimizations(pipe, cfg)
            reset_flux2_sdnq_hs_state(pipe)

        with torch.inference_mode():
            image = pipe(**kwargs).images[0]

        _save_result(job, i, image, seed, p.get("auto_save"), p.get("output_dir"), p["prompt"])
        job["progress"] = (i + 1) / count


def _run_mflux_hs(job, p):
    from mflux_hs_uncensored import generate_mflux_hs_uncensored

    count = max(1, min(8, int(p.get("count", 1))))
    base_seed = p.get("seed")
    if base_seed in (None, -1, ""):
        base_seed = torch.randint(0, 2**31, (1,)).item()
    base_seed = int(base_seed)

    for i in range(count):
        job["stage"] = f"Generating {i + 1}/{count} (MFLUX/MLX)" if count > 1 else "Generating (MFLUX/MLX)"
        out = os.path.join(_job_dir(job["id"]), f"{i}.png")
        result = generate_mflux_hs_uncensored(
            p["prompt"],
            height=int(p["height"]),
            width=int(p["width"]),
            steps=int(p["steps"]),
            seed=base_seed + i,
            output_path=out,
            timeout=300,
        )
        _save_result(job, i, result.image, base_seed + i, p.get("auto_save"), p.get("output_dir"), p["prompt"])
        job["progress"] = (i + 1) / count


def _run_anima(job, p):
    count = max(1, min(8, int(p.get("count", 1))))
    preset = get_anima_preset(p.get("anima_preset") or "Balanced")
    base_seed = p.get("seed")
    if base_seed in (None, -1, ""):
        base_seed = -1
    for i in range(count):
        job["stage"] = f"Generating {i + 1}/{count} (Anima Metal)" if count > 1 else "Generating (Anima Metal)"
        seed = -1 if base_seed == -1 else int(base_seed) + i
        result = generate_anima_aio(
            p["prompt"],
            height=int(p["height"]),
            width=int(p["width"]),
            steps=int(p["steps"]),
            seed=seed,
            cfg_scale=float(p.get("guidance", 1.0)),
            cache_mode=preset["cache_mode"],
            output_dir=None,
        )
        _save_result(job, i, result["image"], result["seed"], p.get("auto_save"), p.get("output_dir"), p["prompt"])
        job["progress"] = (i + 1) / count


def _run_bonsai(job, p):
    import secrets

    pipe = _get_pipe("bonsai", "mlx")
    count = max(1, min(8, int(p.get("count", 1))))
    base_seed = p.get("seed")
    if base_seed in (None, -1, ""):
        base_seed = secrets.randbits(31)
    base_seed = int(base_seed)
    for i in range(count):
        job["stage"] = f"Generating {i + 1}/{count} (MLX)" if count > 1 else "Generating (MLX)"
        image, meta = generate_bonsai(
            pipe, p["prompt"],
            height=int(p["height"]), width=int(p["width"]),
            steps=int(p["steps"]), seed=base_seed + i,
        )
        _save_result(job, i, image, meta["seed"], p.get("auto_save"), p.get("output_dir"), p["prompt"])
        job["progress"] = (i + 1) / count


def _friendly_error(exc):
    msg = str(exc)
    if "gated" in msg.lower() or "401" in msg:
        return (
            "The uncensored text encoder repo is gated on Hugging Face. Accept the "
            "terms at huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder "
            "(instant auto-approval), put HF_TOKEN=... in .env, and retry."
        )
    if isinstance(exc, FileNotFoundError):
        return msg
    tail = msg[-600:] if len(msg) > 600 else msg
    return f"{type(exc).__name__}: {tail}"


def _worker():
    while True:
        _wake.wait()
        while True:
            with _lock:
                if not _queue:
                    _wake.clear()
                    break
                job_id = _queue.pop(0)
            job = _jobs[job_id]
            job["status"] = "running"
            job["started"] = time.time()
            model_id = job["params"].get("model")
            try:
                if model_id not in MODELS:
                    raise ValueError(f"unknown model: {model_id}")
                if model_id == "mflux-hs-2k":
                    _run_mflux_hs(job, job["params"])
                elif model_id == "anima":
                    _run_anima(job, job["params"])
                elif model_id == "bonsai":
                    _run_bonsai(job, job["params"])
                else:
                    _run_diffusers(job, model_id, job["params"])
                job["status"] = "done"
                job["stage"] = "Done"
                job["progress"] = 1.0
            except Exception as exc:
                traceback.print_exc()
                job["status"] = "error"
                job["stage"] = "Error"
                job["error"] = _friendly_error(exc)
            finally:
                job["finished"] = time.time()


_worker_thread = threading.Thread(target=_worker, daemon=True, name="ufig-worker")
_worker_thread.start()


# ---------------------------------------------------------------------------
# Storage management (ported from the old Gradio app)
# ---------------------------------------------------------------------------

KNOWN_MODELS = {
    "aydin99/FLUX.2-klein-4B-int8": "FLUX.2-klein-4B (Int8)",
    "black-forest-labs/FLUX.2-klein-4B": "FLUX.2-klein-4B (Base)",
    "black-forest-labs/FLUX.2-klein-9B": "FLUX.2-klein-9B (Base)",
    "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic": "FLUX.2-klein-4B (4bit SDNQ)",
    "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32": "FLUX.2-klein-9B (4bit SDNQ)",
    "ponpoke/flux2-klein-4b-uncensored-text-encoder": "FLUX.2-klein Uncensored Text Encoder GGUF",
    "Tongyi-MAI/Z-Image-Turbo": "Z-Image Turbo (Full)",
    "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32": "Z-Image Turbo (Quantized)",
    "filipstrand/Z-Image-Turbo-mflux-4bit": "Z-Image Turbo (mflux 4bit)",
}
KNOWN_MODELS.update(bonsai_known_models())


def _hf_cache_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _dir_size(path):
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total


def format_size(n):
    for unit, div in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.2f} {unit}" if unit == "GB" else f"{n / div:.1f} {unit}"
    return f"{n} B"


def scan_storage():
    cache_dir = _hf_cache_dir()
    models = []
    total = 0
    if os.path.exists(cache_dir):
        for repo_id, display in KNOWN_MODELS.items():
            cache_name = f"models--{repo_id.replace('/', '--')}"
            path = os.path.join(cache_dir, cache_name)
            if os.path.exists(path):
                size = _dir_size(path)
                total += size
                models.append({
                    "key": cache_name, "repo_id": repo_id, "name": display,
                    "size": size, "size_str": format_size(size),
                })
    anima_entry = get_anima_storage_entry()
    if anima_entry is not None:
        total += anima_entry["size"]
        models.append({
            "key": "anima", "repo_id": "anima", "name": anima_entry["display_name"],
            "size": anima_entry["size"], "size_str": format_size(anima_entry["size"]),
            "external": "anima",
        })
    models.sort(key=lambda m: m["size"], reverse=True)
    return {"models": models, "total": total, "total_str": format_size(total)}


# HF repos each in-process pipeline actually loads, so storage deletes only
# unload the live pipe when its own weights are being removed.
_MODEL_REPOS = {
    "flux2-4b-int8": ("aydin99/FLUX.2-klein-4B-int8", "black-forest-labs/FLUX.2-klein-4B"),
    "flux2-4b-sdnq": ("Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic", "black-forest-labs/FLUX.2-klein-4B"),
    "flux2-9b-sdnq": ("Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32", "black-forest-labs/FLUX.2-klein-9B"),
    "flux2-4b-uncensored": ("Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic", "ponpoke/flux2-klein-4b-uncensored-text-encoder"),
    "sdnq-hs-2k": ("Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic", "ponpoke/flux2-klein-4b-uncensored-text-encoder"),
    "zimage-quant": ("Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",),
    "zimage-full": ("Tongyi-MAI/Z-Image-Turbo",),
}


def _any_job_active():
    with _lock:
        return any(j["status"] in ("queued", "running") for j in _jobs.values())


def delete_storage(key):
    """Delete one cached model by its key from scan_storage()."""
    # Refuse while work is in flight: a delete could rmtree the HF cache dir a
    # running job is downloading into, or free the pipe under the worker.
    if _any_job_active():
        return False, "A generation is queued or running — wait for it to finish before deleting models."

    info = scan_storage()
    target = next((m for m in info["models"] if m["key"] == key), None)
    if target is None:
        return False, f"Model not found: {key}"

    # Unload the live pipeline if it uses these weights (safe: worker is idle).
    if _pipe is not None and (
        target["repo_id"] in _MODEL_REPOS.get(_pipe_model or "", ())
        or (_pipe_model == "bonsai" and "bonsai" in target["repo_id"].lower())
    ):
        _free_pipe()

    try:
        if target.get("external") == "anima":
            delete_anima_model()
        else:
            shutil.rmtree(os.path.join(_hf_cache_dir(), target["key"]))
        return True, f"Deleted {target['name']} ({target['size_str']} freed)"
    except Exception as exc:
        return False, f"Error deleting {target['name']}: {exc}"
