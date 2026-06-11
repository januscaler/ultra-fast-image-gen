"""
Model loader functions shared between server.py (web UI) and generate.py (CLI).
Keeping them here keeps the CLI free of web-server imports.
"""

import os
import json

os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

import torch


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def print_memory(label):
    """Print memory usage with label."""
    mem = get_memory_usage()
    print(f"  [MEM] {label}: {mem:.2f} GB")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_zimage_pipeline(device="mps", use_full_model=False):
    """Load Z-Image pipeline (quantized or full)."""
    import sdnq  # Required for quantized model
    from diffusers import ZImagePipeline, FlowMatchEulerDiscreteScheduler

    if use_full_model:
        print(f"Loading Z-Image-Turbo (full precision) on {device}...")
        dtype = torch.bfloat16 if device in ["mps", "cuda"] else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:
        print(f"Loading Z-Image-Turbo UINT4 (quantized) on {device}...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = ZImagePipeline.from_pretrained(
            "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        use_beta_sigmas=True,
    )

    pipe.to(device)
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()

    return pipe


def load_flux2_klein_pipeline(device="mps"):
    """Load FLUX.2-klein-4B with int8 quantized transformer and text encoder."""
    from diffusers import Flux2KleinPipeline
    from transformers import Qwen3ForCausalLM, AutoTokenizer, AutoConfig
    from optimum.quanto import requantize
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from quantized_flux2 import QuantizedFlux2Transformer2DModel

    print(f"Loading FLUX.2-klein-4B (int8 quantized) on {device}...")
    print_memory("Before loading")

    model_path = snapshot_download("aydin99/FLUX.2-klein-4B-int8")

    print("  Loading int8 transformer...")
    qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(model_path)
    qtransformer.to(device=device, dtype=torch.bfloat16)
    print_memory("After transformer")

    print("  Loading int8 text encoder...")
    config = AutoConfig.from_pretrained(f"{model_path}/text_encoder", trust_remote_code=True)
    with init_empty_weights():
        text_encoder = Qwen3ForCausalLM(config)

    with open(f"{model_path}/text_encoder/quanto_qmap.json", "r") as f:
        qmap = json.load(f)
    state_dict = load_file(f"{model_path}/text_encoder/model.safetensors")
    requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
    text_encoder.eval()
    text_encoder.to(device, dtype=torch.bfloat16)
    print_memory("After text encoder")

    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")

    print("  Loading VAE and scheduler...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        transformer=None,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    print_memory("After VAE/scheduler download")

    pipe.transformer = qtransformer._wrapped
    pipe.text_encoder = text_encoder
    pipe.tokenizer = tokenizer
    pipe.to(device)
    print_memory("After pipe.to(device)")

    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")

    print("  FLUX.2-klein-4B ready!")
    return pipe


def load_flux2_klein_sdnq_pipeline(device="mps"):
    from sdnq import SDNQConfig
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer

    print(f"Loading FLUX.2-klein-4B (4bit SDNQ) on {device}...")
    print_memory("Before loading")

    print("  Loading tokenizer from base model (SDNQ model missing vocab files)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        subfolder="tokenizer",
        use_fast=False,
        token=HF_TOKEN,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )
    print_memory("After loading")

    pipe.to(device)
    print_memory("After pipe.to(device)")

    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")

    print("  FLUX.2-klein-4B (SDNQ) ready!")
    return pipe


def load_flux2_klein_9b_sdnq_pipeline(device="mps"):
    from sdnq import SDNQConfig
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer

    print(f"Loading FLUX.2-klein-9B (4bit SDNQ) on {device}...")
    print_memory("Before loading")

    print("  Loading tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        subfolder="tokenizer",
        use_fast=False,
        token=HF_TOKEN,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )
    print_memory("After loading")

    pipe.to(device)
    print_memory("After pipe.to(device)")

    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")

    print("  FLUX.2-klein-9B (SDNQ) ready!")
    return pipe


# ---------------------------------------------------------------------------
# FLUX.2-klein-4B with the uncensored (abliterated) Qwen3 text encoder
#
# The text encoder is loaded from a GGUF file and kept QUANTIZED in RAM,
# dequantized on the fly per forward (see gguf_qwen3.py). The 4bit SDNQ
# transformer + VAE are reused as a constant image backbone so the chosen
# text-encoder quant is the only thing that varies between runs.
# ---------------------------------------------------------------------------

UNCENSORED_TE_REPO = "ponpoke/flux2-klein-4b-uncensored-text-encoder"
UNCENSORED_TE_SUBDIR = "flux2-klein-4b-uncensored-text-encoder"
UNCENSORED_TE_QUANTS = {
    "q4_k_m": "flux2-klein-4b-uncensored-q4_k_m.gguf",
    "q6_k": "flux2-klein-4b-uncensored-q6_k.gguf",
    "q8_0": "flux2-klein-4b-uncensored-q8_0.gguf",
}


def _download_uncensored_te(quant):
    """Fetch the chosen GGUF plus the config/tokenizer; return (gguf_path, cfg_dir)."""
    from huggingface_hub import hf_hub_download

    if quant not in UNCENSORED_TE_QUANTS:
        raise ValueError(f"unknown quant {quant!r}; choose from {list(UNCENSORED_TE_QUANTS)}")

    gguf_path = hf_hub_download(UNCENSORED_TE_REPO, UNCENSORED_TE_QUANTS[quant], token=HF_TOKEN)
    cfg_path = hf_hub_download(
        UNCENSORED_TE_REPO, f"{UNCENSORED_TE_SUBDIR}/config.json", token=HF_TOKEN
    )
    for f in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        try:
            hf_hub_download(UNCENSORED_TE_REPO, f"{UNCENSORED_TE_SUBDIR}/{f}", token=HF_TOKEN)
        except Exception:
            pass
    return gguf_path, os.path.dirname(cfg_path)


def load_flux2_klein_uncensored_pipeline(device="mps", quant="q4_k_m"):
    """FLUX.2-klein-4B with the uncensored Qwen3 text encoder from a GGUF (quant kept in RAM)."""
    from sdnq import SDNQConfig  # noqa: F401  (registers the SDNQ quant config)
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer
    from gguf_qwen3 import load_qwen3_gguf_text_encoder

    print(f"Loading FLUX.2-klein-4B UNCENSORED (TE quant: {quant}) on {device}...")
    print_memory("Before loading")

    gguf_path, te_dir = _download_uncensored_te(quant)

    print("  Loading SDNQ transformer + VAE backbone...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
    )
    print_memory("After backbone")

    print(f"  Loading uncensored Qwen3 text encoder from GGUF ({quant})...")
    text_encoder, quant_counts = load_qwen3_gguf_text_encoder(
        gguf_path, te_dir, device=device, compute_dtype=torch.bfloat16
    )
    print(f"  TE quantized tensors: {quant_counts}")
    print_memory("After text encoder")

    pipe.text_encoder = text_encoder
    pipe.tokenizer = AutoTokenizer.from_pretrained(te_dir)
    pipe.to(device)
    print_memory("After pipe.to(device)")

    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    print_memory("After memory optimizations")

    print("  FLUX.2-klein-4B UNCENSORED ready!")
    return pipe
