# Ultra Fast Image Gen

AI image generation and editing on Mac Silicon and CUDA. Generate images from text or transform existing images with state-of-the-art diffusion models.

## Features

- **Image Generation:** Create images from text prompts
- **Image Editing:** Upload up to 6 reference images and transform them with natural language
- **Multiple Models:** FLUX.2-klein and Z-Image Turbo
- **Quantized Models:** Low memory usage with 4bit/int8 quantization
- **Anima Turbo AIO:** Local patched Metal runner with Turbo LoRA baked into the GGUF
- **Uncensored FLUX.2 2K lanes:** MFLUX/MLX fast path and PyTorch SDNQ fallback with MPS optimizations
- **LoRA Support:** Load custom LoRA adapters with Z-Image Full model
- **Cross-Platform:** Apple Silicon (MPS) and NVIDIA GPUs (CUDA)

## Supported Models

| Model | VRAM | Features | Speed |
|-------|------|----------|-------|
| FLUX.2-klein-4B Uncensored MFLUX HS | ~7GB RSS @ 2K | Text-to-image, uncensored GGUF TE, validated to 2K | Fastest current 2K FLUX lane |
| FLUX.2-klein-4B Uncensored SDNQ HS | ~8GB MPS / low RAM @ 2K | Text-to-image, uncensored GGUF TE, exact chunked MPS attention + HS | PyTorch 2K fallback |
| FLUX.2-klein-4B (4bit SDNQ) | <8GB @ 512px, <16GB @ 1024px | Text-to-image + Image editing | Fast |
| FLUX.2-klein-9B (4bit SDNQ) | ~12GB @ 512px, ~20GB @ 1024px | Text-to-image + Image editing (Higher Quality) | Fast |
| FLUX.2-klein-4B (Int8) | ~16GB | Text-to-image + Image editing | Fast |
| Z-Image Turbo (Quantized) | ~8GB | Text-to-image | Fastest |
| Anima Turbo AIO Q4 (Metal) | ~3GB model + unified memory | Text-to-image, baked Turbo LoRA | ~16s internal @ 512x768 / 8 steps |
| Z-Image Turbo (Full) | ~24GB | Text-to-image + LoRA | Slower |

The uncensored variants do not re-download a separate base model: the SDNQ HS
lane and the plain uncensored model reuse the FLUX.2-klein-4B (4bit SDNQ)
backbone, so the only extra download is the uncensored Qwen3 text encoder
(~2.5GB GGUF at the default `q4_k_m` quant).

> **Note:** the [uncensored text encoder repo](https://huggingface.co/ponpoke/flux2-klein-4b-uncensored-text-encoder)
> is gated on Hugging Face (instant auto-approval). Accept the terms on the
> model page once, then put a token in a `.env` file at the repo root:
> `HF_TOKEN=hf_...`

## Quick Start (1-Click)

1. Download/clone the repo
2. **Double-click `Launch.command`**
3. First run will auto-install dependencies (~5 min); later runs reinstall if `requirements.txt` changed
4. The launcher installs/builds the patched Anima Metal runner if needed
5. The launcher installs the patched MFLUX runtime for the Uncensored 2K fast lane if needed
6. The Anima GGUF auto-downloads on first Anima generation
7. Browser opens automatically to the UI

## Manual Installation

```bash
git clone https://github.com/newideas99/ultra-fast-image-gen.git
cd ultra-fast-image-gen

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# For the uncensored models (gated text-encoder repo):
echo "HF_TOKEN=hf_your_token_here" > .env
```

### Anima Fresh Install

`Launch.command` runs this automatically when the Anima runner is missing:

```bash
scripts/setup_anima_metal_runner.sh
```

The setup script clones `stable-diffusion.cpp`, checks out the tested revision,
applies the bundled ggml Metal patch for Anima VAE ops, builds `sd-cli` with
Metal enabled, and writes `~/anima-comfyui/run_anima_aio_metal.sh`.

The downloaded `Anima-P3-Turbo-AIO-Q4_K.gguf` already has the Anima Turbo LoRA
merged in. The app does not need a separate `anima-turbo-lora-v0.1.safetensors`
file for this model.

### MFLUX HS 2K Lane Fresh Install

`Launch.command` runs this automatically when the MFLUX runtime is missing:

```bash
scripts/setup_mflux_hs.sh
```

The setup script clones [mflux](https://github.com/filipstrand/mflux) at the
tested 0.17.5 revision into `~/.cache/ultra-fast-image-gen/mflux`, applies the
bundled hidden-state-compression + uncensored-GGUF-text-encoder patch
(`patches/mflux_hs_uncensored_gguf.patch`), and pre-builds the `uv`
environment. Only the "FLUX.2-klein-4B Uncensored MFLUX HS" model needs this;
the SDNQ HS 2K lane runs on the normal Python dependencies. Override the
install location with `ULTRA_FAST_MFLUX_HS_DIR`.

## Usage

### Web UI

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### Model Selection

- **FLUX.2-klein-4B Uncensored MFLUX HS:** Default. Fastest 2K text-to-image lane (~100s @ 2048x2048). Uses the patched MFLUX runtime (`scripts/setup_mflux_hs.sh`) plus the uncensored Qwen GGUF text encoder
- **FLUX.2-klein-4B Uncensored SDNQ HS:** PyTorch SDNQ 2K text-to-image lane with exact MPS query chunking and hidden-state compression; no extra setup
- **FLUX.2-klein-4B (4bit SDNQ):** Lowest memory, supports image editing
- **FLUX.2-klein-9B (4bit SDNQ):** Higher quality 9B model, more memory
- **FLUX.2-klein-4B (Int8):** Alternative quantization, more memory
- **Z-Image Turbo (Quantized):** Fastest text-to-image, no image editing
- **Anima Turbo AIO Q4 (Metal):** Uses `~/anima-comfyui/run_anima_aio_metal.sh`; auto-downloads the Turbo AIO GGUF if missing and defaults to 512x768, 8 steps, CFG 1
- **Z-Image Turbo (Full):** Use when you need LoRA support

### Image Editing (FLUX.2-klein)

1. Select a classic FLUX.2-klein model from the dropdown (4bit SDNQ, 9B, or Int8 — the 2K uncensored lanes are text-to-image in the UI)
2. Upload up to 6 images in the gallery
3. Write a prompt describing the changes you want
4. Select output resolution (1024px, 1280px, or 1536px)
5. Click Generate

### Command Line

Each model has its own sub-command with the options it needs:

```bash
# Z-Image Turbo (quantized) — fastest, ~3.5 GB
python generate.py zimage-quant a beautiful sunset over mountains

# Z-Image Turbo (full precision) — with optional LoRA
python generate.py zimage-full a beautiful sunset --lora my.safetensors --lora-strength 0.8

# FLUX.2-klein-4B (4bit SDNQ)
python generate.py flux2-4b-sdnq a beautiful sunset --guidance 3.5 --steps 28

# FLUX.2-klein-4B (Int8)
python generate.py flux2-4b-int8 a beautiful sunset --guidance 3.5 --steps 28

# FLUX.2-klein-9B (4bit SDNQ) — higher quality
python generate.py flux2-9b-sdnq a beautiful sunset --guidance 3.5 --steps 28

# FLUX.2-klein-4B Uncensored MFLUX/MLX HS — fastest current 2K path
python generate.py flux2-4b-uncensored-mflux-hs a beautiful sunset --width 2048 --height 2048 --steps 4

# FLUX.2-klein-4B Uncensored PyTorch SDNQ HS — optimized PyTorch/MPS 2K path
python generate.py flux2-4b-uncensored-sdnq-hs a beautiful sunset --width 2048 --height 2048 --steps 4

# Image-to-image editing (FLUX.2-klein models only)
python generate.py flux2-4b-sdnq transform the fox into a wolf --input-images ref.png

# Anima Turbo AIO (Metal runner, baked Turbo LoRA)
python generate.py anima anime portrait, detailed eyes --anima-preset Balanced
```

Quotes around the prompt are optional — all words before the first `--flag` are joined into the prompt.

**Common options** (all sub-commands):

| Option | Default | Description |
|--------|---------|-------------|
| `--height` | 512 (Anima: 768) | Image height in pixels |
| `--width` | 512 | Image width in pixels |
| `--seed` | random | Fixed seed for reproducibility |
| `--output` | output.png | Output file path |
| `--device` | mps | `mps`, `cuda`, or `cpu` |

**Z-Image options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 5 | Inference steps |
| `--lora` | — | Path to LoRA `.safetensors` (`zimage-full` only) |
| `--lora-strength` | 1.0 | LoRA weight (`zimage-full` only) |

**FLUX.2-klein options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 28 | Inference steps |
| `--guidance` | 3.5 | Classifier-free guidance scale |
| `--input-images` | — | Up to 6 reference images for editing |

**Uncensored 2K speed-lane options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 4 | Distilled klein inference steps |
| `--guidance` | 0.0 | Guidance scale |
| `--gguf-quant` | q4_k_m | Uncensored Qwen GGUF text encoder quant |
| `--qchunk` | 1024 | PyTorch MPS attention query chunk (`sdnq-hs` only) |
| `--hs-stride` | 2 | Hidden-state compression stride |
| `--hs-max-transformer-forward` | steps - 1 | Leave the final transformer forward exact |
| `--mflux-dir` | ~/.cache/ultra-fast-image-gen/mflux | Patched MFLUX checkout (`mflux-hs` only; see `scripts/setup_mflux_hs.sh`) |

**Anima options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | from preset | Inference steps (overrides the preset) |
| `--cfg-scale` | 1.0 | Anima CFG scale |
| `--anima-preset` | Balanced | `Fast` (3 steps), `Balanced` (8), or `Quality` (16) |

## Benchmarks

### FLUX.2-klein-4B

| Hardware | Resolution | Steps | Time |
|----------|------------|-------|------|
| Apple Silicon | 512x512 | 4 | ~8s |
| CUDA (RTX 3090) | 512x512 | 4 | ~3s |

### FLUX.2-klein-4B Uncensored 2K Speed Lanes

| Backend | Resolution | Steps | Time | Notes |
|---------|------------|-------|------|-------|
| MFLUX/MLX HS | 2048x2048 | 4 | 100.2s fresh-process wall / ~69s denoise | Includes uncensored GGUF TE load in fresh CLI process |
| PyTorch SDNQ HS | 2048x2048 | 4 | ~110s generation wall | Exact query-chunked MPS attention + HS compression |

### Z-Image Turbo (Quantized)

| Mac | Resolution | Steps | Time |
|-----|------------|-------|------|
| M2 Max | 512x512 | 7 | 14s |
| M2 Max | 768x768 | 7 | 31s |
| M1 Max | 512x512 | 7 | 23s |

### Anima Turbo AIO Q4 (Metal)

Recommended settings:
- Fast: 3 steps with Spectrum cache
- Balanced/default: 8 steps with Spectrum cache
- Quality/model-card default: 16 steps with cache disabled

| Mac | Resolution | Steps | Time |
|-----|------------|-------|------|
| M2 Max | 512x768 | 3 | 8.82s internal / 11.63s wall |
| M2 Max | 512x768 | 4 | 11.13s internal / 13.65s wall |
| M2 Max | 512x768 | 8 | 15.62s internal / 18.69s wall |

## Memory Requirements

| Model | RAM/VRAM Required |
|-------|-------------------|
| FLUX.2-klein-4B Uncensored MFLUX HS | ~7GB RSS @ 2048px |
| FLUX.2-klein-4B Uncensored SDNQ HS | Low MPS memory @ 2048px; slower PyTorch fallback |
| FLUX.2-klein-4B (4bit SDNQ) | 8GB @ 512px, 16GB @ 1024px |
| FLUX.2-klein-9B (4bit SDNQ) | 12GB @ 512px, 20GB @ 1024px |
| FLUX.2-klein-4B (Int8) | 16GB |
| Z-Image (Quantized) | 8GB |
| Z-Image (Full) | 24GB+ |
| Anima Turbo AIO Q4 (Metal) | 32GB recommended for local setup |

## Credits

- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) by Black Forest Labs
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) by Alibaba
- [Anima](https://huggingface.co/circlestone-labs/Anima) by Circlestone Labs
- [SDNQ Quantization](https://huggingface.co/Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic) by Disty0
- [Int8 Quantization](https://huggingface.co/aydin99/FLUX.2-klein-4B-int8) using optimum-quanto

## License

See the original model licenses for usage terms.
