#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANIMA_ROOT="${ULTRA_FAST_ANIMA_ROOT:-$HOME/anima-comfyui}"
SDCPP_SRC="${ULTRA_FAST_ANIMA_SDCPP_SRC:-$ANIMA_ROOT/sdcpp-src}"
BUILD_DIR="${ULTRA_FAST_ANIMA_BUILD_DIR:-$SDCPP_SRC/build-metal-im2col3d}"
RUNNER="${ULTRA_FAST_ANIMA_RUNNER:-$ANIMA_ROOT/run_anima_aio_metal.sh}"
MODEL="${ULTRA_FAST_ANIMA_MODEL:-$ANIMA_ROOT/sdcpp/models/Anima-P3-Turbo-AIO-Q4_K.gguf}"
SDCLI="$BUILD_DIR/bin/sd-cli"
PATCH_FILE="$APP_DIR/patches/anima-ggml-metal-im2col3d-pad.patch"

SDCPP_REPO="${ULTRA_FAST_ANIMA_SDCPP_REPO:-https://github.com/leejet/stable-diffusion.cpp.git}"
SDCPP_COMMIT="${ULTRA_FAST_ANIMA_SDCPP_COMMIT:-90e87bc846f17059771efb8aaa31e9ef0cab6f78}"

die() {
    echo "Error: $*" >&2
    exit 1
}

install_homebrew_if_needed() {
    if command -v brew >/dev/null 2>&1; then
        return
    fi

    echo "Homebrew is required to install cmake."
    read -r -p "Install Homebrew now? (y/n) " install_brew
    if [ "$install_brew" != "y" ] && [ "$install_brew" != "Y" ]; then
        die "Homebrew/cmake is required to build the Anima Metal runner."
    fi

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || eval "$(/usr/local/bin/brew shellenv)" 2>/dev/null
}

ensure_build_tools() {
    if ! xcode-select -p >/dev/null 2>&1; then
        echo "Xcode Command Line Tools are required to build the Anima Metal runner."
        echo "macOS will open the installer. Run Launch.command again after it finishes."
        xcode-select --install || true
        exit 1
    fi

    if ! command -v git >/dev/null 2>&1; then
        die "git is required but was not found."
    fi

    if ! command -v cmake >/dev/null 2>&1; then
        install_homebrew_if_needed
        echo "Installing cmake..."
        brew install cmake
    fi
}

clone_or_update_sdcpp() {
    mkdir -p "$ANIMA_ROOT"

    if [ -e "$SDCPP_SRC" ] && [ ! -d "$SDCPP_SRC/.git" ]; then
        die "$SDCPP_SRC exists but is not a git checkout. Move it aside or set ULTRA_FAST_ANIMA_ROOT."
    fi

    if [ ! -d "$SDCPP_SRC/.git" ]; then
        echo "Cloning stable-diffusion.cpp..."
        git clone --recursive "$SDCPP_REPO" "$SDCPP_SRC"
    fi

    if [ -n "$(git -C "$SDCPP_SRC" status --porcelain)" ]; then
        echo "stable-diffusion.cpp checkout has local changes; leaving it untouched."
        return
    else
        echo "Checking out stable-diffusion.cpp $SDCPP_COMMIT..."
        git -C "$SDCPP_SRC" fetch --depth 1 origin "$SDCPP_COMMIT" || git -C "$SDCPP_SRC" fetch origin
        git -C "$SDCPP_SRC" checkout "$SDCPP_COMMIT"
    fi

    git -C "$SDCPP_SRC" submodule update --init --recursive
}

apply_anima_metal_patch() {
    [ -f "$PATCH_FILE" ] || die "Missing patch file: $PATCH_FILE"

    local ggml_dir="$SDCPP_SRC/ggml"
    git -C "$ggml_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "ggml submodule is missing: $ggml_dir"

    if git -C "$ggml_dir" apply --unidiff-zero --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
        echo "Anima Metal patch already applied."
        return
    fi

    if git -C "$ggml_dir" apply --unidiff-zero --check "$PATCH_FILE" >/dev/null 2>&1; then
        echo "Applying Anima Metal patch..."
        git -C "$ggml_dir" apply --unidiff-zero "$PATCH_FILE"
        return
    fi

    die "Could not apply Anima Metal patch. The sd.cpp/ggml checkout may have drifted."
}

build_sdcli() {
    if [ -x "$SDCLI" ] && [ "${ULTRA_FAST_ANIMA_REBUILD:-0}" != "1" ]; then
        echo "sd-cli already built: $SDCLI"
        return
    fi

    echo "Configuring sd.cpp Metal build..."
    cmake -S "$SDCPP_SRC" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSD_METAL=ON \
        -DSD_BUILD_SHARED_LIBS=OFF \
        -DSD_BUILD_SHARED_GGML_LIB=OFF

    local jobs
    jobs="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

    echo "Building sd-cli..."
    cmake --build "$BUILD_DIR" --target sd-cli -j "$jobs"
}

write_runner() {
    mkdir -p "$(dirname "$RUNNER")" "$(dirname "$MODEL")" "$ANIMA_ROOT/sdcpp/output"

    cat > "$RUNNER" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ULTRA_FAST_ANIMA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SDCPP_SRC="${ULTRA_FAST_ANIMA_SDCPP_SRC:-$ROOT/sdcpp-src}"
BIN="${ULTRA_FAST_ANIMA_SDCLI:-$SDCPP_SRC/build-metal-im2col3d/bin/sd-cli}"
MODEL="${MODEL:-${ULTRA_FAST_ANIMA_MODEL:-$ROOT/sdcpp/models/Anima-P3-Turbo-AIO-Q4_K.gguf}}"
OUT_DIR="${ULTRA_FAST_ANIMA_OUTPUT_DIR:-$ROOT/sdcpp/output}"
PYTHON_BIN="${PYTHON_CMD:-python3}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  ./run_anima_aio_metal.sh "your prompt"

Useful env overrides:
  STEPS=8 WIDTH=512 HEIGHT=768 SEED=123 OPEN_OUTPUT=0 ./run_anima_aio_metal.sh "your prompt"

Defaults:
  STEPS=8 WIDTH=512 HEIGHT=768 CFG_SCALE=1 SAMPLING_METHOD=er_sde SCHEDULER=smoothstep CACHE_MODE=spectrum
USAGE
  exit 0
fi

if [[ ! -x "$BIN" ]]; then
  echo "Anima sd-cli binary not found: $BIN" >&2
  echo "Run scripts/setup_anima_metal_runner.sh from ultra-fast-image-gen." >&2
  exit 1
fi

PROMPT="${1:-anime portrait, cinematic lighting, detailed eyes, clean line art}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-768}"
STEPS="${STEPS:-8}"
CFG_SCALE="${CFG_SCALE:-1}"
SEED="${SEED:--1}"
SAMPLING_METHOD="${SAMPLING_METHOD:-er_sde}"
SCHEDULER="${SCHEDULER:-smoothstep}"
CACHE_MODE="${CACHE_MODE:-spectrum}"
OPEN_OUTPUT="${OPEN_OUTPUT:-1}"

mkdir -p "$OUT_DIR" "$(dirname "$MODEL")"

if [[ ! -f "$MODEL" ]]; then
  echo "Anima model missing, downloading to: $MODEL"
  MODEL_PATH="$MODEL" "$PYTHON_BIN" - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

target = Path(os.environ["MODEL_PATH"]).expanduser()
target.parent.mkdir(parents=True, exist_ok=True)
downloaded = Path(
    hf_hub_download(
        repo_id="n-Arno/Anima-P3-Turbo-AIO-Q4_K",
        filename="Anima-P3-Turbo-AIO-Q4_K.gguf",
        local_dir=str(target.parent),
    )
)
if downloaded != target and downloaded.exists() and not target.exists():
    shutil.copy2(downloaded, target)
if not target.exists():
    raise SystemExit(f"download finished but model is missing: {target}")
PY
fi

stamp="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$OUT_DIR/anima_aio_metal_${WIDTH}x${HEIGHT}_${STEPS}s_${stamp}.png}"
LOG="${LOG:-${OUT%.png}.log}"

cmd=(
  "$BIN"
  -m "$MODEL"
  --fa
  --steps "$STEPS"
  --cfg-scale "$CFG_SCALE"
  -W "$WIDTH"
  -H "$HEIGHT"
  --sampling-method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --cache-mode "$CACHE_MODE"
  --seed "$SEED"
  -p "$PROMPT"
  -o "$OUT"
)

if [[ -n "$NEGATIVE_PROMPT" ]]; then
  cmd+=(--negative-prompt "$NEGATIVE_PROMPT")
fi

echo "Writing: $OUT"
echo "Log:     $LOG"
/usr/bin/time -p "${cmd[@]}" 2>&1 | tee "$LOG"

if [[ "$OPEN_OUTPUT" == "1" ]]; then
  open "$OUT"
fi
RUNNER

    chmod +x "$RUNNER"
}

echo "============================================"
echo "   Setting up Anima Turbo AIO Metal runner"
echo "============================================"
echo "Anima root: $ANIMA_ROOT"

ensure_build_tools
clone_or_update_sdcpp
apply_anima_metal_patch
build_sdcli
write_runner

echo ""
echo "Anima runner ready: $RUNNER"
echo "Anima model path:  $MODEL"
echo "The GGUF will download automatically on first Anima generation."
