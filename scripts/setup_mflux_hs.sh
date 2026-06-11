#!/usr/bin/env bash
set -euo pipefail

# Installs the patched MFLUX/MLX runtime used by the FLUX.2-klein Uncensored
# MFLUX HS 2K lane. Clones mflux at a tested revision into a stable location
# (NOT /tmp, which macOS wipes), applies the bundled hidden-state-compression +
# uncensored-GGUF-text-encoder patch, and pre-builds the uv environment.
#
# The app finds the checkout via ULTRA_FAST_MFLUX_HS_DIR or the default below.

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MFLUX_DIR="${ULTRA_FAST_MFLUX_HS_DIR:-$HOME/.cache/ultra-fast-image-gen/mflux}"
MFLUX_REPO="${ULTRA_FAST_MFLUX_HS_REPO:-https://github.com/filipstrand/mflux.git}"
# mflux 0.17.5 — the newest revision the bundled patch applies to cleanly.
MFLUX_COMMIT="${ULTRA_FAST_MFLUX_HS_COMMIT:-da36fe5e93c761fa7735def46844d05baaa5da2b}"
PATCH_FILE="$APP_DIR/patches/mflux_hs_uncensored_gguf.patch"

die() {
    echo "Error: $*" >&2
    exit 1
}

[ -f "$PATCH_FILE" ] || die "Patch file not found: $PATCH_FILE"

if ! command -v git >/dev/null 2>&1; then
    die "git is required but was not found."
fi

if ! command -v uv >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
        echo "uv is required to run the MFLUX HS lane. Installing with Homebrew..."
        brew install uv
    else
        die "uv is required (https://docs.astral.sh/uv/). Install it (e.g. 'brew install uv') and re-run."
    fi
fi

if [ -e "$MFLUX_DIR" ] && [ ! -d "$MFLUX_DIR/.git" ]; then
    die "$MFLUX_DIR exists but is not a git checkout. Move it aside or set ULTRA_FAST_MFLUX_HS_DIR."
fi

if [ -d "$MFLUX_DIR/.git" ]; then
    if git -C "$MFLUX_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
        echo "Patched MFLUX checkout already present at $MFLUX_DIR."
    elif [ -n "$(git -C "$MFLUX_DIR" status --porcelain)" ]; then
        die "$MFLUX_DIR has local changes that are not the bundled patch. Move it aside or set ULTRA_FAST_MFLUX_HS_DIR."
    else
        echo "Updating MFLUX checkout to $MFLUX_COMMIT..."
        git -C "$MFLUX_DIR" fetch --quiet origin
        git -C "$MFLUX_DIR" checkout --quiet "$MFLUX_COMMIT"
        echo "Applying HS/uncensored-GGUF patch..."
        git -C "$MFLUX_DIR" apply "$PATCH_FILE"
    fi
else
    echo "Cloning mflux into $MFLUX_DIR..."
    mkdir -p "$(dirname "$MFLUX_DIR")"
    git clone --quiet "$MFLUX_REPO" "$MFLUX_DIR"
    git -C "$MFLUX_DIR" checkout --quiet "$MFLUX_COMMIT"
    echo "Applying HS/uncensored-GGUF patch..."
    git -C "$MFLUX_DIR" apply "$PATCH_FILE"
fi

echo "Pre-building the MFLUX uv environment (first run only takes a minute)..."
uv sync --project "$MFLUX_DIR" --quiet

echo ""
echo "MFLUX HS runtime ready at: $MFLUX_DIR"
echo "The 'FLUX.2-klein-4B Uncensored MFLUX HS (2K Fast)' model can now generate."
