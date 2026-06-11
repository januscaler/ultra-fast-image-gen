#!/bin/bash

# Ultra Fast Image Gen - Mac Launcher
# Double-click this file to start the app!

cd "$(dirname "$0")"

echo "============================================"
echo "       Ultra Fast Image Gen for Mac"
echo "============================================"
echo ""

# Check Python version
PYTHON_CMD=""
for cmd in python3.11 python3.10 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd -c 'import sys; print(sys.version_info.minor)')
        if [ "$version" -ge 10 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Python 3.10+ is required but not found."
    echo ""

    # Check if Homebrew is installed
    if command -v brew &> /dev/null; then
        echo "Homebrew detected! Would you like to install Python 3.11? (y/n)"
        read -p "> " install_python
        if [ "$install_python" = "y" ] || [ "$install_python" = "Y" ]; then
            echo ""
            echo "Installing Python 3.11..."
            brew install python@3.11
            PYTHON_CMD="python3.11"
            echo ""
            echo "Python 3.11 installed successfully!"
        else
            echo "Please install Python 3.10+ manually and try again."
            read -p "Press Enter to exit..."
            exit 1
        fi
    else
        echo "Would you like to install Homebrew and Python? (y/n)"
        read -p "> " install_brew
        if [ "$install_brew" = "y" ] || [ "$install_brew" = "Y" ]; then
            echo ""
            echo "Installing Homebrew (you may need to enter your password)..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

            # Add Homebrew to PATH for this session
            eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || eval "$(/usr/local/bin/brew shellenv)" 2>/dev/null

            echo ""
            echo "Installing Python 3.11..."
            brew install python@3.11
            PYTHON_CMD="python3.11"
            echo ""
            echo "Installation complete!"
        else
            echo ""
            echo "To install manually:"
            echo "  1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "  2. Run: brew install python@3.11"
            read -p "Press Enter to exit..."
            exit 1
        fi
    fi
fi

echo "Using: $PYTHON_CMD"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "First time setup - creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed. Keep a requirements hash so double-click
# launches pick up newly added dependencies after updates.
REQ_HASH_FILE="venv/.requirements.sha256"
CURRENT_REQ_HASH=$(shasum -a 256 requirements.txt | awk '{print $1}')
INSTALLED_REQ_HASH=$(cat "$REQ_HASH_FILE" 2>/dev/null || true)

if [ ! -f "venv/.installed" ] || [ "$CURRENT_REQ_HASH" != "$INSTALLED_REQ_HASH" ]; then
    echo ""
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "$CURRENT_REQ_HASH" > "$REQ_HASH_FILE"
    touch venv/.installed
    echo ""
    echo "Installation complete!"
fi

# Anima uses the patched sd.cpp Metal runner outside this repo. The model GGUF
# is downloaded automatically on first Anima generation, but the runner itself
# must exist first.
echo ""
echo "Checking Anima Turbo AIO Metal setup..."
ANIMA_ROOT="${ULTRA_FAST_ANIMA_ROOT:-$HOME/anima-comfyui}"
ANIMA_RUNNER="${ULTRA_FAST_ANIMA_RUNNER:-$ANIMA_ROOT/run_anima_aio_metal.sh}"
ANIMA_MODEL="${ULTRA_FAST_ANIMA_MODEL:-$ANIMA_ROOT/sdcpp/models/Anima-P3-Turbo-AIO-Q4_K.gguf}"
ANIMA_SDCLI="${ULTRA_FAST_ANIMA_SDCLI:-$ANIMA_ROOT/sdcpp-src/build-metal-im2col3d/bin/sd-cli}"
export ULTRA_FAST_ANIMA_RUNNER="$ANIMA_RUNNER"
export ULTRA_FAST_ANIMA_MODEL="$ANIMA_MODEL"
export ULTRA_FAST_ANIMA_ROOT="$ANIMA_ROOT"
export PYTHON_CMD="$(pwd)/venv/bin/python"

if [ ! -x "$ANIMA_RUNNER" ] || [ ! -x "$ANIMA_SDCLI" ]; then
    echo "Anima runner is not installed yet. Installing the patched Metal runner now..."
    if ! bash "$(pwd)/scripts/setup_anima_metal_runner.sh"; then
        echo ""
        echo "Anima setup could not complete."
        echo "Fix the message above, then run Launch.command again."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

if [ ! -x "$ANIMA_RUNNER" ] || [ ! -x "$ANIMA_SDCLI" ]; then
    echo "Anima setup did not finish cleanly."
    read -p "Press Enter to exit..."
    exit 1
elif [ -f "$ANIMA_MODEL" ]; then
    echo "Anima runner ready: $ANIMA_RUNNER"
    echo "Anima model ready: $ANIMA_MODEL"
else
    echo "Anima runner ready: $ANIMA_RUNNER"
    echo "Anima model not downloaded yet."
    echo "It will auto-download on the first Anima generation."
fi

# The fastest 2K lane (FLUX.2-klein Uncensored MFLUX HS) runs through a patched
# MFLUX checkout outside this repo. Install it if missing, but don't block the
# app on failure: the Uncensored SDNQ HS 2K lane works without it.
echo ""
echo "Checking MFLUX HS 2K runtime..."
MFLUX_HS_DIR="${ULTRA_FAST_MFLUX_HS_DIR:-$HOME/.cache/ultra-fast-image-gen/mflux}"
if [ -d "$MFLUX_HS_DIR" ]; then
    echo "MFLUX HS runtime ready: $MFLUX_HS_DIR"
else
    echo "MFLUX HS runtime not installed yet. Installing (only needed for the MFLUX 2K lane)..."
    if bash "$(pwd)/scripts/setup_mflux_hs.sh"; then
        echo "MFLUX HS runtime ready: $MFLUX_HS_DIR"
    else
        echo "MFLUX HS setup did not finish. The 'Uncensored MFLUX HS' model will be unavailable;"
        echo "the 'Uncensored SDNQ HS' 2K lane still works. Re-run scripts/setup_mflux_hs.sh to retry."
    fi
fi
export ULTRA_FAST_MFLUX_HS_DIR="$MFLUX_HS_DIR"

echo ""
echo "Starting Gradio UI..."
echo "Opening browser to http://127.0.0.1:7860"
echo ""
echo "(Press Ctrl+C to stop the server)"
echo ""

# Open browser after server starts (6s delay for model loading)
(sleep 6 && open http://127.0.0.1:7860) &

# Run the app
python app.py
