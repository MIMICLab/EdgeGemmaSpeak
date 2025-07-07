#!/bin/bash

# CosyVoice Installation Script for AgentVox

set -e

echo "=== CosyVoice Installation Script ==="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.10 first."
    exit 1
fi

# Check Python version (CosyVoice requires 3.10)
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [ "$PYTHON_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "Warning: CosyVoice works best with Python $REQUIRED_VERSION. Current version: $PYTHON_VERSION"
    echo "Consider using conda to create a Python 3.10 environment."
fi

echo "✓ Python $PYTHON_VERSION detected"

# Check system dependencies
echo "Checking system dependencies..."

# Check if sox is installed
if ! command -v sox &> /dev/null; then
    echo "Installing sox..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install sox
        else
            echo "Error: Homebrew is required to install sox on macOS"
            exit 1
        fi
    else
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y sox libsox-dev
        elif command -v yum &> /dev/null; then
            sudo yum install -y sox sox-devel
        else
            echo "Error: Unable to install sox. Please install it manually."
            exit 1
        fi
    fi
else
    echo "✓ sox is already installed"
fi

# Skip virtual environment creation - use current environment
echo "Using current Python environment..."


# Clone CosyVoice repository
echo "Setting up CosyVoice..."
if [ ! -d "third_party" ]; then
    mkdir -p third_party
fi

cd third_party

if [ ! -d "CosyVoice" ]; then
    echo "Cloning CosyVoice repository..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use the macOS-specific fork
        echo "Detected macOS, using macOS-optimized fork..."
        git clone --recursive https://github.com/jxwr/CosyVoice.git
    else
        # Other systems - use original repository
        git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    fi
    cd CosyVoice
else
    echo "CosyVoice repository already exists, updating..."
    cd CosyVoice
    git pull
    git submodule update --init --recursive
fi

# Install CosyVoice-specific dependencies
echo "Installing CosyVoice-specific dependencies..."

# Install pynini and WeTextProcessing for text normalization
if command -v conda &> /dev/null; then
    echo "Conda detected. Installing pynini via conda-forge..."
    conda install -c conda-forge pynini==2.1.6 -y || echo "Warning: Failed to install pynini via conda"
    pip install WeTextProcessing || echo "Warning: Failed to install WeTextProcessing"
else
    echo "Warning: Conda not found. Trying to install pynini via pip (may fail on macOS)..."
    pip install pynini==2.1.6 || echo "Warning: Failed to install pynini. Text normalization may not work."
    pip install WeTextProcessing || echo "Warning: Failed to install WeTextProcessing"
fi

# Create directory for pretrained models
if [ ! -d "pretrained_models" ]; then
    mkdir -p pretrained_models
fi

cd pretrained_models

# Check if git lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install git-lfs
    else
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        sudo apt-get install git-lfs
    fi
    git lfs install
fi

# Download CosyVoice-ttsfrd model
if [ ! -d "CosyVoice-ttsfrd" ]; then
    echo "Downloading CosyVoice-ttsfrd model..."
    git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git CosyVoice-ttsfrd
    cd CosyVoice-ttsfrd/
    unzip resource.zip -d .
    # Install ttsfrd wheel files
    echo "Installing ttsfrd dependencies..."
    pip install ttsfrd_dependency-0.1-py3-none-any.whl || echo "Warning: ttsfrd_dependency installation failed"
    
    # Try to install platform-specific ttsfrd wheel
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl || echo "Warning: ttsfrd wheel installation failed"
    else
        echo "Note: ttsfrd pre-built wheel is only available for Linux. Skipping on macOS/Windows."
    fi
    cd ..
else
    echo "✓ CosyVoice-ttsfrd model already exists"
fi

# Download CosyVoice2-0.5B model (recommended)
if [ ! -d "CosyVoice2-0.5B" ]; then
    echo "Downloading CosyVoice2-0.5B model..."
    echo "This may take a while depending on your internet connection..."
    
    # Try git clone first
    if git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git; then
        echo "✓ Model downloaded via git"
    else
        echo "Git clone failed. Trying modelscope SDK..."
        # Fallback to modelscope SDK
        # Note: modelscope should be installed via main requirements.txt
        python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='CosyVoice2-0.5B')"
    fi
else
    echo "✓ CosyVoice2-0.5B model already exists"
fi

# Download example assets if not present
cd ..
if [ ! -d "asset" ]; then
    echo "Downloading example assets..."
    mkdir -p asset
    cd asset
    # Download example audio files from the repository
    wget -q https://github.com/FunAudioLLM/CosyVoice/raw/main/asset/zero_shot_prompt.wav || echo "Failed to download zero_shot_prompt.wav"
    wget -q https://github.com/FunAudioLLM/CosyVoice/raw/main/asset/cross_lingual_prompt.wav || echo "Failed to download cross_lingual_prompt.wav"
    cd ..
fi

cd ../../..

# Note: All Python dependencies should be installed via main requirements.txt

# Test CosyVoice installation
echo
echo "Testing CosyVoice installation..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 -c "
import sys
import os
script_dir = '$SCRIPT_DIR'
sys.path.insert(0, os.path.join(script_dir, 'third_party/CosyVoice'))
sys.path.insert(0, os.path.join(script_dir, 'third_party/CosyVoice/third_party/Matcha-TTS'))
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    print('✓ CosyVoice import successful')
except Exception as e:
    print('✗ CosyVoice import failed:', str(e))
    import traceback
    traceback.print_exc()
    exit(1)
"

echo
echo "=== CosyVoice Installation Complete ==="
echo
echo "To use CosyVoice with AgentVox, run:"
echo "  agentvox --tts cosyvoice"
echo
echo "To use CosyVoice with custom prompt audio, run:"
echo "  agentvox --tts cosyvoice --cosyvoice-prompt /path/to/prompt.wav"
echo
echo "Note: CosyVoice requires a prompt audio file for voice cloning."
echo "Default prompt is located at: third_party/CosyVoice/asset/zero_shot_prompt.wav"
echo