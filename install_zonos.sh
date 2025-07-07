#!/bin/bash

# Zonos Installation Script for AgentVox

set -e

echo "=== Zonos Installation Script ==="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"

# Check system dependencies
echo "Checking system dependencies..."

# Check if espeak-ng is installed
if ! command -v espeak-ng &> /dev/null; then
    echo "Installing espeak-ng..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install espeak-ng
        else
            echo "Error: Homebrew is required to install espeak-ng on macOS"
            echo "Please install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    else
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y espeak-ng
        elif command -v yum &> /dev/null; then
            sudo yum install -y espeak-ng
        else
            echo "Error: Unable to install espeak-ng. Please install it manually."
            exit 1
        fi
    fi
else
    echo "✓ espeak-ng is already installed"
fi

# Skip virtual environment creation - use current environment
echo "Using current Python environment..."

# Clone Zonos repository
echo "Setting up Zonos..."
if [ ! -d "third_party" ]; then
    mkdir -p third_party
fi

cd third_party

if [ ! -d "Zonos" ]; then
    echo "Cloning Zonos repository..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use the MPS-enabled fork
        echo "Detected macOS, using MPS-enabled fork..."
        git clone -b 12v/add-mps-support https://github.com/12v/Zonos.git
    else
        # Other systems - use original repository
        git clone https://github.com/Zyphra/Zonos.git
    fi
    cd Zonos
else
    echo "Zonos repository already exists, updating..."
    cd Zonos
    git pull
fi

# Install Zonos from local directory
echo "Installing Zonos..."
pip install -e .

# Optional: Install compile dependencies for hybrid model
echo "Installing optional compile dependencies for hybrid model..."
pip install --no-build-isolation -e .[compile] || echo "Warning: Failed to install compile dependencies. Hybrid model may not work."

cd ../..

# Note: All Python dependencies should be installed via main requirements.txt

# Test Zonos installation
echo
echo "Testing Zonos installation..."
python3 -c "
try:
    import torch
    from zonos.model import Zonos
    print('✓ Zonos import successful')
    print('  Torch version:', torch.__version__)
    print('  CUDA available:', torch.cuda.is_available())
except Exception as e:
    print('✗ Zonos import failed:', str(e))
    exit(1)
"

echo
echo "=== Zonos Installation Complete ==="
echo
echo "To use Zonos with AgentVox, run:"
echo "  agentvox --tts zonos --speaker-audio /path/to/speaker.mp3"
echo
echo "To use Zonos with auto-generated speaker (default), run:"
echo "  agentvox --tts zonos"
echo
echo "To use EdgeTTS (default), run:"
echo "  agentvox --tts edge"
echo