#!/bin/bash

# Install dependencies for AgentVox

set -e

echo "=== Installing AgentVox Dependencies ==="
echo

# Install build dependencies first
echo "Installing build dependencies..."
pip install Cython numpy setuptools>=60.0.0 wheel

# Install main requirements
echo "Installing main requirements..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

echo
echo "=== Dependencies Installation Complete ==="