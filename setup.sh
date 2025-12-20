#!/bin/bash

set -e

echo "=== Image Analyzer Setup ==="
echo ""

echo "[1/2] Initializing git submodules..."
git submodule update --init --recursive

echo ""
echo "[2/2] Downloading Depth-Anything-V2 model..."
mkdir -p modules/Depth-Anything-V2/checkpoints
cd modules/Depth-Anything-V2/checkpoints/

if [ ! -f "depth_anything_v2_vitb.pth" ]; then
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
    echo "Model downloaded."
else
    echo "Model already exists."
fi

cd ../../..
echo ""
echo "=== Setup complete ==="
echo ""
echo "Project structure:"
echo "  modules/    - Core modules + Depth-Anything-V2 submodule"
echo "  pipeline/   - Dataset generation pipelines"
echo "  finetune/   - YOLO training & inference"
echo "  metric/     - Evaluation & visualization"
echo ""
echo "Run: python pipeline/run.py"


