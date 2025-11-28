#!/bin/bash

set -e

echo "Initializing submodules..."
git submodule update --init --recursive

echo "Downloading model..."
mkdir -p Depth-Anything-V2/checkpoints
cd Depth-Anything-V2/checkpoints/

if [ ! -f "depth_anything_v2_vitb.pth" ]; then
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
    echo "Done."
else
    echo "Model already exists."
fi


