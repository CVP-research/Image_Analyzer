# Image Analyzer

A web-based analysis tool that performs image segmentation and depth estimation.

## Installation

```bash
git clone --recursive https://github.com/CVP-research/Image_Analyzer.git
cd Image_Analyzer
pip install -r requirements.txt
./setup.sh
```

## Usage

```bash
# Place your images in the input/ folder
python server.py
# Access http://localhost:8080
```

## Tech Stack

- **Segmentation**: OneFormer (COCO Swin Large)
- **Depth**: Depth Anything V2 (ViT-Base)
- **UI**: Gradio

## Model Change

Change `ENCODER` in `depth.py`:
- `vits` (100MB) / `vitb` (200MB) / `vitl` (600MB) / `vitg` (1.3GB)

Download other models:
```bash
cd Depth-Anything-V2/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

## Troubleshooting

If submodules are missing:
```bash
git submodule update --init --recursive
```

If models are missing:
```bash
./setup.sh
```