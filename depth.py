# depth.py
import numpy as np
import torch
import matplotlib
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR / "Depth-Anything-V2"
if REPO_ROOT.exists() and str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2

# Depth 모델 설정
ENCODER = "vitb"
INPUT_SIZE = 518

depth_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

# Load model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 우선순위: 1) Depth-Anything-V2/checkpoints, 2) ./checkpoints
checkpoint_path = REPO_ROOT / f"checkpoints/depth_anything_v2_{ENCODER}.pth"
if not checkpoint_path.exists():
    alt_path = BASE_DIR / f"checkpoints/depth_anything_v2_{ENCODER}.pth"
    if alt_path.exists():
        checkpoint_path = alt_path
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} or {alt_path}\n"
            f"Please download the model using setup.sh or manually:"
            f"\n  wget -P Depth-Anything-V2/checkpoints/ "
            f"https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_{ENCODER}.pth"
        )

depth_model = DepthAnythingV2(**depth_model_configs[ENCODER])
depth_model.load_state_dict(
    torch.load(checkpoint_path, map_location="cpu")
)
depth_model = depth_model.to(DEVICE).eval()

# 컬러맵 LUT 미리 계산 (256 레벨)
depth_cmap = matplotlib.colormaps.get_cmap('Spectral_r')
CMAP_LUT = (depth_cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)


def compute_depth(image_pil):
    """
    DepthAnythingV2 depth 추론 → (depth_png_numpy, depth_raw_numpy)
    """
    raw = np.array(image_pil)[:, :, ::-1]  # PIL -> BGR
    depth_raw = depth_model.infer_image(raw, INPUT_SIZE)

    # 깊이 값 반전: 가까운 곳은 밝게, 먼 곳은 어둡게
    d_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min())
    d_norm = 1.0 - d_norm  # 반전
    depth_uint8 = (d_norm * 255).astype(np.uint8)

    depth_color = CMAP_LUT[depth_uint8]

    return depth_color, depth_raw
