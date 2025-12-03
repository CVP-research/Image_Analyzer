# segment.py
import numpy as np
import cv2
from transformers import pipeline
import colorsys
from ultralytics import SAM

# Global SAM model cache
SAM_MODEL = None

# Segment model (GPU)
seg_pipe = pipeline(
    "image-segmentation",
    model="shi-labs/oneformer_coco_swin_large",
    device=0
)


def get_sam_model():
    """
    SAM 모델을 로드하고 반환 (전역 캐싱)
    
    Returns:
        SAM 모델 인스턴스
    """
    global SAM_MODEL
    if SAM_MODEL is None:
        print("Loading SAM model...")
        SAM_MODEL = SAM("sam2_l.pt")
        print("SAM model loaded.")
    return SAM_MODEL

# Distinct colors
def generate_distinct_colors(n=100):
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360
        sat = 0.8 + (i % 3) * 0.1
        val = 0.7 + (i % 4) * 0.1
        r, g, b = colorsys.hsv_to_rgb(hue/360, sat, val)
        colors.append([int(r*255), int(g*255), int(b*255)])
    return colors

distinct_colors = generate_distinct_colors()
color_names = [f"Color_{i+1}" for i in range(100)]


def run_segmentation(image_pil):
    """
    입력: PIL 이미지
    출력:
        annotations_for_gradio AnnotatedImage,
        json_labels
    """
    results = seg_pipe(image_pil)

    annotations = []
    labeled_results = []

    for idx, r in enumerate(results):
        mask = np.array(r["mask"])
        if mask.shape[:2] != image_pil.size[::-1]:
            mask = cv2.resize(mask.astype(np.uint8), image_pil.size, interpolation=cv2.INTER_NEAREST)

        mask_bool = mask > 128
        color = distinct_colors[idx % len(distinct_colors)]

        unique_label = f"{r['label']} #{idx+1}"
        annotations.append((mask_bool, unique_label))

        labeled_results.append({
            "id": idx + 1,
            "label": r["label"],
            "color": color_names[idx % len(color_names)],
            "rgb": color,
            "score": r.get("score", 1.0)
        })

    return annotations, labeled_results
