# server.py
import os
import pickle
from pathlib import Path
import queue
from threading import Thread
from PIL import Image
import numpy as np
import gradio as gr

from segment import run_segmentation
from depth import compute_depth

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
SEGMENT_DIR = BASE_DIR / "segment"
DEPTH_DIR = BASE_DIR / "depth"

SEGMENT_DIR.mkdir(exist_ok=True)
DEPTH_DIR.mkdir(exist_ok=True)

IMAGE_LIST = None
CACHE = {}
cache_queue = queue.Queue()


def build_tag_html(annotations, json_data):
    # 이 함수는 사용되지 않지만, 다른 곳에서 호출될까봐 일단 유지합니다.
    if not annotations or not json_data:
        return "<div style='color:#888;'>No segments detected</div>"
    return ""


def calculate_mean_depths(depth_raw, annotations):
    """각 세그먼트 마스크에 해당하는 평균 깊이를 계산합니다."""
    mean_depths = {}
    
    for item in annotations:
        try:
            mask = item[0]
            label = item[1]
            
            if isinstance(mask, np.ndarray) and mask.shape == depth_raw.shape:
                segment_depths = depth_raw[mask.astype(bool)]
                valid_depths = segment_depths[segment_depths > 0]
                
                if valid_depths.size > 0:
                    mean_depths[label] = np.mean(valid_depths)
                else:
                    mean_depths[label] = np.nan
            
        except Exception:
            continue
            
    return mean_depths

def format_mean_depths(mean_depths, json_data):
    """
    평균 깊이 데이터를 오름차순으로 정렬하여 텍스트박스용 일반 문자열로 포맷합니다.
    """
    if not mean_depths:
        return "No segment depth data available."
    
    sorted_depths = list(mean_depths.items())
    sorted_depths.sort(key=lambda item: item[1] if not np.isnan(item[1]) else float('inf'))
    
    lines = ["--- Average Segment Depths (Ascending) ---"]
    
    for label, depth in sorted_depths:
        if np.isnan(depth):
            lines.append(f"{label}: N/A (No valid depth data)")
        else:
            lines.append(f"{label}: {depth:.2f} meters")
    
    return "\n".join(lines)


def load_image_list():
    global IMAGE_LIST
    if IMAGE_LIST is None:
        IMAGE_LIST = sorted([
            Path(root) / f
            for root, _, files in os.walk(INPUT_DIR)
            for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
    return IMAGE_LIST


def process_full(image_path):
    """
    segmentation + depth 전처리 및 결과 저장
    """
    pil_img = Image.open(image_path).convert("RGB")
    base = image_path.stem

    seg_meta_path = SEGMENT_DIR / f"{base}_seg.pkl"
    depth_png_path = DEPTH_DIR / f"{base}_depth.png"
    depth_raw_path = DEPTH_DIR / f"{base}_depth.npy"

    if not seg_meta_path.exists():
        annotations, json_data = run_segmentation(pil_img)
        with seg_meta_path.open("wb") as f:
            pickle.dump({"annotations": annotations, "json": json_data}, f)
    else:
        with seg_meta_path.open("rb") as f:
            meta = pickle.load(f)
        annotations = meta["annotations"]
        json_data = meta["json"]

    if not depth_png_path.exists() or not depth_raw_path.exists():
        depth_png, depth_raw = compute_depth(pil_img)
        Image.fromarray(depth_png).save(depth_png_path)
        np.save(depth_raw_path, depth_raw)
    else:
        depth_png = np.array(Image.open(depth_png_path).convert("RGB"))
        depth_raw = np.load(depth_raw_path)

    # 원본 이미지를 base로 사용 (PIL Image 그대로)
    mean_depths = calculate_mean_depths(depth_raw, annotations)

    # show 함수의 반환 값 순서에 맞춰 튜플 반환 (총 4개)
    return (
        (pil_img, annotations),  # seg (AnnotatedImage input format: PIL Image + annotations)
        json_data,               # json_data
        depth_png,               # depth_png
        mean_depths              # Mean depths dictionary
    )

def preload_worker():
    while True:
        idx = cache_queue.get()
        if idx is None:
            break
        img_list = load_image_list()
        if idx not in CACHE and 0 <= idx < len(img_list):
            CACHE[idx] = process_full(img_list[idx])
        cache_queue.task_done()


worker_thread = Thread(target=preload_worker, daemon=True)
worker_thread.start()


def show(idx):
    img_list = load_image_list()

    if idx < 0 or idx >= len(img_list):
        return "Out of range", (None, []), None, None, "No image loaded", idx

    if idx not in CACHE:
        CACHE[idx] = process_full(img_list[idx])

    # preload around
    for off in [-2, -1, 1, 2]:
        t = idx + off
        if 0 <= t < len(img_list) and t not in CACHE:
            cache_queue.put(t)

    # process_full의 반환 값 4개
    seg, json_data, depth_png, mean_depths = CACHE[idx]
    info = f"Image {idx+1}/{len(img_list)}\nFile: {img_list[idx].name}"

    mean_depths_str = format_mean_depths(mean_depths, json_data)

    # Gradio의 outputs과 일치하는 6개 요소 반환
    return (
        info,                  # 1. info (info_box)
        seg,                   # 2. seg (seg_out) - (PIL Image, annotations) 튜플
        depth_png,             # 3. depth_png (depth_out)
        json_data,             # 4. json_data (json_out)
        mean_depths_str,       # 5. mean_depths_str (mean_depths_out)
        idx                    # 6. current (current)
    )

with gr.Blocks(title="Segment + Depth Viewer") as demo:
    current = gr.State(0)
    
    info_box = gr.Textbox(label="Info", interactive=False)

    with gr.Row():
        seg_out = gr.AnnotatedImage(
            label="Segmentation (Hover to see labels)",
            height=600
        )
        depth_out = gr.Image(label="Depth Map", height=600)

    with gr.Row():
        json_out = gr.JSON(label="Labels")
        mean_depths_out = gr.Textbox(label="Segment Depth Averages (Ascending)", lines=8)

    prev_btn = gr.Button("⬅️ Previous")
    next_btn = gr.Button("Next ➡️")

    # OUTPUT_COMPONENTS
    OUTPUT_COMPONENTS = [
        info_box,
        seg_out,
        depth_out,
        json_out,
        mean_depths_out,
        current
    ]


    demo.load(
        fn=show,
        inputs=[current],
        outputs=OUTPUT_COMPONENTS
    )

    prev_btn.click(
        fn=lambda idx: show(max(idx-1, 0)),
        inputs=[current],
        outputs=OUTPUT_COMPONENTS
    )

    next_btn.click(
        fn=lambda idx: show(min(idx+1, len(load_image_list())-1)),
        inputs=[current],
        outputs=OUTPUT_COMPONENTS
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)