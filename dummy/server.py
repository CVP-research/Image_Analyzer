# server.py
import os
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

IMAGE_LIST = None
CACHE = {}
cache_queue = queue.Queue()


def calculate_mean_depths(depth_raw, annotations):
    """각 세그먼트 마스크에 해당하는 평균 깊이를 계산합니다."""
    mean_depths = {}
    
    for mask, label in annotations:
        try:
            segment_depths = depth_raw[mask]
            if segment_depths.size > 0:
                mean_depths[label] = np.mean(segment_depths)
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
    segmentation + depth 처리 (메모리에서만 처리)
    """
    pil_img = Image.open(image_path).convert("RGB")
    
    annotations, json_data = run_segmentation(pil_img)
    depth_png, depth_raw = compute_depth(pil_img)
    mean_depths = calculate_mean_depths(depth_raw, annotations)

    return (
        (pil_img, annotations),
        json_data,
        depth_png,
        mean_depths
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

    seg, json_data, depth_png, mean_depths = CACHE[idx]
    info = f"Image {idx+1}/{len(img_list)}\nFile: {img_list[idx].name}"
    mean_depths_str = format_mean_depths(mean_depths, json_data)

    return (
        info,
        seg,
        depth_png,
        json_data,
        mean_depths_str,
        idx
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