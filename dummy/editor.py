# composer.py
import os
from pathlib import Path
from PIL import Image
import numpy as np
import gradio as gr
from depth import compute_depth
from segment import run_segmentation
from composite import composite_with_depth

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OVERLAY_DIR = BASE_DIR / "overlays"
OVERLAY_DIR.mkdir(exist_ok=True)

# 캐시: 배경 이미지별로 depth 및 segment 저장
BG_CACHE = {}


def get_background_list():
    """배경 이미지 목록 가져오기"""
    return sorted([
        f.name for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])


def get_overlay_list():
    """누끼 이미지 목록 가져오기"""
    return sorted([
        f.name for f in OVERLAY_DIR.iterdir()
        if f.suffix.lower() in ['.png']
    ])


def load_background(bg_name):
    """배경 이미지 로드 및 depth, segmentation 계산"""
    if not bg_name:
        return None, None, None, None
    
    bg_path = INPUT_DIR / bg_name
    
    if bg_name in BG_CACHE:
        return BG_CACHE[bg_name]
    
    bg_img = Image.open(bg_path).convert("RGB")
    
    # Depth 계산
    depth_vis, depth_raw = compute_depth(bg_img)
    
    # Segmentation 계산
    annotations, json_data = run_segmentation(bg_img)
    
    # Segment별 평균 depth 계산 및 depth map 생성
    h, w = depth_raw.shape
    segment_depth_map = np.zeros((h, w), dtype=np.float32)
    
    # 각 segment의 평균 depth를 계산하고 해당 영역을 평균값으로 채움
    for mask, label in annotations:
        if mask.sum() > 0:
            # 해당 segment 영역의 평균 depth
            segment_depths = depth_raw[mask]
            mean_depth = np.mean(segment_depths)
            
            # 해당 segment 전체를 평균 depth로 설정
            segment_depth_map[mask] = mean_depth
    
    # 0-1로 정규화 (값이 클수록 가까움 유지)
    if segment_depth_map.max() > segment_depth_map.min():
        depth_norm = (segment_depth_map - segment_depth_map.min()) / (segment_depth_map.max() - segment_depth_map.min())
    else:
        depth_norm = np.zeros_like(segment_depth_map)
    
    BG_CACHE[bg_name] = (bg_img, depth_vis, depth_norm, annotations)
    return bg_img, depth_vis, depth_norm, annotations


def composite_preview(bg_name, overlay_name, depth_value, x_pos, y_pos, overlay_scale):
    """미리보기 합성"""
    if not bg_name or not overlay_name:
        return None
    
    bg_img, _, depth_norm, annotations = load_background(bg_name)
    if bg_img is None:
        return None
    
    overlay_path = OVERLAY_DIR / overlay_name
    if not overlay_path.exists():
        return bg_img
    
    overlay_img = Image.open(overlay_path).convert("RGBA")
    
    # 스케일 조정
    if overlay_scale != 100:
        w, h = overlay_img.size
        new_w = int(w * overlay_scale / 100)
        new_h = int(h * overlay_scale / 100)
        overlay_img = overlay_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    result = composite_with_depth(
        bg_img,
        depth_norm,  # segment별 평균 depth map
        overlay_img,
        depth_value,
        (int(x_pos), int(y_pos))
    )
    
    return result


def render_final(bg_name, overlay_name, depth_value, x_pos, y_pos, overlay_scale):
    """최종 이미지 렌더링"""
    result = composite_preview(bg_name, overlay_name, depth_value, x_pos, y_pos, overlay_scale)
    
    if result is None:
        return None, "No image to render"
    
    # 결과 저장
    output_path = BASE_DIR / "output" / f"composed_{bg_name}"
    output_path.parent.mkdir(exist_ok=True)
    result.save(output_path)
    
    return result, f"Saved to: {output_path}"


def upload_overlay(files):
    """누끼 이미지 업로드"""
    if files is None:
        return "No files uploaded", gr.Dropdown(choices=get_overlay_list())
    
    uploaded = []
    for file in files:
        filename = Path(file.name).name
        save_path = OVERLAY_DIR / filename
        Image.open(file.name).save(save_path)
        uploaded.append(filename)
    
    new_list = get_overlay_list()
    return f"Uploaded: {', '.join(uploaded)}", gr.Dropdown(choices=new_list, value=new_list[0] if new_list else None)


# Gradio UI
with gr.Blocks(title="Depth-Based Compositor", css="""
    .draggable { cursor: move; }
    .controls { background: #f5f5f5; padding: 15px; border-radius: 8px; }
""") as demo:
    
    gr.Markdown("# 🎨 Depth-Based Image Compositor")
    gr.Markdown("배경 이미지에 누끼 객체를 깊이 기반으로 합성합니다. 드래그로 위치 조정, depth로 앞뒤 조절")
    
    with gr.Row():
        # 왼쪽: 설정
        with gr.Column(scale=1, elem_classes="controls"):
            gr.Markdown("### 📂 Files")
            
            # 배경 이미지 선택
            bg_list = get_background_list()
            bg_dropdown = gr.Dropdown(
                choices=bg_list,
                label="Background Image",
                value=bg_list[0] if bg_list else None,
                allow_custom_value=False
            )
            
            # 누끼 업로드
            overlay_upload = gr.File(
                label="Upload Overlay (PNG with transparency)",
                file_count="multiple",
                file_types=[".png"]
            )
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            # 누끼 선택
            overlay_list = get_overlay_list()
            overlay_dropdown = gr.Dropdown(
                choices=overlay_list,
                label="Overlay Image",
                value=overlay_list[0] if overlay_list else None,
                allow_custom_value=False
            )
            
            gr.Markdown("### 🎚️ Controls")
            
            # Depth 슬라이더
            depth_slider = gr.Slider(
                minimum=-0.5,
                maximum=1.5,
                value=0.5,
                step=0.01,
                label="Depth (1=앞, 0=뒤)",
                info="값이 클수록 카메라에 가까움 (앞). -0.5=배경 전체보다 뒤, 1.5=배경 전체보다 앞"
            )
            
            # 위치 조정 (누끼 이미지 중앙 기준)
            x_slider = gr.Slider(
                minimum=-500,
                maximum=2000,
                value=500,
                step=1,
                label="X Position (중앙)"
            )
            
            y_slider = gr.Slider(
                minimum=-500,
                maximum=2000,
                value=500,
                step=1,
                label="Y Position (중앙)"
            )
            
            # 크기 조정
            scale_slider = gr.Slider(
                minimum=10,
                maximum=300,
                value=100,
                step=5,
                label="Scale (%)"
            )
            
            # 버튼들
            with gr.Row():
                preview_btn = gr.Button("🔄 Preview", variant="secondary")
                render_btn = gr.Button("💾 Render & Save", variant="primary")
        
        # 오른쪽: 미리보기
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Preview")
            
            with gr.Tabs():
                with gr.Tab("Composite"):
                    composite_output = gr.Image(
                        label="Composite Result",
                        type="pil",
                        height=600,
                        elem_classes="draggable"
                    )
                
                with gr.Tab("Background"):
                    bg_preview = gr.Image(
                        label="Background Image",
                        type="pil",
                        height=600
                    )
                
                with gr.Tab("Depth Map"):
                    depth_preview = gr.Image(
                        label="Background Depth",
                        type="numpy",
                        height=600
                    )
            
            render_status = gr.Textbox(label="Render Status", interactive=False)
    
    # 이벤트 핸들러
    
    # 배경 이미지 로드
    def on_bg_change(bg_name):
        bg_img, depth_vis, _, annotations = load_background(bg_name)
        if bg_img:
            w, h = bg_img.size
            return (
                bg_img, 
                depth_vis,
                gr.Slider(minimum=-w//2, maximum=int(w*1.5), value=w//2, step=1),
                gr.Slider(minimum=-h//2, maximum=int(h*1.5), value=h//2, step=1)
            )
        return bg_img, depth_vis, gr.Slider(), gr.Slider()
    
    bg_dropdown.change(
        fn=on_bg_change,
        inputs=[bg_dropdown],
        outputs=[bg_preview, depth_preview, x_slider, y_slider]
    )
    
    # 누끼 업로드
    overlay_upload.upload(
        fn=upload_overlay,
        inputs=[overlay_upload],
        outputs=[upload_status, overlay_dropdown]
    )
    
    # 미리보기 업데이트
    def auto_preview(bg_name, overlay_name, depth_value, x_pos, y_pos, scale):
        return composite_preview(bg_name, overlay_name, depth_value, x_pos, y_pos, scale)
    
    preview_inputs = [
        bg_dropdown,
        overlay_dropdown,
        depth_slider,
        x_slider,
        y_slider,
        scale_slider
    ]
    
    # Preview 버튼
    preview_btn.click(
        fn=auto_preview,
        inputs=preview_inputs,
        outputs=composite_output
    )
    
    # 실시간 미리보기 (슬라이더 변경 시)
    for inp in preview_inputs:
        inp.change(
            fn=auto_preview,
            inputs=preview_inputs,
            outputs=composite_output
        )
    
    # Render 버튼
    render_btn.click(
        fn=render_final,
        inputs=preview_inputs,
        outputs=[composite_output, render_status]
    )
    
    # 초기 로드
    def initial_load():
        bg_list = get_background_list()
        overlay_list = get_overlay_list()
        
        bg_img, depth_vis = None, None
        x_update = gr.Slider()
        y_update = gr.Slider()
        
        if bg_list:
            bg_img, depth_vis, _, _ = load_background(bg_list[0])
            if bg_img:
                w, h = bg_img.size
                x_update = gr.Slider(minimum=-w//2, maximum=int(w*1.5), value=w//2, step=1)
                y_update = gr.Slider(minimum=-h//2, maximum=int(h*1.5), value=h//2, step=1)
        
        return (
            bg_img, 
            depth_vis,
            gr.Dropdown(choices=bg_list, value=bg_list[0] if bg_list else None),
            gr.Dropdown(choices=overlay_list, value=overlay_list[0] if overlay_list else None),
            x_update,
            y_update
        )
    
    demo.load(
        fn=initial_load,
        inputs=None,
        outputs=[bg_preview, depth_preview, bg_dropdown, overlay_dropdown, x_slider, y_slider]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8082, share=True)
