# editor.py
import numpy as np
from PIL import Image


def composite_with_depth(background_img, background_depth, overlay_img, overlay_depth_value, overlay_position=(0, 0)):
    """
    배경 이미지와 누끼 객체를 depth 기반으로 합성

    Args:
        background_img: 배경 PIL 이미지 (RGB)
        background_depth: 배경의 depth map (numpy array, 값이 클수록 가까움)
        overlay_img: 누끼 객체 PIL 이미지 (RGBA, 투명도 포함)
        overlay_depth_value: 누끼 객체의 depth 값 (0-1, 클수록 가까움)
        overlay_position: 누끼 객체의 (x, y) 위치

    Returns:
        합성된 PIL 이미지 (RGB)
    """
    # 배경 이미지를 numpy로 변환 (한 번만)
    if not isinstance(background_img, np.ndarray):
        bg_array = np.array(background_img).copy()
    else:
        bg_array = background_img.copy()

    bg_h, bg_w = bg_array.shape[:2]

    # 누끼 객체를 numpy로 변환
    if overlay_img.mode != 'RGBA':
        overlay_img = overlay_img.convert('RGBA')
    overlay_array = np.array(overlay_img)
    overlay_h, overlay_w = overlay_array.shape[:2]

    # 누끼 객체의 RGB와 알파 채널 분리
    overlay_rgb = overlay_array[:, :, :3]
    overlay_alpha = overlay_array[:, :, 3] / 255.0  # 0-1 범위로 정규화

    # 배경 depth를 0-1로 정규화 (캐싱 가능하지만 여기서는 빠르므로 그대로)
    if background_depth.max() > background_depth.min():
        bg_depth_norm = (background_depth - background_depth.min()) / (background_depth.max() - background_depth.min())
    else:
        bg_depth_norm = np.zeros_like(background_depth)

    # 합성 위치 계산
    x_offset, y_offset = overlay_position

    # 합성 영역 계산 (이미지 범위를 벗어나지 않도록)
    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    x_end = min(bg_w, x_offset + overlay_w)
    y_end = min(bg_h, y_offset + overlay_h)

    # 누끼 객체에서 실제로 사용할 영역
    overlay_x_start = max(0, -x_offset)
    overlay_y_start = max(0, -y_offset)
    overlay_x_end = overlay_x_start + (x_end - x_start)
    overlay_y_end = overlay_y_start + (y_end - y_start)

    if x_end <= x_start or y_end <= y_start:
        # 겹치는 영역이 없으면 배경만 반환
        return background_img

    # 해당 영역의 배경 depth 가져오기
    region_bg_depth = bg_depth_norm[y_start:y_end, x_start:x_end]

    # 누끼 객체 영역 추출
    region_overlay_rgb = overlay_rgb[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
    region_overlay_alpha = overlay_alpha[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]

    # Depth 기반 occlusion mask 생성
    # overlay_depth_value보다 배경이 가까우면 (depth 값이 크면) 누끼를 지움
    # 배경 depth가 더 크다 = 배경이 더 가깝다 = 누끼를 가려야 함
    should_hide = region_bg_depth > overlay_depth_value

    # 알파 값을 occlusion으로 조정
    # should_hide가 True인 곳은 알파를 0으로 (배경이 보임)
    adjusted_alpha = region_overlay_alpha.copy()
    adjusted_alpha[should_hide] = 0

    # 알파 블렌딩: 누끼를 배경 위에 덮어씀
    adjusted_alpha_3d = adjusted_alpha[:, :, np.newaxis]  # (H, W, 1)로 확장

    bg_array[y_start:y_end, x_start:x_end] = (
        adjusted_alpha_3d * region_overlay_rgb +
        (1 - adjusted_alpha_3d) * bg_array[y_start:y_end, x_start:x_end]
    ).astype(np.uint8)

    return Image.fromarray(bg_array)


def resize_overlay_keep_aspect(overlay_img, max_width, max_height):
    """
    누끼 객체를 비율 유지하며 리사이즈

    Args:
        overlay_img: PIL 이미지
        max_width: 최대 너비
        max_height: 최대 높이

    Returns:
        리사이즈된 PIL 이미지
    """
    w, h = overlay_img.size
    aspect = w / h

    if w > max_width or h > max_height:
        if w / max_width > h / max_height:
            new_w = max_width
            new_h = int(max_width / aspect)
        else:
            new_h = max_height
            new_w = int(max_height * aspect)

        return overlay_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return overlay_img


if __name__ == "__main__":
    import gradio as gr
    from pathlib import Path
    import sys

    BASE_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(BASE_DIR))

    from depth import compute_depth

    def demo_composite(background_img, overlay_img, depth_value, x_pos, y_pos):
        """Gradio demo function for depth-based compositing"""
        if background_img is None or overlay_img is None:
            return None

        # Convert to PIL if needed
        if isinstance(background_img, np.ndarray):
            background_img = Image.fromarray(background_img)
        if isinstance(overlay_img, np.ndarray):
            overlay_img = Image.fromarray(overlay_img)

        # Compute depth for background
        _, bg_depth = compute_depth(background_img)

        # Composite
        result = composite_with_depth(
            background_img,
            bg_depth,
            overlay_img,
            depth_value,
            (int(x_pos), int(y_pos))
        )

        return result

    # Gradio interface
    with gr.Blocks(title="Depth-Based Image Compositor") as demo:
        gr.Markdown("# Depth-Based Image Compositor")
        gr.Markdown("Upload a background image and an overlay image (with transparency) to composite them based on depth.")

        with gr.Row():
            with gr.Column():
                bg_input = gr.Image(label="Background Image", type="pil")
                overlay_input = gr.Image(label="Overlay Image (RGBA)", type="pil")

            with gr.Column():
                depth_slider = gr.Slider(0, 1, value=0.5, label="Overlay Depth (0=far, 1=near)")
                x_slider = gr.Slider(0, 1000, value=0, step=1, label="X Position")
                y_slider = gr.Slider(0, 1000, value=0, step=1, label="Y Position")
                composite_btn = gr.Button("Composite")

        output_img = gr.Image(label="Result", type="pil")

        composite_btn.click(
            fn=demo_composite,
            inputs=[bg_input, overlay_input, depth_slider, x_slider, y_slider],
            outputs=output_img
        )

    demo.launch(server_name="0.0.0.0", server_port=8081, share=True)
