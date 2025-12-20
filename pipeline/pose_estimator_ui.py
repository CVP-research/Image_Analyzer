"""
Pose Estimator - Gradio 통합 버전

run.py 파이프라인 내에서 직접 실행되는 카메라 포즈 등록 UI
- Gradio로 래핑된 3D 뷰어
- 완료 시 자동으로 파이프라인 재개
"""

import json
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Callable
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

import gradio as gr


class PoseEstimatorUI:
    """카메라 포즈 등록 UI"""

    def __init__(self, job_dir: Path):
        """
        Args:
            job_dir: Job 디렉토리 경로
        """
        self.job_dir = job_dir
        self.input_dir = job_dir / "input"
        self.pose_dir = self.input_dir / "pose"
        self.poses_file = self.pose_dir / "camera_poses.json"

        self.pose_dir.mkdir(parents=True, exist_ok=True)

        self.poses: Dict[str, dict] = {}
        self.images: List[str] = []
        self.current_index = 0
        self.completed = False

        self._load_images()
        self._load_existing_poses()

    def _load_images(self):
        """input 폴더의 이미지 목록 로드"""
        if not self.input_dir.exists():
            return

        self.images = sorted([
            f.name for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])

    def _load_existing_poses(self):
        """기존 포즈 파일 로드"""
        if self.poses_file.exists():
            with open(self.poses_file, 'r', encoding='utf-8') as f:
                self.poses = json.load(f)

    def save_poses(self):
        """포즈 저장"""
        with open(self.poses_file, 'w', encoding='utf-8') as f:
            json.dump(self.poses, f, indent=2, ensure_ascii=False)

    def get_current_image_path(self) -> Optional[str]:
        """현재 이미지 경로 반환"""
        if self.current_index < len(self.images):
            return str(self.input_dir / self.images[self.current_index])
        return None

    def capture_pose(self, azimuth: float, elevation: float, distance: float):
        """현재 포즈 저장"""
        if self.current_index >= len(self.images):
            return "모든 이미지 완료!", None, f"{len(self.images)}/{len(self.images)}"

        import math
        filename = self.images[self.current_index]

        # 구면 좌표를 직교 좌표로 변환
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)

        x = distance * math.cos(el_rad) * math.sin(az_rad)
        y = distance * math.sin(el_rad)
        z = distance * math.cos(el_rad) * math.cos(az_rad)

        self.poses[filename] = {
            "position": [x, y, z],
            "look_at": [0, 0, 0],
            "up": [0, 1, 0],
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance
        }

        self.current_index += 1
        self.save_poses()

        if self.current_index >= len(self.images):
            self.completed = True
            return (
                f"완료! {len(self.poses)}개 포즈 저장됨",
                None,
                f"{len(self.images)}/{len(self.images)}"
            )

        next_image = self.get_current_image_path()
        progress = f"{self.current_index + 1}/{len(self.images)}"
        return f"'{filename}' 저장됨", next_image, progress

    def skip_image(self):
        """이미지 건너뛰기"""
        if self.current_index >= len(self.images):
            return "모든 이미지 완료!", None, f"{len(self.images)}/{len(self.images)}"

        self.current_index += 1

        if self.current_index >= len(self.images):
            self.completed = True
            return (
                f"완료! {len(self.poses)}개 포즈 저장됨",
                None,
                f"{len(self.images)}/{len(self.images)}"
            )

        next_image = self.get_current_image_path()
        progress = f"{self.current_index + 1}/{len(self.images)}"
        return "건너뜀", next_image, progress

    def create_ui(self) -> gr.Blocks:
        """Gradio UI 생성"""

        with gr.Blocks(title="Camera Pose Estimator", theme=gr.themes.Dark()) as demo:
            gr.Markdown("# Camera Pose Registration")
            gr.Markdown("각 이미지가 촬영된 카메라 위치(각도)를 지정하세요.")

            with gr.Row():
                # 왼쪽: 이미지
                with gr.Column(scale=1):
                    image_display = gr.Image(
                        value=self.get_current_image_path(),
                        label="현재 이미지",
                        type="filepath",
                        height=400
                    )
                    progress_text = gr.Textbox(
                        value=f"1/{len(self.images)}" if self.images else "0/0",
                        label="진행 상황",
                        interactive=False
                    )

                # 오른쪽: 컨트롤
                with gr.Column(scale=1):
                    gr.Markdown("### 카메라 위치 설정")
                    gr.Markdown("""
                    - **Azimuth**: 수평 회전 각도 (0°=정면, 90°=오른쪽, -90°=왼쪽, 180°=뒤)
                    - **Elevation**: 수직 각도 (0°=수평, 90°=위, -90°=아래)
                    - **Distance**: 객체로부터의 거리
                    """)

                    azimuth = gr.Slider(
                        minimum=-180,
                        maximum=180,
                        value=0,
                        step=5,
                        label="Azimuth (수평 각도)",
                        info="0°=정면"
                    )
                    elevation = gr.Slider(
                        minimum=-90,
                        maximum=90,
                        value=0,
                        step=5,
                        label="Elevation (수직 각도)",
                        info="0°=수평"
                    )
                    distance = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=0.5,
                        label="Distance (거리)"
                    )

                    status_text = gr.Textbox(
                        value="",
                        label="상태",
                        interactive=False
                    )

                    with gr.Row():
                        capture_btn = gr.Button("포즈 저장", variant="primary", size="lg")
                        skip_btn = gr.Button("건너뛰기", size="lg")

                    gr.Markdown("---")
                    finish_btn = gr.Button(
                        "완료 & 파이프라인 계속",
                        variant="stop",
                        size="lg"
                    )

            # 이벤트 핸들러
            def on_capture(az, el, dist):
                status, img, prog = self.capture_pose(az, el, dist)
                return status, img, prog

            def on_skip():
                status, img, prog = self.skip_image()
                return status, img, prog

            def on_finish():
                self.completed = True
                self.save_poses()
                # Gradio를 종료하기 위해 JavaScript 사용
                return gr.update(value=f"저장 완료! {len(self.poses)}개 포즈. 창을 닫으세요.")

            capture_btn.click(
                fn=on_capture,
                inputs=[azimuth, elevation, distance],
                outputs=[status_text, image_display, progress_text]
            )

            skip_btn.click(
                fn=on_skip,
                inputs=[],
                outputs=[status_text, image_display, progress_text]
            )

            finish_btn.click(
                fn=on_finish,
                inputs=[],
                outputs=[status_text]
            )

        return demo


def run_pose_estimator(job_dir: Path, auto_open: bool = True) -> Dict[str, dict]:
    """
    Pose Estimator UI 실행

    Args:
        job_dir: Job 디렉토리 경로
        auto_open: 자동으로 브라우저 열기

    Returns:
        저장된 포즈 딕셔너리
    """
    ui = PoseEstimatorUI(job_dir)

    if not ui.images:
        print("[Pose Estimator] No images found in input directory")
        return {}

    # 이미 모든 이미지에 포즈가 있으면 스킵
    if len(ui.poses) >= len(ui.images):
        print(f"[Pose Estimator] All {len(ui.images)} images already have poses. Skipping.")
        return ui.poses

    print("\n" + "=" * 60)
    print("Camera Pose Estimator")
    print("=" * 60)
    print(f"Images: {len(ui.images)}")
    print(f"Existing poses: {len(ui.poses)}")
    print("=" * 60)

    demo = ui.create_ui()

    # Gradio 실행 (blocking)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=auto_open,
        quiet=True,
        prevent_thread_lock=False,
        share=False
    )

    return ui.poses


def check_poses_exist(job_dir: Path) -> bool:
    """포즈 파일이 존재하는지 확인"""
    poses_file = job_dir / "input" / "pose" / "camera_poses.json"
    if not poses_file.exists():
        return False

    with open(poses_file, 'r') as f:
        poses = json.load(f)

    # input 폴더의 이미지 수와 비교
    input_dir = job_dir / "input"
    images = [
        f.name for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ]

    return len(poses) >= len(images)


if __name__ == "__main__":
    # 테스트용
    import sys

    if len(sys.argv) > 1:
        job_path = Path(sys.argv[1])
    else:
        job_path = Path("pipeline/jobs/test")
        job_path.mkdir(parents=True, exist_ok=True)
        (job_path / "input").mkdir(exist_ok=True)

    poses = run_pose_estimator(job_path)
    print(f"\nCaptured {len(poses)} poses")
