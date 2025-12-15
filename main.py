"""
Few-shot Semantic Compositing Pipeline - Main Entry Point

전체 파이프라인:
1. 입력 이미지에서 가장 좋은 3장 선택
2. Veo3로 360도 뷰 영상 생성 (시뮬레이션)
3. 객체만 segmentation
4. 의미론적으로 적합한 배경 위치 찾기
5. 자연스러운 합성 (depth, lighting)
모듈 구조:
- depth.py: Depth 추론
- segment.py: Segmentation
- composite.py: 합성
- embedding.py: 텍스트 임베딩 및 유사도 계산
- utils.py: 파일 유틸리티 및 캐싱
- semantic_matcher.py: 의미론적 배경 매칭
"""

import os
import random
import sys
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

from google import genai
from google.genai import types

from semantic_matcher import SemanticMatcher
from composite import composite_on_segment, compute_segment_averaged_depth
from utils import frame_from_video, get_all_objects, load_camera_poses, compute_center, compute_azimuth_elevation, _fixed_grid_selection, _robust_greedy_selection
from depth import compute_depth
from segment import get_sam_model
from yolo_utils import (
    save_yolo_annotation,
    create_yolo_dataset_structure,
    split_train_val
)
from logging_utils import setup_logger


# ============================================================
# Option G: 밝기 + 색온도 조정 함수들
# ============================================================
def adjust_lighting_and_temperature(
    obj_cv: np.ndarray, 
    bg_cv: np.ndarray, 
    mask_bin: np.ndarray,
    brightness_strength: float = 0.2,
    temperature_strength: float = 0.2
) -> np.ndarray:
    """
    배경의 밝기와 색온도를 분석하여 객체에 적용 (고유 색상 유지)
    
    Args:
        obj_cv: 객체 이미지 (BGR)
        bg_cv: 배경 이미지 (BGR)
        mask_bin: 객체 마스크
        brightness_strength: 밝기 조정 강도 (사용되지 않음, 레거시)
        temperature_strength: 색온도 조정 강도 (사용되지 않음, 레거시)
    
    Returns:
        조정된 객체 이미지 (BGR)
    """
    # YOLO 학습 데이터 다양성을 위해 최대 조정 강도를 랜덤화
    # 30% ~ 80% 사이의 값을 사용
    max_adjustment_strength = random.uniform(0.3, 0.8)

    # 1. 배경 밝기 분석 (Lab의 L 채널)
    bg_lab = cv2.cvtColor(bg_cv, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_brightness = np.mean(bg_lab[:, :, 0])  # L 채널 평균
    
    # 2. 배경 색온도 분석 (R-B 차이)
    bg_r = np.mean(bg_cv[:, :, 2])  # Red
    bg_b = np.mean(bg_cv[:, :, 0])  # Blue
    bg_temperature = (bg_r - bg_b) / 255.0  # -1(차가움) ~ 1(따뜻함)
    
    # 3. 객체를 Lab으로 변환
    obj_lab = cv2.cvtColor(obj_cv, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # 객체의 현재 밝기
    obj_mask_bool = mask_bin > 0
    if np.sum(obj_mask_bool) > 0:
        obj_brightness = np.mean(obj_lab[:, :, 0][obj_mask_bool[:obj_lab.shape[0], :obj_lab.shape[1]]])
    else:
        obj_brightness = np.mean(obj_lab[:, :, 0])
    
    # 4. 밝기 조정 (Lab의 L 채널만) - 지수적으로 적용
    brightness_diff = bg_brightness - obj_brightness
    # 차이가 클수록 지수적으로 증가, 최대 강도는 랜덤화됨
    brightness_factor = min(max_adjustment_strength, 1 - np.exp(-abs(brightness_diff) / 30))
    obj_lab[:, :, 0] = np.clip(
        obj_lab[:, :, 0] + brightness_diff * brightness_factor,
        0, 255
    )
    
    # Lab → BGR 변환
    obj_adjusted = cv2.cvtColor(obj_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # 5. 색온도 조정 (R, B 채널) - 지수적으로 적용
    obj_temperature = (np.mean(obj_adjusted[:, :, 2]) - np.mean(obj_adjusted[:, :, 0])) / 255.0
    temperature_diff = bg_temperature - obj_temperature
    
    # 차이가 클수록 지수적으로 증가, 최대 강도는 랜덤화됨
    temperature_factor = min(max_adjustment_strength, 1 - np.exp(-abs(temperature_diff) / 0.2))
    temperature_shift = temperature_diff * temperature_factor * 50  # 최대 ±30 정도
    
    obj_adjusted[:, :, 2] = np.clip(
        obj_adjusted[:, :, 2].astype(np.float32) + temperature_shift,
        0, 255
    ).astype(np.uint8)  # Red
    
    obj_adjusted[:, :, 0] = np.clip(
        obj_adjusted[:, :, 0].astype(np.float32) - temperature_shift,
        0, 255
    ).astype(np.uint8)  # Blue
    
    return obj_adjusted


# 전역 설정
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "train"
INPUT_DIR = BASE_DIR / "input"
VIDEO_DIR = BASE_DIR / "output" / "video"
OUTPUT_DIR = BASE_DIR / "output" / "dataset"
MASKED_FRAMES_DIR = BASE_DIR / "output" / "dataset" / "masked_frames"
POSE_DIR = INPUT_DIR / "pose"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASKED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
POSE_DIR.mkdir(parents=True, exist_ok=True)
POSES_FILE = POSE_DIR / "camera_poses.json"

# Generative AI 클라이언트 초기화
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"  # 자신의 GCP 프로젝트 ID로 변경
    client = genai.Client(
        vertexai=True,
        project="kinetic-cosmos-479912-s0",
        location="us-central1"
    )
except Exception as e:
    print(f"Error initializing Google Generative AI client: {e}")
    sys.exit(1)

# ============================================================
# Step 1: 카메라 포즈를 이용해서 "좋은" 3장 선택
# ============================================================
def select_best_views(input_images: List[Path], num_views: int = 3) -> List[Path]:
    """
    Adaptive Hybrid 알고리즘으로 Veo3 최적화 뷰 선택
    
    - 균일 분포: Fixed Grid (0°, 120°, -120° 근처 선택)
    - 불규칙 분포: Robust Greedy (3D 거리 최대화)
    
    자동으로 입력 분포를 분석하여 최적 전략 사용
    만약 포즈 파일이 없거나 문제 있으면, 이전처럼 랜덤 3장으로 fallback.
    """
    def pick_best_3(
        poses: Dict[str, dict],
        elevation_limit: float = 30.0,
        num_views: int = 3
    ):
        """
        Adaptive Hybrid: Veo3 few-shot 최적화 알고리즘
        
        전략:
        - 균일 분포 입력: Fixed Grid (이상적 위치 0°, 120°, -120°)
        - 불규칙 입력: Robust Greedy (3D 거리 최대화)
        
        자동으로 입력 분포를 분석하여 최적 전략 선택
        """
        center = compute_center(poses)
        view_infos = []
        
        for filename, p in poses.items():
            azimuth, elevation, distance = compute_azimuth_elevation(p["position"], center)
            
            # 3D 단위 벡터로 변환 (구면 거리 계산용)
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            x = math.cos(el_rad) * math.sin(az_rad)
            y = math.sin(el_rad)
            z = math.cos(el_rad) * math.cos(az_rad)
            
            view_infos.append({
                "filename": filename,
                "azimuth": azimuth,
                "elevation": elevation,
                "distance": distance,
                "vec": (x, y, z)
            })
        
        # 1) Elevation 필터링
        filtered = [v for v in view_infos if abs(v["elevation"]) <= elevation_limit]
        if len(filtered) < num_views:
            print(f"  [WARN] Only {len(filtered)} views within ±{elevation_limit}°, relaxing filter")
            filtered = sorted(view_infos, key=lambda v: abs(v["elevation"]))[:max(num_views, len(view_infos))]
        
        # 2) 입력 분포 분석: 이상적 위치에 가까운 뷰가 있는지 확인
        ideal_positions = [(0, 0), (120, 0), (-120, 0)]
        close_to_ideal = []
        
        for ideal_az, ideal_el in ideal_positions:
            for v in filtered:
                # 원형 거리 계산
                az_diff = min(abs(v["azimuth"] - ideal_az), 360 - abs(v["azimuth"] - ideal_az))
                el_diff = abs(v["elevation"] - ideal_el)
                
                if az_diff < 25 and el_diff < 15:  # 이상적 위치 근처
                    close_to_ideal.append((ideal_az, v))
                    break
        
        # 3) 전략 선택
        if len(close_to_ideal) >= 2:
            print(f"  [INFO] Uniform distribution detected, using Fixed Grid strategy")
            return _fixed_grid_selection(filtered, ideal_positions)
        else:
            print(f"  [INFO] Irregular distribution detected, using Robust Greedy strategy")
            return _robust_greedy_selection(filtered)
    
    print(f"\n[Step 1] Selecting {num_views} best views from {len(input_images)} images using camera poses...")

    # 1) 포즈 JSON 로드
    if not POSES_FILE.exists():
        print(f"  [WARN] {POSES_FILE} not found. Falling back to random selection.")
        if len(input_images) <= num_views:
            return input_images
        return random.sample(input_images, num_views)

    poses = load_camera_poses(POSES_FILE)
    if not poses:
        print("  [WARN] No poses found in camera_poses.json. Falling back to random selection.")
        if len(input_images) <= num_views:
            return input_images
        return random.sample(input_images, num_views)

    # 2) 포즈 기반으로 '좋은 3장' 선택 (Veo3 최적화)
    chosen = pick_best_3(poses, elevation_limit=30.0, num_views=num_views)
    chosen_filenames = [v["filename"] for v in chosen]
    print("  Chosen filenames (from poses):", chosen_filenames)
    
    # 선택된 뷰의 상세 정보 출력
    for v in chosen:
        print(f"    - {v['filename']}: azimuth={v['azimuth']:.1f}°, elevation={v['elevation']:.1f}°")

    # 3) 파일 이름 -> 실제 Path로 매핑
    name_to_path = {p.name: p for p in input_images}
    selected_paths: List[Path] = []
    for name in chosen_filenames:
        if name in name_to_path:
            selected_paths.append(name_to_path[name])
        else:
            print(f"  [WARN] Pose exists for {name} but file not found in INPUT_DIR.")

    # 4) 혹시 3장 다 못 찾았으면, 나머지는 랜덤으로 채우기
    if len(selected_paths) < num_views:
        print(f"  [INFO] Only {len(selected_paths)} matched. Filling the rest randomly.")
        remaining = [p for p in input_images if p not in selected_paths]
        extra = random.sample(remaining, min(num_views - len(selected_paths), len(remaining)))
        selected_paths.extend(extra)

    print(f"  Final selected views: {[p.name for p in selected_paths]}")
    return selected_paths




# ============================================================
# Step 2: Veo3로 360도 뷰 영상 생성
# ============================================================
def generate_360_video_with_veo3(selected_images: List[Path]) -> Path:
    """
    Veo3 API를 호출하여 360도 뷰 영상 생성

    Args:
        selected_images: 선택된 로컬 이미지 경로 리스트

    Returns:
        생성된 360도 영상 파일 경로
    """
    print(f"\n[Step 2] Generating 360° video with Veo3 API...")

    # ------------------------------------
    # 1) Prompt 구성
    # ------------------------------------
    prompt_text = (
        f"Generate a high-fidelity 360-degree panoramic video of the object shown in the provided reference images.\n"
        f"Requirements:\n"
        f"1. The object must be fully visible in every frame; no parts should be cropped or cut off.\n"
        f"2. Maintain consistent scale, orientation, and position of the object across all frames.\n"
        f"3. Ensure smooth rotation; no sudden jumps, distortions, or unnatural movements.\n"
        f"4. The background should be minimal, neutral, or lightly styled, emphasizing the object.\n"
        f"5. Ensure consistent lighting, perspective, and color tone throughout.\n"
        f"6. Produce frames of uniform high visual fidelity, suitable for downstream training datasets.\n"
        f"7. Avoid any sudden changes or anomalies between consecutive frames.\n"
        f"Duration: 5 seconds."
    )

    print("  Calling Veo API... This may take a while...")
    reference_images = [
        types.VideoGenerationReferenceImage(
            image=types.Image.from_file(location=str(file)),
            reference_type="asset",
        )
        for file in selected_images
    ]
    
    try:
        operation = client.models.generate_videos(
            model="veo-3.1-generate-preview",
            source=types.GenerateVideosSource(prompt=prompt_text),
            config=types.GenerateVideosConfig(
                reference_images=reference_images[:3],
                aspect_ratio="16:9",
            ),
        )

        # Poll until done
        while not operation.done:
            print("Waiting for video generation...")
            time.sleep(15)
            operation = client.operations.get(operation)

        print("✓ Video generation completed!")
        print(operation.response)
        print(operation.result)
        if operation.response and operation.result.generated_videos:
            video_bytes = operation.result.generated_videos[0].video.video_bytes

            local_path = VIDEO_DIR / f"veo3_output_{int(time.time())}.mp4"
            with open(local_path, "wb") as f:
                f.write(video_bytes)

            print(f"✓ Saved generated video locally: {local_path}")
            return local_path


    except Exception as e:
        print(f"Veo API Exception: {e}")
        raise e

# ============================================================
# Step 3: 객체만 segmentation
# ============================================================
def segment_objects(frame_dir: Path) -> None:
    """
    Segments the main object from each frame in a directory and saves them to MASKED_FRAMES_DIR.

    Args:
        frame_dir: Directory containing the image frames to segment.
    """
    print(f"\n[Step 3] Segmenting objects from frames in {frame_dir} and saving to {MASKED_FRAMES_DIR}...")
    
    model = get_sam_model()
    
    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    if not frames:
        print(f"Warning: No frames found in {frame_dir} to segment.")
        return

    for idx, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"[WARN] Cannot read {frame_path.name}, skipping.")
            continue

        pred_results = model.predict(img, task="segment")

        if not pred_results or not hasattr(pred_results[0], 'masks') or pred_results[0].masks is None:
            print(f"[WARN] No mask detected in {frame_path.name}, skipping.")
            continue

        largest_mask = None
        largest_area = 0
        for mask in pred_results[0].masks.data:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            area = mask_np.sum()
            if area > largest_area:
                largest_area = area
                largest_mask = mask_np

        if largest_mask is None:
            print(f"[WARN] No mask data found in {frame_path.name}, skipping.")
            continue

        largest_mask = 1 - largest_mask
        kernel = np.ones((3, 3), np.uint8)
        largest_mask = cv2.erode(largest_mask, kernel, iterations=1)

        obj_only = np.zeros_like(img)
        obj_only[largest_mask == 1] = img[largest_mask == 1]
        alpha = (largest_mask * 255).astype(np.uint8)
        rgba_cv = cv2.cvtColor(obj_only, cv2.COLOR_BGR2BGRA)
        rgba_cv[:, :, 3] = alpha
        
        # 객체 영역만 crop (투명 픽셀 제거)
        ys, xs = np.where(alpha > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            rgba_cv_cropped = rgba_cv[y_min:y_max+1, x_min:x_max+1]
        else:
            # 객체가 없으면 원본 사용
            rgba_cv_cropped = rgba_cv
        
        rgba_pil = Image.fromarray(cv2.cvtColor(rgba_cv_cropped, cv2.COLOR_BGRA2RGBA))

        out_path = MASKED_FRAMES_DIR / f"{frame_path.stem}_masked.png"
        rgba_pil.save(out_path)
        original_size = rgba_cv.shape[:2]
        cropped_size = rgba_cv_cropped.shape[:2]
        print(f"✓ Segmented and saved: {out_path.name} ({original_size[0]}x{original_size[1]} → {cropped_size[0]}x{cropped_size[1]})")
    print(f"Completed segmentation of {len(frames)} frames.")

# ============================================================
# Step 4: 의미론적으로 적합한 배경 위치 찾기
# ============================================================
def find_suitable_backgrounds(
    object_category: str,
    semantic_locations: List[str],
    broad_categories: List[str] = None,
    max_backgrounds: int = 5,
    similarity_threshold: float = 0.8,
    max_workers: int = 5,
    upscale_backgrounds: bool = True,
    upscale_factor: float = 4.0
) -> List[Dict]:
    """
    의미론적으로 적합한 배경 위치 찾기
    
    Args:
        object_category: 객체 카테고리
        semantic_locations: 의미론적 위치 리스트 (예: ["shelf", "table"])
        broad_categories: 대분류 카테고리 (예: ["home", "indoor"])
        max_backgrounds: 최대 배경 개수
        similarity_threshold: 유사도 임계값
        max_workers: 병렬 워커 수
        upscale_backgrounds: 배경 이미지 업스케일링 여부 (기본 True)
        upscale_factor: 업스케일 배율 (기본 4배: 256x256 → 1024x1024)
    
    Returns:
        배경 정보 리스트 [{"bg_image": Image, "segment_mask": np.ndarray, ...}]
    """
    print(f"\n[Step 4] Finding semantically suitable backgrounds...")
    print(f"Object category: {object_category}")
    print(f"Semantic locations: {semantic_locations}")
    if broad_categories:
        print(f"  Broad categories: {broad_categories}")
    if upscale_backgrounds:
        print(f"  Background upscaling: ENABLED ({upscale_factor}x)")
    
    matcher = SemanticMatcher(
        similarity_threshold=similarity_threshold,
        upscale_backgrounds=upscale_backgrounds,
        upscale_factor=upscale_factor
    )
    
    suitable_backgrounds = matcher.find_suitable_backgrounds(
        semantic_locations=semantic_locations,
        dataset_dir=DATASET_DIR,
        max_backgrounds=max_backgrounds,
        max_workers=max_workers,
        broad_categories=broad_categories
    )
    
    # 캐시 저장
    matcher.embedding_manager.save_label_cache()
    
    print(f"Found {len(suitable_backgrounds)} suitable backgrounds")
    return suitable_backgrounds


# ============================================================
# Poisson blending helper
# ============================================================
def blend_object(
    obj_image: Image.Image,
    bg_image: Image.Image,
    brightness_strength: float = 0.3,
    temperature_strength: float = 0.4,
    max_strength: float = 0.6,  # 최대 60%
    erode_iterations: int = 1,  # 마스크 침식 반복 횟수
):
    """
    Option G: 밝기 + 색온도 조정 + 순수 알파 블렌딩
    
    Args:
        obj_image: 객체 이미지 (RGBA)
        bg_image: 배경 이미지
        brightness_strength: 밝기 조정 강도 (기본 30%)
        temperature_strength: 색온도 조정 강도 (기본 40%)
        max_strength: 최대 강도 (기본 60%)
    
    Returns:
        합성된 이미지
    """
    obj_rgba = obj_image.convert("RGBA")
    bg_rgba = bg_image.convert("RGBA")

    obj_cv = cv2.cvtColor(np.array(obj_rgba), cv2.COLOR_RGBA2BGRA)
    bg_cv = cv2.cvtColor(np.array(bg_rgba), cv2.COLOR_RGBA2BGRA)

    bg_h, bg_w = bg_cv.shape[:2]
    obj_h, obj_w = obj_cv.shape[:2]
    
    # 객체 리사이즈
    if obj_w > bg_w or obj_h > bg_h:
        scale = min(bg_w / obj_w, bg_h / obj_h) * 0.9
        obj_cv = cv2.resize(obj_cv, (int(obj_w * scale), int(obj_h * scale)), interpolation=cv2.INTER_AREA)
        obj_h, obj_w = obj_cv.shape[:2]

    # 마스크 생성
    if obj_cv.shape[2] == 4:
        mask = obj_cv[:, :, 3]
    else:
        mask = cv2.cvtColor(obj_cv, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    
    # 마스크 침식: 테두리 픽셀 제거 (흰색/검은색 경계선 문제 해결)
    if erode_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_bin = cv2.erode(mask_bin, kernel, iterations=erode_iterations)
    
    # 알파 채널 부드럽게 (경계 블렌딩) - 최소한만 적용
    mask_float = mask_bin.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0.3)

    # 밝기 + 색온도 조정
    obj_adjusted = adjust_lighting_and_temperature(
        obj_cv[:, :, :3], 
        bg_cv[:, :, :3], 
        (mask_float * 255).astype(np.uint8),
        brightness_strength=brightness_strength,
        temperature_strength=temperature_strength
    )
    
    # 순수 알파 블렌딩 (Poisson 제거)
    result = bg_cv[:, :, :3].copy()
    y_offset = (bg_h - obj_h) // 2
    x_offset = (bg_w - obj_w) // 2
    
    for i in range(obj_h):
        for j in range(obj_w):
            alpha = mask_float[i, j]
            if alpha > 0.01:  # 거의 투명한 픽셀 무시
                by, bx = y_offset + i, x_offset + j
                if 0 <= by < bg_h and 0 <= bx < bg_w:
                    result[by, bx] = (
                        obj_adjusted[i, j] * alpha + 
                        result[by, bx] * (1 - alpha)
                    ).astype(np.uint8)
    
    # BGR로 변환 후 반환
    result_bgra = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_bgra[:, :, 3] = 255  # 완전 불투명
    return Image.fromarray(cv2.cvtColor(result_bgra, cv2.COLOR_BGRA2RGBA))


# ============================================================
# Step 5: 자연스러운 합성 (동적 목표 달성)
# ============================================================

def process_single_composite(
    task: Dict,
    use_depth: bool,
    use_lighting: bool,
    depth_offset: float,
    occlusion_threshold: float,
    split_name: str,
    image_dir: Path,
    label_dir: Path,
    yolo_base: Path
) -> Tuple[Path, int]:
    """
    단일 합성 작업 처리 (병렬화용)
    
    Returns:
        (image_path, original_idx) 또는 None (실패 시)
    """
    logger = logging.getLogger(__name__)
    
    bg_idx = task['bg_idx']
    bg_info = task['bg_info']
    obj_image = task['obj_image'].copy()  # 멀티스레드 안전성
    obj_meta = task['obj_meta']
    original_idx = task['original_idx']
    
    try:
        # Depth 맵 계산
        bg_depth_map, averaged_depth_map = None, None
        if use_depth:
            _, bg_depth_map = compute_depth(bg_info["bg_image"])
            from segment import run_segmentation
            annotations, _ = run_segmentation(bg_info["bg_image"])
            segments = [{"segmentation": mask} for mask, label in annotations]
            averaged_depth_map = compute_segment_averaged_depth(bg_depth_map, segments)
        
        # 데이터 증강: 크기, 회전, 반전
        max_dim = max(obj_image.size)
        target_size = random.uniform(350, 430)
        scale = target_size / max_dim
        rotation = random.uniform(-15, 15)
        if random.random() < 0.5:
            obj_image = obj_image.transpose(Image.FLIP_LEFT_RIGHT)

        # 합성
        composite_img, occlusion_ratio, _ = composite_on_segment(
            bg_info["bg_image"], bg_info["segment_mask"], obj_image,
            base_scale=scale, use_depth=use_depth,
            bg_depth_map=averaged_depth_map,
            depth_offset=depth_offset, rotation_angle=rotation
        )

        if occlusion_ratio >= occlusion_threshold:
            return None
        
        # 최종 마스크 계산
        bg_rgb = bg_info["bg_image"].convert("RGB")
        comp_rgb = composite_img.convert("RGB")
        bg_cv = cv2.cvtColor(np.array(bg_rgb), cv2.COLOR_RGB2BGR)
        comp_cv = cv2.cvtColor(np.array(comp_rgb), cv2.COLOR_RGB2BGR)
        diff_gray = cv2.cvtColor(cv2.absdiff(comp_cv, bg_cv), cv2.COLOR_BGR2GRAY)
        _, obj_mask = cv2.threshold(diff_gray, 1, 255, cv2.THRESH_BINARY)

        if obj_mask.max() == 0:
            return None
        
        # 조명 조정 및 블렌딩
        obj_only = cv2.bitwise_and(comp_cv, comp_cv, mask=obj_mask)
        obj_only_rgba = cv2.cvtColor(obj_only, cv2.COLOR_BGR2BGRA)
        obj_only_rgba[:, :, 3] = obj_mask
        obj_cutout = Image.fromarray(cv2.cvtColor(obj_only_rgba, cv2.COLOR_BGRA2RGBA))
        blended_img = blend_object(obj_cutout, bg_info["bg_image"])
        
        # 파일 저장
        output_filename = f"composite_{split_name}_bg{bg_idx:04d}_obj{original_idx:04d}.png"
        image_path = image_dir / output_filename
        label_path = label_dir / output_filename.replace('.png', '.txt')
        blended_img.save(image_path)
        
        # 배경 원본도 저장 (negative sampling용)
        bg_save_dir = yolo_base / "backgrounds" / split_name
        bg_save_dir.mkdir(parents=True, exist_ok=True)
        bg_filename = f"bg{bg_idx:04d}.png"
        bg_save_path = bg_save_dir / bg_filename
        if not bg_save_path.exists():  # 중복 저장 방지
            bg_info["bg_image"].save(bg_save_path)
        
        # YOLO 어노테이션 저장
        if save_yolo_annotation(obj_mask, blended_img.size[0], blended_img.size[1], label_path, class_id=0):
            return (image_path, original_idx)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error processing bg{bg_idx}_obj{original_idx}: {e}")
        return None


def composite_naturally(
    objects: List[Tuple[Image.Image, Dict]],
    backgrounds: List[Dict],
    use_depth: bool = False,
    use_lighting: bool = False,
    overlay_scale: float = 1.0,
    depth_offset: float = 0.05,
    target_images: int = 1600,
    occlusion_threshold: float = 0.3,
    class_name: str = "object",
    max_workers: int = 5
) -> List[Path]:
    """
    동적으로 목표 개수에 맞춰 객체를 배경에 합성 (YOLO 데이터셋 형식, 병렬 처리)
    
    전략:
    - 목표: target_images개 생성
    - 배경 부족 시 → 배경당 객체 개수 자동 증가
    - 객체 균등 분배 → 모든 객체가 비슷한 횟수로 사용됨
    - YOLO segmentation format으로 저장 (images/ + labels/)
    - 병렬 처리로 성능 향상 (max_workers 설정 가능)
    
    Args:
        objects: [(누끼 이미지, 메타데이터)] 리스트
        backgrounds: 배경 정보 리스트
        use_depth: Depth 기반 occlusion 사용 여부
        use_lighting: Lighting 조정 사용 여부 (Option G 적용)
        overlay_scale: 오버레이 스케일
        depth_offset: overlay를 앞으로 당길 오프셋 (기본 0.05)
        target_images: 목표 이미지 개수 (기본 1600)
        occlusion_threshold: 가려짐 비율 임계값 (0.3 = 30% 이상 가려지면 스킵)
        class_name: YOLO 클래스 이름 (기본 "object")
        max_workers: 병렬 처리 워커 수 (기본 7)
    
    Returns:
        저장된 합성 이미지 경로 리스트
    """
    
    # 로거 설정
    logger, log_file = setup_logger()
    logger.info(f"Starting composite_naturally (parallel with {max_workers} workers)")
    logger.info(f"Target images: {target_images}")
    logger.info(f"Backgrounds: {len(backgrounds)}")
    logger.info(f"Objects: {len(objects)}")
    logger.info(f"Use depth: {use_depth}")
    logger.info(f"Use lighting: {use_lighting}")
    logger.info(f"YOLO class: {class_name}")

    print(f"\n[Step 5] Compositing objects with parallel processing ({max_workers} workers)...")
    print(f"  Target: {target_images} images from {len(backgrounds)} backgrounds")
    
    if not objects or not backgrounds:
        logger.error("No objects or backgrounds to process.")
        print("  Error: No objects or backgrounds to process.")
        return []

    # YOLO 데이터셋 구조 생성
    yolo_base = OUTPUT_DIR / "result"
    images_train, images_val, labels_train, labels_val = create_yolo_dataset_structure(
        yolo_base, [class_name]
    )

    # 1. 배경을 train/val로 미리 분리하여 데이터 유출 방지
    random.shuffle(backgrounds)
    val_ratio = 0.2
    val_count = int(len(backgrounds) * val_ratio)
    if len(backgrounds) > 1 and val_count == 0:  # 최소 1개는 검증용으로
        val_count = 1
    
    train_bgs = backgrounds[val_count:]
    val_bgs = backgrounds[:val_count]
    
    logger.info(f"Splitting backgrounds: {len(train_bgs)} train / {len(val_bgs)} val")
    print(f"  Splitting backgrounds: {len(train_bgs)} train / {len(val_bgs)} val")
    print(f"  Occlusion threshold: {occlusion_threshold:.1%}")

    output_paths = []
    object_usage_count = {i: 0 for i in range(len(objects))}
    
    total_generated_images = 0

    # 2. Train/Val 각 세트에 대해 생성 루프 실행
    dataset_splits = [
        ("train", train_bgs, images_train, labels_train),
        ("val", val_bgs, images_val, labels_val),
    ]

    pbar = tqdm(total=target_images, desc="Generating images", unit="img")

    for split_name, split_bgs, image_dir, label_dir in dataset_splits:
        if not split_bgs:
            logger.warning(f"No backgrounds for {split_name} split, skipping.")
            continue
        
        # 해당 스플릿의 목표 이미지 수 계산
        total_bgs = len(backgrounds)
        split_ratio = len(split_bgs) / total_bgs if total_bgs > 0 else 0
        
        # val 셋에 최소 1개 이상의 bg가 할당되었을 때, val_ratio가 0이 되는 것을 방지
        if split_name == 'val' and val_count > 0 and total_bgs > 0:
            actual_val_ratio = val_count / total_bgs
            target_split_images = int(target_images * actual_val_ratio)
        else:
            target_split_images = int(target_images * split_ratio)

        # val set에 bg가 있는데 target image가 0개이면 최소 1개 생성
        if target_split_images == 0 and len(split_bgs) > 0:
            target_split_images = 1

        objects_per_bg = max(1, int(target_split_images / len(split_bgs))) if len(split_bgs) > 0 else 1
        
        logger.info(f"Generating {split_name} set (target: ~{target_split_images} images)...")
        logger.info(f"  - Using {len(split_bgs)} backgrounds, ~{objects_per_bg} objects per background.")
        logger.info(f"  - Parallel workers: {max_workers}")
        
        current_split_image_count = 0

        # 작업 리스트 생성 (배경 + 객체 조합)
        tasks = []
        for bg_idx, bg_info in enumerate(split_bgs):
            if current_split_image_count >= target_split_images:
                break
            if total_generated_images >= target_images:
                break

            # 객체 선택 (사용 횟수 적은 순)
            sorted_obj_indices = sorted(object_usage_count.keys(), key=lambda x: object_usage_count[x])
            selected_indices = sorted_obj_indices[:min(objects_per_bg, len(objects))]
            
            for obj_idx in selected_indices:
                tasks.append({
                    'bg_idx': bg_idx,
                    'bg_info': bg_info,
                    'obj_idx': obj_idx,
                    'obj_image': objects[obj_idx][0],
                    'obj_meta': objects[obj_idx][1],
                    'original_idx': obj_idx
                })
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    process_single_composite,
                    task,
                    use_depth,
                    use_lighting,
                    depth_offset,
                    occlusion_threshold,
                    split_name,
                    image_dir,
                    label_dir,
                    yolo_base
                ): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        image_path, original_idx = result
                        output_paths.append(image_path)
                        current_split_image_count += 1
                        total_generated_images += 1
                        object_usage_count[original_idx] += 1
                        pbar.update(1)
                        pbar.set_postfix({"split": split_name})
                        
                        # 목표 달성 시 중단
                        if current_split_image_count >= target_split_images or total_generated_images >= target_images:
                            for f in future_to_task:
                                f.cancel()
                            break
                            
                except Exception as e:
                    logger.error(f"Error processing task: {e}", exc_info=True)

    
    pbar.close()
    logger.info(f"\nGenerated {len(output_paths)} total composite images")
    print(f"\n\n✓ Generated {len(output_paths)} total composite images")
    
    if any(object_usage_count.values()):
        min_usage = min(object_usage_count.values())
        max_usage = max(object_usage_count.values())
        avg_usage = sum(object_usage_count.values()) / len(object_usage_count)
        logger.info(f"Object usage: min={min_usage}, max={max_usage}, avg={avg_usage:.1f}")
        print(f"  Object usage: min={min_usage}, max={max_usage}, avg={avg_usage:.1f}")
    
    return output_paths


def composite_naturally_old(
    objects: List[Tuple[Image.Image, Dict]],
    backgrounds: List[Dict],
    use_depth: bool = False,
    use_lighting: bool = False,
    overlay_scale: float = 1.0,
    depth_offset: float = 0.05,
    target_images: int = 1600,
    occlusion_threshold: float = 0.3,
    class_name: str = "object"
) -> List[Path]:
    """
    동적으로 목표 개수에 맞춰 객체를 배경에 합성 (YOLO 데이터셋 형식)
    
    전략:
    - 목표: target_images개 생성
    - 배경 부족 시 → 배경당 객체 개수 자동 증가
    - 객체 균등 분배 → 모든 객체가 비슷한 횟수로 사용됨
    - YOLO segmentation format으로 저장 (images/ + labels/)
    
    Args:
        objects: [(누끼 이미지, 메타데이터)] 리스트
        backgrounds: 배경 정보 리스트
        use_depth: Depth 기반 occlusion 사용 여부
        use_lighting: Lighting 조정 사용 여부 (Option G 적용)
        overlay_scale: 오버레이 스케일
        depth_offset: overlay를 앞으로 당길 오프셋 (기본 0.05)
        target_images: 목표 이미지 개수 (기본 1200)
        occlusion_threshold: 가려짐 비율 임계값 (0.15 = 15% 이상 가려지면 스킵)
        class_name: YOLO 클래스 이름 (기본 "object")
    
    Returns:
        저장된 합성 이미지 경로 리스트
    """
    
    print(f"\n[Step 5] Compositing objects naturally with clean train/val split...")
    print(f"  Target images: {target_images}")
    print(f"  Total backgrounds: {len(backgrounds)}")
    print(f"  Objects: {len(objects)}")
    print(f"  Use depth: {use_depth}")
    print(f"  Use lighting: {use_lighting}")
    print(f"  YOLO class: {class_name}")

    if not objects or not backgrounds:
        print("  Error: No objects or backgrounds to process.")
        return []

    # YOLO 데이터셋 구조 생성
    yolo_base = OUTPUT_DIR / "result"
    images_train, images_val, labels_train, labels_val = create_yolo_dataset_structure(
        yolo_base, [class_name]
    )

    # 1. 배경을 train/val로 미리 분리
    random.shuffle(backgrounds)
    val_ratio = 0.2
    val_count = int(len(backgrounds) * val_ratio)
    if len(backgrounds) > 1 and val_count == 0:  # 최소 1개는 검증용으로
        val_count = 1
    
    train_bgs = backgrounds[val_count:]
    val_bgs = backgrounds[:val_count]
    
    print(f"  Splitting backgrounds: {len(train_bgs)} train / {len(val_bgs)} val")
    print(f"  Occlusion threshold: {occlusion_threshold:.1%}")

    output_paths = []
    object_usage_count = {i: 0 for i in range(len(objects))}

    # 2. Train/Val 각 세트에 대해 생성 루프 실행
    dataset_splits = [
        ("train", train_bgs, images_train, labels_train),
        ("val", val_bgs, images_val, labels_val),
    ]

    for split_name, split_bgs, image_dir, label_dir in dataset_splits:
        if not split_bgs:
            print(f"\nNo backgrounds for {split_name} split, skipping.")
            continue
        
        # 해당 스플릿의 목표 이미지 수 계산
        split_ratio = len(split_bgs) / len(backgrounds) if len(backgrounds) > 0 else 0
        target_split_images = int(target_images * split_ratio) if split_name == 'train' else int(target_images * val_ratio)

        if target_split_images == 0 and len(split_bgs) > 0:
            target_split_images = 1 # Ensure at least one image is generated for val if there are val backgrounds

        objects_per_bg = max(1, int(target_split_images / len(split_bgs))) if len(split_bgs) > 0 else 1
        
        print(f"\nGenerating {split_name} set (target: ~{target_split_images} images)...")
        print(f"  - Using {len(split_bgs)} backgrounds, ~{objects_per_bg} objects per background.")
        
        current_split_image_count = 0

        for bg_idx, bg_info in enumerate(split_bgs):
            if current_split_image_count >= target_split_images:
                break
        
            sorted_obj_indices = sorted(object_usage_count.keys(), key=lambda x: object_usage_count[x])
            selected_indices = sorted_obj_indices[:min(objects_per_bg, len(objects))]
            selected_objects = [(objects[i][0], objects[i][1], i) for i in selected_indices]
            
            bg_depth_map, averaged_depth_map = None, None
            if use_depth:
                _, bg_depth_map = compute_depth(bg_info["bg_image"])
                from segment import run_segmentation
                annotations, _ = run_segmentation(bg_info["bg_image"])
                segments = [{"segmentation": mask} for mask, label in annotations]
                averaged_depth_map = compute_segment_averaged_depth(bg_depth_map, segments)
            
            for obj_idx, (obj_image, obj_meta, original_idx) in enumerate(selected_objects):
                if current_split_image_count >= target_split_images:
                    break
                    
                try:
                    min_dim = min(obj_image.size)
                    scale = random.uniform(330 / min_dim, 400 / min_dim)
                    rotation = random.uniform(-30, 30)
                    
                    composite_img, occlusion_ratio, visible_mask = composite_on_segment(
                        bg_info["bg_image"], bg_info["segment_mask"], obj_image,
                        base_scale=scale, use_depth=use_depth,
                        bg_depth_map=averaged_depth_map,
                        depth_offset=depth_offset, rotation_angle=rotation
                    )

                    if occlusion_ratio >= occlusion_threshold:
                        continue
                    
                    bg_rgb = bg_info["bg_image"].convert("RGB")
                    comp_rgb = composite_img.convert("RGB")
                    bg_cv = cv2.cvtColor(np.array(bg_rgb), cv2.COLOR_RGB2BGR)
                    comp_cv = cv2.cvtColor(np.array(comp_rgb), cv2.COLOR_RGB2BGR)
                    diff_gray = cv2.cvtColor(cv2.absdiff(comp_cv, bg_cv), cv2.COLOR_BGR2GRAY)
                    _, obj_mask = cv2.threshold(diff_gray, 1, 255, cv2.THRESH_BINARY)

                    if obj_mask.max() == 0: continue
                    
                    obj_only = cv2.bitwise_and(comp_cv, comp_cv, mask=obj_mask)
                    obj_only_rgba = cv2.cvtColor(obj_only, cv2.COLOR_BGR2BGRA)
                    obj_only_rgba[:, :, 3] = obj_mask
                    obj_cutout = Image.fromarray(cv2.cvtColor(obj_only_rgba, cv2.COLOR_BGRA2RGBA))
                    blended_img = blend_object(obj_cutout, bg_info["bg_image"])
                    
                    output_filename = f"composite_{split_name}_bg{bg_idx:04d}_obj{original_idx:04d}.png"
                    image_path = image_dir / output_filename
                    label_path = label_dir / output_filename.replace('.png', '.txt')

                    blended_img.save(image_path)
                    
                    if save_yolo_annotation(visible_mask, blended_img.size[0], blended_img.size[1], label_path, class_id=0):
                        output_paths.append(image_path)
                        current_split_image_count += 1
                        object_usage_count[original_idx] += 1
                        print(f"      Saved: {output_filename} (Total: {len(output_paths)})", end='\r')
                    
                except Exception as e:
                    print(f"\n    ✗ Error on obj {obj_idx+1} in {split_name}: {e}")

    print(f"\n\n✓ Generated {len(output_paths)} total composite images")
    
    if any(object_usage_count.values()):
        min_usage = min(object_usage_count.values())
        max_usage = max(object_usage_count.values())
        avg_usage = sum(object_usage_count.values()) / len(object_usage_count)
        print(f"  Object usage: min={min_usage}, max={max_usage}, avg={avg_usage:.1f}")
    
    return output_paths




# ============================================================
# 전체 파이프라인 실행 함수
# ============================================================
def run_full_pipeline(
    object_category: str = "object",
    semantic_locations: List[str] = None,
    broad_categories: List[str] = None,
    num_views: int = 3,
    max_backgrounds: int = 800,
    similarity_threshold: float = 0.7,
    overlay_scale: float = 1.0,
    use_depth: bool = True,
    use_lighting: bool = False,
    max_workers: int = 5,
    target_images: int = 1600
) -> List[Path]:
    """
    전체 파이프라인을 순차적으로 실행
    
    Args:
        object_category: 객체 카테고리
        semantic_locations: 의미론적 위치 리스트
        broad_categories: 대분류 카테고리 리스트
        num_views: 선택할 뷰 개수
        max_backgrounds: 최대 배경 개수 (기본 800)
        similarity_threshold: 유사도 임계값 (기본 0.7)
        overlay_scale: 오버레이 스케일
        use_depth: Depth 사용 여부
        use_lighting: Lighting 사용 여부
        max_workers: 병렬 워커 수
        target_images: 목표 이미지 개수 (기본 1200)
    
    Returns:
        생성된 합성 이미지 경로 리스트
    """
    print("=" * 60)
    print("Few-shot Semantic Compositing Pipeline")
    print(f"Target: {target_images} images")
    print("=" * 60)
    
    # Step 1: 입력 이미지 로드
    input_images = (
        list(INPUT_DIR.glob("*.png")) +
        list(INPUT_DIR.glob("*.jpg"))
    )
    if len(input_images) == 0:
        print(f"Error: No images found in {INPUT_DIR}")
        return []
    print(f"Found {len(input_images)} input images")
    
    # Step 1: 뷰 선택
    selected_views = select_best_views(input_images, num_views=num_views)
    
    # Step 2: Veo3 360도 영상 생성
    video_path = generate_360_video_with_veo3(selected_views)

    # 프레임 분리
    frames_dir = OUTPUT_DIR / "frames"
    frame_from_video(video_path, frames_dir)
    
    # Step 3: 객체 segmentation
    segment_objects(frames_dir)
    
    # Step 4: 적합한 배경 찾기
    backgrounds = find_suitable_backgrounds(
        object_category=object_category,
        semantic_locations=semantic_locations,
        broad_categories=broad_categories,
        max_backgrounds=max_backgrounds,
        similarity_threshold=similarity_threshold,
        max_workers=max_workers,
        upscale_backgrounds=True,  # 배경 업스케일링 활성화
        upscale_factor=4.0  # 256x256 → 1024x1024
    )
    
    if len(backgrounds) == 0:
        print("Error: No suitable backgrounds found!")
        return []
    
    objects = get_all_objects(
        original_input_dir=INPUT_DIR,
        masked_frames_dir=MASKED_FRAMES_DIR
    )

    # Step 5: 자연스러운 합성 (YOLO 데이터셋 생성)
    output_paths = composite_naturally(
        objects=objects,
        backgrounds=backgrounds,
        use_depth=use_depth,
        use_lighting=use_lighting,
        overlay_scale=overlay_scale,
        depth_offset=0.05,  # overlay를 segment보다 5% 앞에 배치
        target_images=target_images,
        class_name=object_category.replace(" ", "_")  # 공백을 언더스코어로
    )
    
    print("\n" + "=" * 60)
    print(f"Pipeline completed! Generated {len(output_paths)} images")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return output_paths


def main():
    """
    메인 엔트리 포인트
    
    전체 파이프라인:
    1. 입력 이미지에서 가장 좋은 3장 선택
    2. Veo3로 360도 뷰 영상 생성 (시뮬레이션)
    3. 객체만 segmentation
    4. 의미론적으로 적합한 배경 위치 찾기
    5. 자연스러운 합성 (depth, lighting)
    """
    print("=== Few-shot Semantic Compositing Pipeline ===")
    print("Step 1: Select 3 appropriate views from input images")
    print("Step 2: Generate 360° video with Veo3 (Simulated)")
    print("Step 3: Segment objects from video")
    print("Step 4: Find semantically suitable backgrounds")
    print("Step 5: Composite naturally with depth and lighting")
    print()
    
    # 전체 파이프라인 실행
    results = run_full_pipeline(
        object_category="monkey doll",
        semantic_locations=["shelf", "bed", "couch", "table", "toy box"],
        broad_categories=["home", "indoor", "living room", "bedroom", "house interior"],
        num_views=3,
        max_backgrounds=5,
        similarity_threshold=0.8,
        overlay_scale=0.8,
        use_depth=True,  # Depth 기반 occlusion 사용
        use_lighting=False,
        max_workers=5
    )
    
    if results:
        print(f"\nGenerated images:")
        for r in results:
            print(f"  - {r.name}")
    else:
        print("\nNo images generated. Check the logs above for errors.")


if __name__ == "__main__":
    main()
