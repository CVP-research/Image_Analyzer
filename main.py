"""
Few-shot Semantic Compositing Pipeline - Main Entry Point

전체 파이프라인:
1. [TODO] 입력 이미지에서 가장 좋은 3장 선택
2. [IMPLEMENTED] Veo3로 360도 뷰 영상 생성 (시뮬레이션)
3. [IMPLEMENTED] 객체만 segmentation
4. [IMPLEMENTED] 의미론적으로 적합한 배경 위치 찾기
5. [TODO] 자연스러운 합성 (depth, lighting)

모듈 구조:
- depth.py: Depth 추론
- segment.py: Segmentation
- composite.py: 합성
- embedding.py: 텍스트 임베딩 및 유사도 계산
- utils.py: 파일 유틸리티 및 캐싱
- semantic_matcher.py: 의미론적 배경 매칭
"""

import hashlib
import os
import random
import sys
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import SAM

from google import genai
from google.genai import types

from semantic_matcher import SemanticMatcher
from composite import composite_on_segment, compute_segment_averaged_depth
from utils import get_all_objects
from depth import compute_depth

# 전역 설정
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "train"
INPUT_DIR = BASE_DIR / "input"
VIDEO_DIR = BASE_DIR / "output" / "video"
OUTPUT_DIR = BASE_DIR / "output" / "dataset"
MASKED_FRAMES_DIR = BASE_DIR / "output" / "dataset" / "masked_frames"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASKED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
POSES_FILE = BASE_DIR.parent / "backend" / "camera_poses.json"

SAM_MODEL = None

# Generative AI 클라이언트 초기화
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"  # 자신의 GCP 프로젝트 ID로 변경
    client = genai.Client(
        vertexai=True,
        project="gen-lang-client-0127629302",
        location="us-central1"
    )
except Exception as e:
    print(f"Error initializing Google Generative AI client: {e}")
    sys.exit(1)

# Global variable for the SAM model to ensure it's loaded only once.


def get_sam_model():
    """Loads and returns the SAM model, caching it globally."""
    global SAM_MODEL
    if SAM_MODEL is None:
        print("Loading SAM model...")
        SAM_MODEL = SAM("sam2_l.pt")
        print("SAM model loaded.")
    return SAM_MODEL


def load_camera_poses(path: Path) -> Dict[str, dict]:
    """backend/camera_poses.json 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_center(poses: Dict[str, dict]) -> List[float]:
    """
    모든 look_at의 평균을 '객체 중심'으로 사용.
    (대부분 [0,0,0]일 가능성이 높지만, 일반화해서 계산)
    """
    xs, ys, zs = [], [], []
    for p in poses.values():
        lx, ly, lz = p["look_at"]
        xs.append(lx); ys.append(ly); zs.append(lz)

    n = len(xs) if xs else 1
    return [sum(xs)/n, sum(ys)/n, sum(zs)/n]


def circular_distance(a: float, b: float) -> float:
    """원 위에서 두 각도의 차이를 0~180도 범위로 계산"""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def pick_best_3(
    poses: Dict[str, dict],
    elevation_limit: float = 60.0,
    num_views: int = 3
):
    """
    포즈들로부터 '좋은' 3장 선택
    - elevation 너무 큰 건 제외 (탑뷰/바닥뷰 제거)
    - 정면(azimuth≈0°) 1장은 무조건 포함
    - 나머지 2장은 수평 방향(azimuth)이 서로 최대한 멀리 떨어지게
    """
    center = compute_center(poses)
    cx, cy, cz = center

    view_infos = []
    for filename, p in poses.items():
        px, py, pz = p["position"]

        # 객체 중심 기준 위치 벡터
        vx, vy, vz = px - cx, py - cy, pz - cz
        r = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-8

        # 수평각(azimuth), 수직각(elevation)
        azimuth = math.degrees(math.atan2(vx, vz))      # x-z 평면 기준
        elevation = math.degrees(math.asin(vy / r))     # y 기준

        view_infos.append({
            "filename": filename,
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": r,
        })

    # 1) elevation 필터: 너무 위/아래(예: > 60도)는 제거
    filtered = [
        v for v in view_infos
        if abs(v["elevation"]) <= elevation_limit
    ]
    if len(filtered) < num_views:
        # 필터링하고 남은 게 너무 적으면 그냥 다 씀
        filtered = view_infos

    # 2) 정면 찾기: azimuth 절대값이 최소인 뷰
    front_view = min(filtered, key=lambda v: abs(v["azimuth"]))

    # front를 뺀 나머지 후보들
    remaining = [v for v in filtered if v is not front_view]

    if len(remaining) <= 2:
        chosen = [front_view] + remaining
        chosen = chosen[:3]
        chosen.sort(key=lambda v: v["azimuth"])
        return chosen

    # 3) 두 번째 뷰: front와 수평각이 가장 멀리 떨어진 뷰
    second = max(
        remaining,
        key=lambda v: circular_distance(v["azimuth"], front_view["azimuth"])
    )

    # 4) 세 번째 뷰: front / second 양쪽과 모두 멀리 떨어진 뷰
    remaining2 = [v for v in remaining if v is not second]

    def score(v):
        return min(
            circular_distance(v["azimuth"], front_view["azimuth"]),
            circular_distance(v["azimuth"], second["azimuth"])
        )

    third = max(remaining2, key=score)

    chosen = [front_view, second, third]
    chosen.sort(key=lambda v: v["azimuth"])
    return chosen

# ============================================================
# Step 1: 입력 이미지에서 랜덤으로 3장 선택
# ============================================================

# ============================================================
# Step 1: 카메라 포즈를 이용해서 "좋은" 3장 선택
# ============================================================
def select_best_views(input_images: List[Path], num_views: int = 3) -> List[Path]:
    """
    camera_poses.json에 저장된 카메라 포즈 정보를 사용해서
    - 너무 위/아래에서 찍힌 샷(elevation 큰 것)은 제외하고
    - 정면(azimuth≈0°) 1장을 무조건 포함시키고
    - 나머지 2장은 수평각(azimuth)이 서로 최대한 벌어지도록 선택

    만약 포즈 파일이 없거나 문제 있으면, 이전처럼 랜덤 3장으로 fallback.
    """
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

    # 2) 포즈 기반으로 '좋은 3장' 선택 (정면 포함 로직)
    chosen = pick_best_3(poses, elevation_limit=60.0, num_views=num_views)
    chosen_filenames = [v["filename"] for v in chosen]
    print("  Chosen filenames (from poses):", chosen_filenames)

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


def frame_from_video(video_path: Path, output_dir: Path) -> None:
    """
    Veo3로 생성된 360도 영상에서 유일한 프레임만 분리 및 저장
    
    Args:
        video_path: Veo3에서 생성된 360도 영상 파일 경로
        output_dir: 분리된 프레임 저장 디렉토리 경로
    """
    print(f"\n[Frame Extraction] Extracting unique frames from video: {video_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    saved_idx = 0
    seen_hashes = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 해시로 변환
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        if frame_hash in seen_hashes:
            frame_idx += 1
            continue  # 이미 저장된 프레임이면 건너뜀

        seen_hashes.add(frame_hash)
        frame_filename = output_dir / f"frame_{saved_idx:04d}.png"
        cv2.imwrite(str(frame_filename), frame)
        print(f"  Saved unique frame: {frame_filename.name}")
        frame_idx += 1
        saved_idx += 1

    cap.release()
    print(f"✓ Extracted {saved_idx} unique frames to {output_dir}")



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
        kernel = np.ones((5, 5), np.uint8)
        largest_mask = cv2.erode(largest_mask, kernel, iterations=1)

        obj_only = np.zeros_like(img)
        obj_only[largest_mask == 1] = img[largest_mask == 1]
        alpha = (largest_mask * 255).astype(np.uint8)
        rgba_cv = cv2.cvtColor(obj_only, cv2.COLOR_BGR2BGRA)
        rgba_cv[:, :, 3] = alpha
        
        rgba_pil = Image.fromarray(cv2.cvtColor(rgba_cv, cv2.COLOR_BGRA2RGBA))

        out_path = MASKED_FRAMES_DIR / f"{frame_path.stem}_masked.png"
        rgba_pil.save(out_path)
        print(f"✓ Segmented and saved: {out_path.name}")
    print(f"Completed segmentation of {len(frames)} frames.")

# ============================================================
# Step 4: 의미론적으로 적합한 배경 위치 찾기 [IMPLEMENTED]
# ============================================================
def find_suitable_backgrounds(
    object_category: str,
    semantic_locations: List[str],
    broad_categories: List[str] = None,
    max_backgrounds: int = 5,
    similarity_threshold: float = 0.8,
    max_workers: int = 5
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
    
    Returns:
        배경 정보 리스트 [{"bg_image": Image, "segment_mask": np.ndarray, ...}]
    """
    print(f"\n[Step 4] Finding semantically suitable backgrounds...")
    print(f"Object category: {object_category}")
    print(f"Semantic locations: {semantic_locations}")
    if broad_categories:
        print(f"  Broad categories: {broad_categories}")
    
    matcher = SemanticMatcher(similarity_threshold=similarity_threshold)
    
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
def blend_object(obj_image: Image.Image, bg_image: Image.Image) -> Image.Image:
    """
    Perform Poisson (seamless) blending of an object image onto a background.
    The object is centered via mask centroid when available.
    """
    obj_rgba = obj_image.convert("RGBA")
    bg_rgba = bg_image.convert("RGBA")

    obj_cv = cv2.cvtColor(np.array(obj_rgba), cv2.COLOR_RGBA2BGRA)
    bg_cv = cv2.cvtColor(np.array(bg_rgba), cv2.COLOR_RGBA2BGRA)

    # Resize object only if larger than background
    bg_h, bg_w = bg_cv.shape[:2]
    obj_h, obj_w = obj_cv.shape[:2]
    if obj_w > bg_w or obj_h > bg_h:
        scale = min(bg_w / obj_w, bg_h / obj_h) * 0.9
        obj_cv = cv2.resize(obj_cv, (int(obj_w * scale), int(obj_h * scale)), interpolation=cv2.INTER_AREA)
        obj_h, obj_w = obj_cv.shape[:2]

    # Mask: alpha if present, else non-zero RGB
    if obj_cv.shape[2] == 4:
        mask = obj_cv[:, :, 3]
    else:
        mask = cv2.cvtColor(obj_cv, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Position: mask centroid if available, else center
    M = cv2.moments(mask_bin)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (cX, cY)
    else:
        center = (bg_w // 2, bg_h // 2)

    blended = cv2.seamlessClone(
        obj_cv[:, :, :3],           # src BGR
        bg_cv[:, :, :3],            # dst BGR
        mask_bin,                   # mask uint8
        center,                     # center point
        cv2.MIXED_CLONE
    )

    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))


# ============================================================
# Step 5: 자연스러운 합성
# ============================================================
def composite_naturally(
    objects: List[Tuple[Image.Image, Dict]],
    backgrounds: List[Dict],
    use_depth: bool = False,
    use_lighting: bool = False,
    overlay_scale: float = 1.0,
    depth_offset: float = 0.05
) -> List[Path]:
    """
    객체를 배경에 자연스럽게 합성
    
    Args:
        objects: [(누끼 이미지, 메타데이터)] 리스트
        backgrounds: 배경 정보 리스트
        use_depth: Depth 기반 occlusion 사용 여부
        use_lighting: Lighting 조정 사용 여부
        overlay_scale: 오버레이 스케일
        depth_offset: overlay를 앞으로 당길 오프셋 (기본 0.05)
    
    Returns:
        저장된 합성 이미지 경로 리스트
    
    구현 가이드:
        - [IMPLEMENTED] Depth 기반 occlusion 처리
        - [TODO] Lighting/shadow 조정
        - [TODO] Color harmonization
        - [TODO] Edge blending
    """
    
    print(f"\n[Step 5] Compositing objects naturally...")
    print(f"  Use depth: {use_depth}")
    print(f"  Use lighting: {use_lighting}")
    if use_lighting:
        print("[TODO] Lighting adjustment not implemented yet")
    
    output_paths = []
    
    for bg_idx, bg_info in enumerate(backgrounds):
        # Depth 계산 (use_depth=True일 때만)
        bg_depth_map = None
        averaged_depth_map = None
        if use_depth:
            print(f"\n  Computing depth for background {bg_idx}...")
            _, bg_depth_map = compute_depth(bg_info["bg_image"])
            print(f"    Depth range: [{bg_depth_map.min():.2f}, {bg_depth_map.max():.2f}]")
            
            # Segment별 평균 depth map 계산
            from segment import run_segmentation
            annotations, _ = run_segmentation(bg_info["bg_image"])
            
            # annotations를 Dict 형식으로 변환 (segmentation 키 필요)
            segments = [{"segmentation": mask} for mask, label in annotations]
            
            # 평균화된 depth map 생성
            averaged_depth_map = compute_segment_averaged_depth(bg_depth_map, segments)
            
            # 평균화된 depth map 시각화 및 저장
            # depth_vis_avg = (
            #     (averaged_depth_map - averaged_depth_map.min()) / 
            #     (averaged_depth_map.max() - averaged_depth_map.min() + 1e-8) * 255
            # ).astype(np.uint8)
            # depth_avg_filename = f"depth_averaged_bg{bg_idx}.png"
            # depth_avg_path = OUTPUT_DIR / depth_avg_filename
            # Image.fromarray(depth_vis_avg, mode='L').save(depth_avg_path)
            # print(f"Saved averaged depth map: {depth_avg_filename}")
        
        for obj_idx, (obj_image, obj_meta) in enumerate(objects):
            try:
                # 합성 (평균화된 depth map 사용)
                composite_img = composite_on_segment(
                    bg_info["bg_image"],
                    bg_info["segment_mask"],
                    obj_image,
                    base_scale=overlay_scale,
                    use_depth=use_depth,
                    bg_depth_map=averaged_depth_map if use_depth else None,
                    depth_offset=depth_offset
                )
                
                # Composite vs background diff -> object mask -> Poisson blend
                bg_rgb = bg_info["bg_image"].convert("RGB")
                comp_rgb = composite_img.convert("RGB")

                bg_cv = cv2.cvtColor(np.array(bg_rgb), cv2.COLOR_RGB2BGR)
                comp_cv = cv2.cvtColor(np.array(comp_rgb), cv2.COLOR_RGB2BGR)

                diff_gray = cv2.cvtColor(cv2.absdiff(comp_cv, bg_cv), cv2.COLOR_BGR2GRAY)
                _, obj_mask = cv2.threshold(diff_gray, 1, 255, cv2.THRESH_BINARY)

                if obj_mask.max() == 0:
                    blended_img = composite_img  # fallback if mask is empty
                else:
                    obj_only = cv2.bitwise_and(comp_cv, comp_cv, mask=obj_mask)
                    obj_only_rgba = cv2.cvtColor(obj_only, cv2.COLOR_BGR2BGRA)
                    obj_only_rgba[:, :, 3] = obj_mask
                    obj_cutout = Image.fromarray(cv2.cvtColor(obj_only_rgba, cv2.COLOR_BGRA2RGBA))
                    blended_img = blend_object(obj_cutout, bg_info["bg_image"])
                
                depth_suffix = "_depth" if use_depth else ""
                output_filename = (
                    f"composite_bg{bg_idx}_obj{obj_idx}_"
                    f"{bg_info['segment_label'].replace(' ', '_')}{depth_suffix}.png"
                )
                output_path = OUTPUT_DIR / output_filename
                blended_img.save(output_path)
                output_paths.append(output_path)
                
                print(f"??Saved: {output_filename}")
            
            except Exception as e:
                print(f"✗ Error: bg{bg_idx} + obj{obj_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"Generated {len(output_paths)} composite images")
    return output_paths




# ============================================================
# 전체 파이프라인 실행 함수
# ============================================================


def run_full_pipeline(
    object_category: str = "object",
    semantic_locations: List[str] = None,
    broad_categories: List[str] = None,
    num_views: int = 3,
    max_backgrounds: int = 5,
    similarity_threshold: float = 0.8,
    overlay_scale: float = 1.0,
    use_depth: bool = False,
    use_lighting: bool = False,
    max_workers: int = 5
) -> List[Path]:
    """
    전체 파이프라인을 순차적으로 실행
    
    Args:
        object_category: 객체 카테고리
        semantic_locations: 의미론적 위치 리스트
        broad_categories: 대분류 카테고리 리스트
        num_views: 선택할 뷰 개수
        max_backgrounds: 최대 배경 개수
        similarity_threshold: 유사도 임계값
        overlay_scale: 오버레이 스케일
        use_depth: Depth 사용 여부
        use_lighting: Lighting 사용 여부
        max_workers: 병렬 워커 수
    
    Returns:
        생성된 합성 이미지 경로 리스트
    """
    print("=" * 60)
    print("Few-shot Semantic Compositing Pipeline")
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
        max_workers=max_workers
    )
    
    if len(backgrounds) == 0:
        print("Error: No suitable backgrounds found!")
        return []
    
    objects = get_all_objects(
        original_input_dir=INPUT_DIR,
        masked_frames_dir=MASKED_FRAMES_DIR
    )

    # Step 5: 자연스러운 합성
    output_paths = composite_naturally(
        objects=objects,
        backgrounds=backgrounds,
        use_depth=use_depth,
        
        use_lighting=use_lighting,
        overlay_scale=overlay_scale,
        depth_offset=0.05  # overlay를 segment보다 5% 앞에 배치
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
    1. [TODO] 입력 이미지에서 가장 좋은 3장 선택
    2. [IMPLEMENTED] Veo3로 360도 뷰 영상 생성 (시뮬레이션)
    3. [IMPLEMENTED] 객체만 segmentation
    4. [IMPLEMENTED] 의미론적으로 적합한 배경 위치 찾기
    5. [TODO] 자연스러운 합성 (depth, lighting)
    """
    print("=== Few-shot Semantic Compositing Pipeline ===")
    print("[TODO] Step 1: Select 3 appropriate views from input images")
    print("[IMPLEMENTED] Step 2: Generate 360° video with Veo3 (Simulated)")
    print("[IMPLEMENTED] Step 3: Segment objects from video")
    print("[IMPLEMENTED] Step 4: Find semantically suitable backgrounds")
    print("[TODO] Step 5: Composite naturally with depth and lighting")
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
