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

import random
import time
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
import os
import base64
import cv2
from ultralytics import SAM

from google import genai
from google.genai import types

from semantic_matcher import SemanticMatcher
from composite import composite_on_segment


# 전역 설정
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "train"
INPUT_DIR = BASE_DIR / "input"
VIDEO_DIR = BASE_DIR / "output" / "video"
OUTPUT_DIR = BASE_DIR / "output" / "dataset"
MASKED_FRAMES_DIR = BASE_DIR / "output" / "dataset" / "masked_frames" # New
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
MASKED_FRAMES_DIR.mkdir(parents=True, exist_ok=True) # New
# Generative AI 클라이언트 초기화 (인증은 환경 변수 또는 genai.configure()를 통해 외부에서 처리된다고 가정)
try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
    client = genai.Client(
        vertexai=True,
        project="gen-lang-client-0127629302",
        location="us-central1"
    )
except Exception as e:
    print(f"Error initializing Google Generative AI client: {e}")
    client = None # 클라이언트 초기화 실패 시 None으로 설정

# Global variable for the SAM model to ensure it's loaded only once.
SAM_MODEL = None

def get_sam_model():
    """Loads and returns the SAM model, caching it globally."""
    global SAM_MODEL
    if SAM_MODEL is None:
        print("Loading SAM model...")
        SAM_MODEL = SAM("sam2_l.pt")
        print("SAM model loaded.")
    return SAM_MODEL

# ============================================================
# Step 1: 입력 이미지에서 랜덤으로 3장 선택
# ============================================================
def select_best_views(input_images: List[Path], num_views: int = 3) -> List[Path]:
    """
    입력 이미지에서 N장을 랜덤으로 선택
    
    [TODO] 추후 이미지 품질, 객체 가시성, 다양성을 고려한 뷰 선택 알고리즘으로 교체
    
    Args:
        input_images: 입력 이미지 경로 리스트
        num_views: 선택할 이미지 개수
    
    Returns:
        선택된 이미지 경로 리스트
    """
    print(f"\n[Step 1] Selecting {num_views} random views from {len(input_images)} images...")
    
    if len(input_images) <= num_views:
        print("  Number of images is less than or equal to num_views. Using all images.")
        selected = input_images
    else:
        selected = random.sample(input_images, num_views)
        
    print(f"  Selected: {[img.name for img in selected]}")
    return selected


# ============================================================
# Step 2: Veo3로 360도 뷰 영상 생성
# ============================================================
def generate_360_video_with_veo3(selected_images: List[Path]) -> Path:
    """
    Veo3 API를 호출하여 360도 뷰 영상 생성 (GCS 업로드 없이 로컬 이미지로 직접 호출)

    Args:
        selected_images: 선택된 로컬 이미지 경로 리스트

    Returns:
        생성된 360도 영상 파일 경로 (로컬 저장)
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

    selected_images.insert(0, INPUT_DIR / "input.png")
    print("  Calling Veo API... This may take a while...")
    reference_images = [types.VideoGenerationReferenceImage(
                        image=types.Image.from_file(location=str(file)),
                        reference_type="asset",
                    ) for file in selected_images]
    
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
            print("  Waiting for video generation...")
            time.sleep(15)
            operation = client.operations.get(operation)

        print("  ✓ Video generation completed!")
        print(operation.response)
        print(operation.result)
        if operation.response and operation.result.generated_videos:
            video_bytes = operation.result.generated_videos[0].video.video_bytes

            local_path = VIDEO_DIR / f"veo3_output_{int(time.time())}.mp4"
            with open(local_path, "wb") as f:
                f.write(video_bytes)

            print(f"  ✓ Saved generated video locally: {local_path}")
            return local_path


    except Exception as e:
        print(f"  Veo API Exception: {e}")
        raise e


def frame_from_video(video_path: Path, output_dir: Path) -> None:
    """
    Veo3로 생성된 360도 영상에서 유일한 프레임만 분리 및 저장
    
    Args:
        video_path: Veo3에서 생성된 360도 영상 파일 경로
        output_dir: 분리된 프레임 저장 디렉토리 경로
    """
    import cv2
    import numpy as np
    import hashlib

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
    print(f"  ✓ Extracted {saved_idx} unique frames to {output_dir}")



# ============================================================
# Step 3: 객체만 segmentation
# ============================================================
def segment_objects(frame_dir: Path) -> None: # Modified return type
    """
    Segments the main object from each frame in a directory and saves them to MASKED_FRAMES_DIR.

    Args:
        frame_dir: Directory containing the image frames to segment.
    """
    print(f"\n[Step 3] Segmenting objects from frames in {frame_dir} and saving to {MASKED_FRAMES_DIR}...")
    
    model = get_sam_model()
    
    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    if not frames:
        print(f"  Warning: No frames found in {frame_dir} to segment.")
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
        print(f"  ✓ Segmented and saved: {out_path.name}")
    print(f"  Completed segmentation of {len(frames)} frames.")


def get_all_objects() -> List[Tuple[Image.Image, Dict]]:
    """
    Loads all segmented object images from MASKED_FRAMES_DIR and original input
    images from INPUT_DIR, returning them with metadata.

    Returns:
        A list of tuples, where each tuple contains an object image (PIL RGBA)
        and its metadata.
    """
    objects = []
    
    # Load from MASKED_FRAMES_DIR (segmented video frames)
    print(f"\n[Step 3c] Loading segmented frames from {MASKED_FRAMES_DIR}...")
    masked_files = sorted([p for p in MASKED_FRAMES_DIR.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    for idx, file_path in enumerate(masked_files):
        try:
            img = Image.open(file_path).convert("RGBA")
            metadata = {
                "frame_idx": idx,
                "category": "object", # Assuming all are objects
                "filename": file_path.name
            }
            objects.append((img, metadata))
            print(f"  Loaded masked frame: {file_path.name}")
        except Exception as e:
            print(f"  [WARN] Could not load masked frame {file_path.name}: {e}")

    # Load from INPUT_DIR (original input images, assumed to be pre-masked or objects)
    print(f"\n[Step 3d] Loading original input images from {INPUT_DIR}...")
    original_files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    start_idx = len(objects) # Continue indexing from where masked_files left off

    for i, file_path in enumerate(original_files):
        try:
            img = Image.open(file_path).convert("RGBA")
            metadata = {
                "frame_idx": start_idx + i,
                "category": "object", # Assuming all are objects
                "filename": file_path.name
            }
            objects.append((img, metadata))
            print(f"  Loaded original input image: {file_path.name}")
        except Exception as e:
            print(f"  [WARN] Could not load original input image {file_path.name}: {e}")
            
    if not objects:
        print("  Warning: No objects were loaded from any directory.")
        
    return objects


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
    [IMPLEMENTED] 의미론적으로 적합한 배경 위치 찾기
    
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
    print(f"  Object category: {object_category}")
    print(f"  Semantic locations: {semantic_locations}")
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
    
    print(f"  Found {len(suitable_backgrounds)} suitable backgrounds")
    return suitable_backgrounds


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
    from depth import compute_depth
    from composite import compute_segment_averaged_depth
    
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
            depth_vis_avg = ((averaged_depth_map - averaged_depth_map.min()) / 
                           (averaged_depth_map.max() - averaged_depth_map.min() + 1e-8) * 255).astype(np.uint8)
            depth_avg_filename = f"depth_averaged_bg{bg_idx}.png"
            depth_avg_path = OUTPUT_DIR / depth_avg_filename
            Image.fromarray(depth_vis_avg, mode='L').save(depth_avg_path)
            print(f"    Saved averaged depth map: {depth_avg_filename}")
        
        for obj_idx, (obj_image, obj_meta) in enumerate(objects):
            try:
                # 합성 (평균화된 depth map 사용)
                result = composite_on_segment(
                    bg_info["bg_image"],
                    bg_info["segment_mask"],
                    obj_image,
                    base_scale=overlay_scale,
                    use_depth=use_depth,
                    bg_depth_map=averaged_depth_map if use_depth else None,
                    depth_offset=depth_offset
                )
                
                # 저장
                depth_suffix = "_depth" if use_depth else ""
                output_filename = f"composite_bg{bg_idx}_obj{obj_idx}_{bg_info['segment_label'].replace(' ', '_')}{depth_suffix}.png"
                output_path = OUTPUT_DIR / output_filename
                result.save(output_path)
                output_paths.append(output_path)
                
                print(f"  ✓ Saved: {output_filename}")
            
            except Exception as e:
                print(f"  ✗ Error: bg{bg_idx} + obj{obj_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"  Generated {len(output_paths)} composite images")
    return output_paths


# ============================================================
# 전체 파이프라인 실행 함수
# ============================================================
def run_full_pipeline(
    input_dir: Path = INPUT_DIR,
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
        input_dir: 입력 이미지 디렉토리
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
    input_images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    if len(input_images) == 0:
        print(f"Error: No images found in {input_dir}")
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
    
    objects = get_all_objects()

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
    1. [IMPLEMENTED] 입력 이미지 중 랜덤으로 3장 선택
    2. [IMPLEMENTED] Veo3로 360도 뷰 영상 생성 (시뮬레이션)
    3. [IMPLEMENTED] 객체만 segmentation
    4. [IMPLEMENTED] 의미론적으로 적합한 배경 위치 찾기
    5. [TODO] 자연스러운 합성 (depth, lighting)
    """
    print("=== Few-shot Semantic Compositing Pipeline ===")
    print("[IMPLEMENTED] Step 1: Select 3 random views from input images")
    print("[IMPLEMENTED] Step 2: Generate 360° video with Veo3 (Simulated)")
    print("[IMPLEMENTED] Step 3: Segment objects from video")
    print("[IMPLEMENTED] Step 4: Find semantically suitable backgrounds")
    print("[TODO] Step 5: Composite naturally with depth and lighting")
    print()
    
    # 전체 파이프라인 실행
    results = run_full_pipeline(
        input_dir=INPUT_DIR,
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
