"""
Few-shot Semantic Compositing Pipeline - Main Entry Point

전체 파이프라인:
1. [TODO] 입력 이미지에서 뷰가 좋은 3장 선택
2. [TODO] Veo3로 프레임 생성
3. [TODO] 객체만 segmentation
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

from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple

from semantic_matcher import SemanticMatcher
from composite import composite_on_segment


# 전역 설정
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "train"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output" / "compositing_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Step 1: 입력 이미지에서 뷰가 좋은 3장 선택
# ============================================================
def select_best_views(input_images: List[Path], num_views: int = 3) -> List[Path]:
    """
    [TODO] 입력 이미지에서 뷰가 좋은 상위 N장을 선택
    
    Args:
        input_images: 입력 이미지 경로 리스트
        num_views: 선택할 이미지 개수
    
    Returns:
        선택된 이미지 경로 리스트
    
    구현 가이드:
        - 이미지 품질 평가 (블러, 노이즈, 해상도)
        - 객체 가시성 평가 (occlusion, 각도)
        - 다양성 평가 (서로 다른 각도)
    """
    print(f"\n[Step 1] Selecting best {num_views} views from {len(input_images)} images...")
    print("[TODO] Implement view selection algorithm")
    
    # 임시: 처음 N개 반환
    selected = input_images[:num_views]
    print(f"  Selected: {[img.name for img in selected]}")
    return selected


# ============================================================
# Step 2: Veo3로 프레임 생성
# ============================================================
def generate_frames_with_veo3(selected_images: List[Path]) -> List[Image.Image]:
    """
    [TODO] Veo3 모델을 사용해 선택된 이미지로부터 프레임 생성
    
    Args:
        selected_images: 선택된 이미지 경로 리스트
    
    Returns:
        생성된 프레임 이미지 리스트
    
    구현 가이드:
        - Veo3 API 호출
        - 프레임 생성 파라미터 설정
        - 생성된 프레임 후처리
    """
    print(f"\n[Step 2] Generating frames with Veo3...")
    print("[TODO] Implement Veo3 frame generation")
    
    # 임시: 원본 이미지를 PIL로 로드해서 반환
    frames = [Image.open(img).convert("RGB") for img in selected_images]
    print(f"  Generated {len(frames)} frames")
    return frames


# ============================================================
# Step 3: 객체만 segmentation
# ============================================================
def segment_objects(frames: List[Image.Image]) -> List[Tuple[Image.Image, Dict]]:
    """
    [TODO] 생성된 프레임에서 객체만 segmentation
    
    임시: input 폴더의 누끼 이미지(PNG)를 로드
    
    Args:
        frames: 프레임 이미지 리스트 (사용 안 함, TODO)
    
    Returns:
        [(누끼 이미지, 메타데이터)] 리스트
    """
    print(f"\n[Step 3] Loading overlay images from input...")
    
    # input 폴더에서 PNG 파일 찾기
    overlay_files = sorted(INPUT_DIR.glob("*.png"))
    
    results = []
    for idx, overlay_path in enumerate(overlay_files):
        # 누끼 이미지 로드 (RGBA)
        overlay_img = Image.open(overlay_path).convert("RGBA")
        
        metadata = {
            "frame_idx": idx,
            "category": "object",
            "filename": overlay_path.name
        }
        
        results.append((overlay_img, metadata))
        print(f"  Loaded: {overlay_path.name}")
    
    if not results:
        print("  Warning: No PNG files found in input folder")
    
    return results


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
    
    # Step 2: Veo3 프레임 생성
    frames = generate_frames_with_veo3(selected_views)
    
    # Step 3: 객체 segmentation
    objects = segment_objects(frames)
    
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
    1. [TODO] 입력 이미지 중 뷰가 좋은 3장 선택
    2. [TODO] Veo3로 프레임 생성
    3. [TODO] 객체만 segmentation
    4. [IMPLEMENTED] 의미론적으로 적합한 배경 위치 찾기
    5. [TODO] 자연스러운 합성 (depth, lighting)
    """
    print("=== Few-shot Semantic Compositing Pipeline ===")
    print("[TODO] Step 1: Select 3 best views from input images")
    print("[TODO] Step 2: Generate frames with Veo3")
    print("[TODO] Step 3: Segment objects only")
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
