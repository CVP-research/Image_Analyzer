"""
이미 segmentation이 완료된 데이터로 파이프라인 실행

Segmentation을 건너뛰고:
- 배경 찾기 (find_suitable_backgrounds)
- 합성 (composite_naturally)

만 실행. 프레임 추출과 segmentation은 이미 완료되었다고 가정.
"""

import sys
from pathlib import Path
from typing import List

# 상위 디렉토리의 main.py 함수들을 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    find_suitable_backgrounds,
    composite_naturally,
    get_all_objects,
    BASE_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    MASKED_FRAMES_DIR
)


def process_from_backgrounds(
    object_category: str = "object",
    semantic_locations: List[str] = None,
    broad_categories: List[str] = None,
    max_backgrounds: int = 5,
    similarity_threshold: float = 0.8,
    overlay_scale: float = 0.8,
    use_depth: bool = True,
    use_lighting: bool = False,
    max_workers: int = 5
) -> List[Path]:
    """
    이미 segmentation이 완료된 데이터로 파이프라인 실행
    (배경 찾기 → 합성만 수행)
    
    Args:
        object_category: 객체 카테고리
        semantic_locations: 의미론적 위치 리스트
        broad_categories: 대분류 카테고리 리스트
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
    print("Processing from Pre-segmented Data")
    print("=" * 60)
    print("Skipping: Frame extraction, Segmentation")
    print("Running: Background finding, Compositing")
    print()
    
    # 데이터 존재 확인
    if not MASKED_FRAMES_DIR.exists():
        print(f"Error: MASKED_FRAMES_DIR not found: {MASKED_FRAMES_DIR}")
        print("Please run segmentation first!")
        return []
    
    masked_files = list(MASKED_FRAMES_DIR.glob("*.png")) + list(MASKED_FRAMES_DIR.glob("*.jpg"))
    if len(masked_files) == 0:
        print(f"Error: No masked frames found in {MASKED_FRAMES_DIR}")
        print("Please run segmentation first!")
        return []
    
    print(f"Found {len(masked_files)} segmented frames")
    
    # Step 1: 적합한 배경 찾기
    print(f"\n[Step 1] Finding suitable backgrounds...")
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
    
    # Step 2: 객체 로드
    print(f"\n[Step 2] Loading segmented objects...")
    objects = get_all_objects(
        original_input_dir=INPUT_DIR,
        masked_frames_dir=MASKED_FRAMES_DIR
    )
    print(f"Loaded {len(objects)} objects")
    
    if len(objects) == 0:
        print("Warning: No objects loaded!")
        return []
    
    # Step 3: 자연스러운 합성
    print(f"\n[Step 3] Compositing naturally...")
    output_paths = composite_naturally(
        objects=objects,
        backgrounds=backgrounds,
        use_depth=use_depth,
        use_lighting=use_lighting,
        overlay_scale=overlay_scale,
        depth_offset=0.05,
        objects_per_bg=2,  # 배경당 2개 객체만 선택
        occlusion_threshold=0.3  # 30% 이상 가려지면 스킵
    )
    
    print("\n" + "=" * 60)
    print(f"Pipeline completed! Generated {len(output_paths)} images")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return output_paths


def main():
    """
    메인 엔트리 포인트
    이미 준비된 segmentation 데이터로 작업
    """
    print("Using pre-segmented data from:")
    print(f"  Input: {INPUT_DIR}")
    print(f"  Masked frames: {MASKED_FRAMES_DIR}")
    print()
    
    # 파이프라인 실행
    results = process_from_backgrounds(
        object_category="monkey doll",
        semantic_locations=["shelf", "bed", "couch", "table", "toy box"],
        broad_categories=["home", "indoor", "living room", "bedroom", "house interior"],
        max_backgrounds=5,
        similarity_threshold=0.8,
        overlay_scale=0.8,
        use_depth=True,
        use_lighting=True,  # Option G 활성화
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
