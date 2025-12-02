"""
기존 생성된 비디오로 파이프라인 실행

기존 output/video에 있는 비디오를 사용해서
main.py의 파이프라인 함수들을 그대로 재사용:
- 프레임 추출 (frame_from_video)
- 객체 세그멘테이션 (segment_objects)
- 배경 찾기 (find_suitable_backgrounds)
- 합성 (composite_naturally)

영상 생성 단계는 건너뛰고 기존 비디오부터 시작
"""

import sys
from pathlib import Path
from typing import List, Tuple

# 상위 디렉토리의 main.py 함수들을 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    frame_from_video,
    segment_objects,
    find_suitable_backgrounds,
    composite_naturally,
    get_all_objects,
    BASE_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    MASKED_FRAMES_DIR,
    VIDEO_DIR
)


def process_existing_video(
    video_path: Path,
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
    기존 비디오로 파이프라인 실행 (비디오 생성 단계 제외)
    
    Args:
        video_path: 처리할 기존 비디오 경로
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
    print("Processing Existing Video Pipeline")
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print()
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return []
    
    # 프레임 추출
    frames_dir = OUTPUT_DIR / "frames"
    print(f"\n[Step 1] Extracting frames from video...")
    frame_from_video(video_path, frames_dir)
    
    # 객체 segmentation
    print(f"\n[Step 2] Segmenting objects...")
    segment_objects(frames_dir)
    
    # 적합한 배경 찾기
    print(f"\n[Step 3] Finding suitable backgrounds...")
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
    
    # 객체 로드
    print(f"\n[Step 4] Loading segmented objects...")
    objects = get_all_objects(
        original_input_dir=INPUT_DIR,
        masked_frames_dir=MASKED_FRAMES_DIR
    )
    print(f"Loaded {len(objects)} objects")
    
    # 자연스러운 합성
    print(f"\n[Step 5] Compositing naturally...")
    output_paths = composite_naturally(
        objects=objects,
        backgrounds=backgrounds,
        use_depth=use_depth,
        use_lighting=use_lighting,
        overlay_scale=overlay_scale,
        depth_offset=0.05
    )
    
    print("\n" + "=" * 60)
    print(f"Pipeline completed! Generated {len(output_paths)} images")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return output_paths


def main():
    """
    메인 엔트리 포인트
    output/video에서 가장 최근 비디오를 사용하거나
    특정 비디오 경로를 지정할 수 있음
    """
    # output/video에서 비디오 찾기
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    
    if not video_files:
        print(f"Error: No video files found in {VIDEO_DIR}")
        print("Please generate a video first using main.py")
        sys.exit(1)
    
    # 가장 최근 비디오 사용 (또는 원하는 비디오 선택)
    video_path = video_files[-1]  # 가장 최근 비디오
    
    # 특정 비디오를 사용하려면 아래 주석 해제하고 파일명 수정
    # video_path = VIDEO_DIR / "veo3_output_1764441918.mp4"
    
    print(f"Using video: {video_path.name}")
    print()
    
    # 파이프라인 실행
    results = process_existing_video(
        video_path=video_path,
        object_category="monkey doll",
        semantic_locations=["shelf", "bed", "couch", "table", "toy box"],
        broad_categories=["home", "indoor", "living room", "bedroom", "house interior"],
        max_backgrounds=5,
        similarity_threshold=0.8,
        overlay_scale=0.8,
        use_depth=True,
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
