"""
YOLO 데이터셋 생성 테스트 스크립트

배경 10개, 객체 2개씩 총 20장 생성
- 합성 이미지
- Segment visualization (객체 영역을 색으로 표시)
- YOLO annotation

결과 확인용으로 시각화 이미지도 함께 저장
"""

import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

# 상위 디렉토리의 main.py 함수들을 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    find_suitable_backgrounds,
    get_all_objects,
    blend_object,
    BASE_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    MASKED_FRAMES_DIR
)
from composite import composite_on_segment, compute_segment_averaged_depth
from segment import run_segmentation
from depth import compute_depth
from yolo_utils import save_yolo_annotation, create_yolo_dataset_structure
import random


def visualize_segment_mask(
    composite_img: Image.Image,
    visible_mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple = (0, 255, 0)  # Green
) -> Image.Image:
    """
    합성 이미지에 segment mask를 색상으로 오버레이
    
    Args:
        composite_img: 합성된 이미지
        visible_mask: Visible mask (H x W, bool)
        alpha: 투명도 (0-1)
        color: RGB 색상 (기본 녹색)
    
    Returns:
        Visualization 이미지
    """
    # 이미지를 numpy array로 변환
    img_array = np.array(composite_img.convert("RGB"))
    
    # 색상 오버레이 생성
    overlay = img_array.copy()
    overlay[visible_mask] = (
        img_array[visible_mask] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    return Image.fromarray(overlay)


def test_yolo_generation(
    object_category: str = "monkey doll",
    semantic_locations: List[str] = None,
    broad_categories: List[str] = None,
    max_backgrounds: int = 10,
    objects_per_bg: int = 2,
    similarity_threshold: float = 0.7,
    overlay_scale: float = 0.8,
    use_depth: bool = True,
    use_lighting: bool = True,
    max_workers: int = 5
) -> List[Path]:
    """
    YOLO 데이터셋 생성 테스트 (10 배경 × 2 객체 = 20 이미지)
    
    결과:
    - output/test_result/images/  (합성 이미지)
    - output/test_result/labels/  (YOLO annotations)
    - output/test_result/visualizations/  (segment 시각화)
    
    Args:
        object_category: 객체 카테고리
        semantic_locations: 의미론적 위치 리스트
        broad_categories: 대분류 카테고리 리스트
        max_backgrounds: 최대 배경 개수 (기본 3)
        objects_per_bg: 배경당 객체 개수 (기본 4)
        similarity_threshold: 유사도 임계값
        overlay_scale: 오버레이 스케일
        use_depth: Depth 사용 여부
        use_lighting: Lighting 사용 여부
        max_workers: 병렬 워커 수
    
    Returns:
        생성된 이미지 경로 리스트
    """
    print("=" * 60)
    print("YOLO Dataset Generation Test")
    print(f"Target: {max_backgrounds} backgrounds × {objects_per_bg} objects")
    print("=" * 60)
    print("Skipping: Frame extraction, Segmentation")
    print("Running: Background finding, Compositing, Visualization")
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
    
    # 테스트용 디렉토리 구조 생성
    test_base = OUTPUT_DIR / "test_result"
    images_dir = test_base / "images"
    labels_dir = test_base / "labels"
    viz_dir = test_base / "visualizations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Test directories created at: {test_base}")
    
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
    
    print(f"Found {len(backgrounds)} backgrounds")
    
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
    
    # Step 3: 합성 및 시각화
    print(f"\n[Step 3] Compositing and visualizing...")
    output_paths = []
    
    for bg_idx, bg_info in enumerate(backgrounds):
        print(f"\n  Background {bg_idx+1}/{len(backgrounds)}")
        
        # 랜덤하게 객체 선택
        selected_indices = random.sample(range(len(objects)), min(objects_per_bg, len(objects)))
        selected_objects = [(objects[i][0], objects[i][1], i) for i in selected_indices]
        
        # Depth 계산
        bg_depth_map = None
        averaged_depth_map = None
        if use_depth:
            print(f"    Computing depth...")
            _, bg_depth_map = compute_depth(bg_info["bg_image"])
            
            # Segment별 평균 depth map 계산
            annotations, _ = run_segmentation(bg_info["bg_image"])
            segments = [{"segmentation": mask} for mask, label in annotations]
            averaged_depth_map = compute_segment_averaged_depth(bg_depth_map, segments)
        
        for obj_idx, (obj_image, obj_meta, original_idx) in enumerate(selected_objects):
            try:
                # 크기: 긴 쪽을 300-350px로 통일 (일관된 크기 보장)
                max_dim = max(obj_image.size)
                target_size = random.uniform(350, 430)
                scale = target_size / max_dim
                rotation = random.uniform(-15, 15)  # -15° ~ +15°
                
                # 50% 확률로 수평 반전
                if random.random() < 0.5:
                    obj_image = obj_image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # 합성 (visible_mask 포함)
                composite_img, occlusion_ratio, visible_mask = composite_on_segment(
                    bg_info["bg_image"],
                    bg_info["segment_mask"],
                    obj_image,
                    base_scale=scale,
                    use_depth=use_depth,
                    bg_depth_map=averaged_depth_map if use_depth else None,
                    depth_offset=0.05,
                    rotation_angle=rotation
                )
                
                # 가려짐 체크
                if occlusion_ratio >= 0.15:
                    print(f"    ⊗ Object {obj_idx+1}: {occlusion_ratio:.1%} occluded, skipping")
                    continue
                
                print(f"    ✓ Object {obj_idx+1}: {occlusion_ratio:.1%} occluded")
                
                # Composite vs background diff -> object mask
                bg_rgb = bg_info["bg_image"].convert("RGB")
                comp_rgb = composite_img.convert("RGB")

                bg_cv = cv2.cvtColor(np.array(bg_rgb), cv2.COLOR_RGB2BGR)
                comp_cv = cv2.cvtColor(np.array(comp_rgb), cv2.COLOR_RGB2BGR)

                diff_gray = cv2.cvtColor(cv2.absdiff(comp_cv, bg_cv), cv2.COLOR_BGR2GRAY)
                _, obj_mask = cv2.threshold(diff_gray, 1, 255, cv2.THRESH_BINARY)

                if obj_mask.max() == 0:
                    print(f"    ⚠ Object {obj_idx+1}: No difference detected, skipping")
                    continue
                
                # 객체 추출 및 Option G 블렌딩
                obj_only = cv2.bitwise_and(comp_cv, comp_cv, mask=obj_mask)
                obj_only_rgba = cv2.cvtColor(obj_only, cv2.COLOR_BGR2BGRA)
                obj_only_rgba[:, :, 3] = obj_mask
                obj_cutout = Image.fromarray(cv2.cvtColor(obj_only_rgba, cv2.COLOR_BGRA2RGBA))
                blended_img = blend_object(obj_cutout, bg_info["bg_image"])
                
                # 파일명
                output_filename = (
                    f"test_bg{bg_idx:02d}_obj{original_idx:03d}_"
                    f"{bg_info['segment_label'].replace(' ', '_')}.png"
                )
                
                # 1. 합성 이미지 저장
                image_path = images_dir / output_filename
                blended_img.save(image_path)
                
                # 2. YOLO annotation 저장 (obj_mask 사용 - 실제 최종 객체 마스크)
                label_path = labels_dir / output_filename.replace('.png', '.txt')
                img_width, img_height = blended_img.size
                annotation_saved = save_yolo_annotation(
                    obj_mask,  # obj_mask로 변경 (visible_mask 대신)
                    img_width,
                    img_height,
                    label_path,
                    class_id=0
                )
                
                if not annotation_saved:
                    print(f"    ⚠ Failed to save annotation")
                    continue
                
                # 3. Segment 시각화 저장 (obj_mask 사용 - annotation과 동일)
                # obj_mask를 boolean으로 변환
                obj_mask_bool = obj_mask > 0
                viz_img = visualize_segment_mask(
                    blended_img,
                    obj_mask_bool,  # visible_mask 대신 obj_mask 사용
                    alpha=0.4,
                    color=(0, 255, 0)  # Green overlay
                )
                viz_path = viz_dir / output_filename
                viz_img.save(viz_path)
                
                # 4. Averaged depth map 저장 (시각화)
                if use_depth and averaged_depth_map is not None:
                    depth_normalized = ((averaged_depth_map - averaged_depth_map.min()) / 
                                      (averaged_depth_map.max() - averaged_depth_map.min() + 1e-8) * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                    depth_pil = Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
                    depth_path = viz_dir / output_filename.replace('.png', '_depth.png')
                    depth_pil.save(depth_path)
                
                output_paths.append(image_path)
                
                print(f"      Saved: {output_filename} (Total: {len(output_paths)})")
                print(f"        - Image: {image_path.name}")
                print(f"        - Label: {label_path.name}")
                print(f"        - Viz: {viz_path.name}")
                if use_depth:
                    print(f"        - Depth: {output_filename.replace('.png', '_depth.png')}")
            
            except Exception as e:
                print(f"    ✗ Error: obj{obj_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "=" * 60)
    print(f"Test completed! Generated {len(output_paths)} images")
    print(f"\nResults:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"  Visualizations: {viz_dir}")
    print("=" * 60)
    
    return output_paths


def main():
    """
    메인 엔트리 포인트
    """
    print("Test configuration:")
    print(f"  Input: {INPUT_DIR}")
    print(f"  Masked frames: {MASKED_FRAMES_DIR}")
    print()
    
    # 테스트 실행
    results = test_yolo_generation(
        object_category="monkey_doll",
        semantic_locations=["shelf", "bed", "couch", "table", "toy box"],
        broad_categories=["home", "indoor", "living room", "bedroom", "house interior"],
        max_backgrounds=10,
        objects_per_bg=2,
        similarity_threshold=0.7,
        overlay_scale=0.8,
        use_depth=True,
        use_lighting=True,
        max_workers=5
    )
    
    if results:
        print(f"\n✓ Successfully generated {len(results)} test images!")
        print("\nCheck the output folders:")
        print("  - output/test_result/images/  (합성 이미지)")
        print("  - output/test_result/labels/  (YOLO annotations)")
        print("  - output/test_result/visualizations/  (segment 시각화)")
    else:
        print("\n✗ No images generated. Check the logs above for errors.")


if __name__ == "__main__":
    main()
