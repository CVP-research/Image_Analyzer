"""
DoveNet을 이용한 자연스러운 합성 파이프라인

1. 배경 찾기
2. 간단한 depth 기반 합성 (색감 조정 없이)
3. DoveNet 데이터셋 형식으로 저장
4. DoveNet 실행하여 최종 harmonization
"""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 상위 디렉토리의 main.py 함수들을 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    find_suitable_backgrounds,
    get_all_objects,
    BASE_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    MASKED_FRAMES_DIR
)
from composite import composite_on_segment, compute_segment_averaged_depth
from depth import compute_depth
from segment import run_segmentation
from yolo_utils import save_yolo_annotation, create_yolo_dataset_structure


# DoveNet 경로 설정
DOVENET_ROOT = Path("/home/rocknroll1397/Image-Harmonization-Dataset-iHarmony4/DoveNet")
DOVENET_VENV_PYTHON = Path("/home/rocknroll1397/Image-Harmonization-Dataset-iHarmony4/venv/bin/python")
DOVENET_DATA_DIR = DOVENET_ROOT / "data" / "image_analyzer"


def simple_composite_with_depth(
    bg_image: Image.Image,
    segment_mask: Image.Image,
    obj_image: Image.Image,
    base_scale: float = 1.0,
    use_depth: bool = True,
    bg_depth_map: np.ndarray = None,
    depth_offset: float = 0.05,
    rotation_angle: float = 0.0
) -> Tuple[Image.Image, Image.Image, float]:
    """
    색감 조정 없이 depth만 고려한 간단한 합성
    
    Returns:
        (합성 이미지, 객체 마스크, occlusion_ratio)
    """
    # 기본 합성 (depth 기반 원근법만 적용)
    composite_img, occlusion_ratio, placed_obj_mask = composite_on_segment(
        bg_image, segment_mask, obj_image,
        base_scale=base_scale,
        use_depth=use_depth,
        bg_depth_map=bg_depth_map,
        depth_offset=depth_offset,
        rotation_angle=rotation_angle
    )
    
    # 객체 마스크 추출 (배경과 합성 이미지의 차이)
    bg_rgb = bg_image.convert("RGB")
    comp_rgb = composite_img.convert("RGB")
    bg_cv = cv2.cvtColor(np.array(bg_rgb), cv2.COLOR_RGB2BGR)
    comp_cv = cv2.cvtColor(np.array(comp_rgb), cv2.COLOR_RGB2BGR)
    
    diff_gray = cv2.cvtColor(cv2.absdiff(comp_cv, bg_cv), cv2.COLOR_BGR2GRAY)
    _, obj_mask = cv2.threshold(diff_gray, 1, 255, cv2.THRESH_BINARY)
    
    # PIL Image로 변환
    obj_mask_pil = Image.fromarray(obj_mask, mode='L')
    
    return composite_img, obj_mask_pil, occlusion_ratio


def prepare_dovenet_dataset(
    objects: List[Tuple[Image.Image, Dict]],
    backgrounds: List[Dict],
    use_depth: bool = True,
    depth_offset: float = 0.05,
    occlusion_threshold: float = 0.3,
    max_composites: int = 100
) -> Path:
    """
    DoveNet 데이터셋 형식으로 준비
    
    폴더 구조:
    dovenet/data/image_analyzer/
        composite_images/
        masks/
        real_images/
        IHD_test.txt
    
    Returns:
        준비된 데이터셋 디렉토리 경로
    """
    # 디렉토리 생성
    composite_dir = DOVENET_DATA_DIR / "composite_images"
    masks_dir = DOVENET_DATA_DIR / "masks"
    real_dir = DOVENET_DATA_DIR / "real_images"
    
    # 기존 데이터 삭제 후 새로 생성
    if DOVENET_DATA_DIR.exists():
        shutil.rmtree(DOVENET_DATA_DIR)
    
    composite_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[DoveNet Preparation] Creating dataset at {DOVENET_DATA_DIR}")
    print(f"  Max composites: {max_composites}")
    
    image_list = []
    generated_count = 0
    
    pbar = tqdm(total=max_composites, desc="Preparing DoveNet data", unit="img")
    
    for bg_idx, bg_info in enumerate(backgrounds):
        if generated_count >= max_composites:
            break
        
        # Depth 맵 계산 (옵션)
        bg_depth_map = None
        averaged_depth_map = None
        if use_depth:
            _, bg_depth_map = compute_depth(bg_info["bg_image"])
            annotations, _ = run_segmentation(bg_info["bg_image"])
            segments = [{"segmentation": mask} for mask, label in annotations]
            averaged_depth_map = compute_segment_averaged_depth(bg_depth_map, segments)
        
        # 각 배경당 1-2개 객체 배치
        num_objects = min(2, len(objects))
        selected_objects = random.sample(objects, num_objects)
        
        for obj_image, obj_meta in selected_objects:
            if generated_count >= max_composites:
                break
            
            try:
                # 데이터 증강
                obj_img_copy = obj_image.copy()
                max_dim = max(obj_img_copy.size)
                target_size = random.uniform(350, 430)
                scale = target_size / max_dim
                rotation = random.uniform(-15, 15)
                if random.random() < 0.5:
                    obj_img_copy = obj_img_copy.transpose(Image.FLIP_LEFT_RIGHT)
                
                # 간단한 합성 (depth만 적용, 색감 조정 없음)
                composite_img, obj_mask, occlusion_ratio = simple_composite_with_depth(
                    bg_info["bg_image"],
                    bg_info["segment_mask"],
                    obj_img_copy,
                    base_scale=scale,
                    use_depth=use_depth,
                    bg_depth_map=averaged_depth_map,
                    depth_offset=depth_offset,
                    rotation_angle=rotation
                )
                
                # Occlusion 체크
                if occlusion_ratio >= occlusion_threshold:
                    continue
                
                # 마스크가 비어있는지 확인
                mask_array = np.array(obj_mask)
                if mask_array.max() == 0:
                    continue
                
                # 파일명 생성
                filename = f"composite_{generated_count:04d}"
                
                # 저장
                composite_img.save(composite_dir / f"{filename}.jpg")
                obj_mask.save(masks_dir / f"{filename}.png")
                bg_info["bg_image"].save(real_dir / f"{filename}.jpg")
                
                # 리스트에 추가
                image_list.append(f"composite_images/{filename}.jpg")
                
                generated_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing bg{bg_idx}: {e}")
                continue
    
    pbar.close()
    
    # IHD_test.txt 생성
    test_list_file = DOVENET_DATA_DIR / "IHD_test.txt"
    with open(test_list_file, 'w') as f:
        f.write('\n'.join(image_list))
    
    print(f"\n✓ DoveNet dataset prepared: {generated_count} images")
    print(f"  Composite images: {composite_dir}")
    print(f"  Masks: {masks_dir}")
    print(f"  Real images: {real_dir}")
    print(f"  Test list: {test_list_file}")
    
    return DOVENET_DATA_DIR


def run_dovenet_harmonization(
    dataset_root: Path,
    experiment_name: str = "image_analyzer_harmonization",
    num_test: int = None
) -> Path:
    """
    DoveNet 실행하여 harmonization 수행
    
    Returns:
        결과 이미지 디렉토리 경로
    """
    print(f"\n[DoveNet Harmonization] Running DoveNet...")
    print(f"  Dataset: {dataset_root}")
    print(f"  Experiment: {experiment_name}")
    
    # 생성된 이미지 개수 확인
    composite_images = list((dataset_root / "composite_images").glob("*.jpg"))
    if num_test is None:
        num_test = len(composite_images)
    
    print(f"  Processing {num_test} images...")
    
    # DoveNet 명령어 구성
    cmd = [
        str(DOVENET_VENV_PYTHON),
        "test.py",
        "--dataset_root", str(dataset_root),
        "--name", experiment_name,
        "--model", "dovenet",
        "--dataset_mode", "iharmony4",
        "--netG", "s2ad",
        "--is_train", "0",
        "--norm", "batch",
        "--no_flip",
        "--preprocess", "none",
        "--num_test", str(num_test)
    ]
    
    # DoveNet 디렉토리에서 실행
    try:
        result = subprocess.run(
            cmd,
            cwd=str(DOVENET_ROOT),
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"\n✓ DoveNet harmonization completed!")
        
        # 결과 디렉토리 찾기
        results_dir = DOVENET_ROOT / "results" / experiment_name / "test_latest" / "images"
        
        if results_dir.exists():
            harmonized_images = list(results_dir.glob("*.jpg"))
            print(f"  Results: {results_dir}")
            print(f"  Generated {len(harmonized_images)} harmonized images")
            return results_dir
        else:
            print(f"  Warning: Results directory not found: {results_dir}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ DoveNet execution failed!")
        print(f"  Error: {e}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return None


def convert_to_yolo_format(
    harmonized_dir: Path,
    dovenet_data_dir: Path,
    class_name: str = "object"
) -> List[Path]:
    """
    DoveNet 결과를 YOLO 데이터셋 형식으로 변환
    
    Returns:
        생성된 YOLO 이미지 경로 리스트
    """
    print(f"\n[YOLO Conversion] Converting harmonized images to YOLO format...")
    
    # YOLO 데이터셋 구조 생성
    yolo_base = OUTPUT_DIR / "dovenet_result"
    images_train, images_val, labels_train, labels_val = create_yolo_dataset_structure(
        yolo_base, [class_name]
    )
    
    # harmonized 이미지 및 마스크 처리
    harmonized_images = sorted(harmonized_dir.glob("*.jpg"))
    masks_dir = dovenet_data_dir / "masks"
    
    output_paths = []
    val_ratio = 0.2
    
    pbar = tqdm(harmonized_images, desc="Converting to YOLO", unit="img")
    
    for idx, harm_img_path in enumerate(pbar):
        try:
            # 파일명에서 인덱스 추출 (composite_0000.jpg -> 0000)
            base_name = harm_img_path.stem  # composite_0000
            
            # train/val 분리
            is_train = (idx % 5) != 0  # 80/20 split
            image_dir = images_train if is_train else images_val
            label_dir = labels_train if is_train else labels_val
            split_name = "train" if is_train else "val"
            
            # 마스크 로드
            mask_path = masks_dir / f"{base_name}.png"
            if not mask_path.exists():
                print(f"\nWarning: Mask not found for {base_name}")
                continue
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.max() == 0:
                continue
            
            # harmonized 이미지 복사
            harm_image = Image.open(harm_img_path)
            output_filename = f"{base_name}.jpg"
            image_path = image_dir / output_filename
            harm_image.save(image_path)
            
            # YOLO 어노테이션 저장
            label_path = label_dir / output_filename.replace('.jpg', '.txt')
            if save_yolo_annotation(mask, harm_image.size[0], harm_image.size[1], label_path, class_id=0):
                output_paths.append(image_path)
            
        except Exception as e:
            print(f"\nError converting {harm_img_path.name}: {e}")
            continue
    
    pbar.close()
    
    print(f"\n✓ YOLO conversion completed: {len(output_paths)} images")
    print(f"  Output directory: {yolo_base}")
    
    return output_paths


def process_with_dovenet(
    object_category: str = "object",
    semantic_locations: List[str] = None,
    broad_categories: List[str] = None,
    max_backgrounds: int = 800,
    similarity_threshold: float = 0.7,
    use_depth: bool = True,
    max_workers: int = 5,
    max_composites: int = 100,
    occlusion_threshold: float = 0.3,
    class_name: str = "object"
) -> List[Path]:
    """
    DoveNet을 사용한 전체 파이프라인 실행
    
    Args:
        object_category: 객체 카테고리
        semantic_locations: 의미론적 위치 리스트
        broad_categories: 대분류 카테고리 리스트
        max_backgrounds: 최대 배경 개수
        similarity_threshold: 유사도 임계값
        use_depth: Depth 사용 여부
        max_workers: 병렬 워커 수
        max_composites: 생성할 합성 이미지 개수
        occlusion_threshold: Occlusion 임계값
        class_name: YOLO 클래스 이름
    
    Returns:
        최종 생성된 YOLO 이미지 경로 리스트
    """
    print("=" * 60)
    print("Processing with DoveNet Harmonization")
    print(f"Target: {max_composites} composite images → DoveNet → YOLO format")
    print("=" * 60)
    
    # Step 1: 배경 찾기
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
    
    print(f"✓ Found {len(backgrounds)} backgrounds")
    
    # Step 2: 객체 로드
    print(f"\n[Step 2] Loading segmented objects...")
    objects = get_all_objects(
        original_input_dir=INPUT_DIR,
        masked_frames_dir=MASKED_FRAMES_DIR
    )
    print(f"✓ Loaded {len(objects)} objects")
    
    if len(objects) == 0:
        print("Warning: No objects loaded!")
        return []
    
    # Step 3: DoveNet 데이터셋 준비 (간단한 합성)
    print(f"\n[Step 3] Preparing DoveNet dataset...")
    dataset_dir = prepare_dovenet_dataset(
        objects=objects,
        backgrounds=backgrounds,
        use_depth=use_depth,
        depth_offset=0.05,
        occlusion_threshold=occlusion_threshold,
        max_composites=max_composites
    )
    
    if dataset_dir is None:
        print("Error: Failed to prepare DoveNet dataset!")
        return []
    
    # Step 4: DoveNet 실행
    print(f"\n[Step 4] Running DoveNet harmonization...")
    harmonized_dir = run_dovenet_harmonization(
        dataset_root=dataset_dir,
        experiment_name="image_analyzer_harmonization"
    )
    
    if harmonized_dir is None:
        print("Error: DoveNet harmonization failed!")
        return []
    
    # Step 5: YOLO 형식으로 변환
    print(f"\n[Step 5] Converting to YOLO format...")
    output_paths = convert_to_yolo_format(
        harmonized_dir=harmonized_dir,
        dovenet_data_dir=dataset_dir,
        class_name=class_name
    )
    
    print("\n" + "=" * 60)
    print(f"Pipeline completed! Generated {len(output_paths)} YOLO images")
    print(f"Results saved to: {OUTPUT_DIR / 'dovenet_result'}")
    print("=" * 60)
    
    return output_paths


def main():
    """
    메인 엔트리 포인트
    """
    print("Using DoveNet for natural harmonization")
    print(f"  Input: {INPUT_DIR}")
    print(f"  Masked frames: {MASKED_FRAMES_DIR}")
    print(f"  DoveNet: {DOVENET_ROOT}")
    print()
    
    # 파이프라인 실행
    results = process_with_dovenet(
        object_category="monkey_doll",
        semantic_locations=["shelf", "bed", "couch", "table", "toy box"],
        broad_categories=["home", "indoor", "living room", "bedroom", "house interior"],
        max_backgrounds=100,  # 빠른 테스트를 위해 줄임
        similarity_threshold=0.7,
        use_depth=True,
        max_workers=7,
        max_composites=50,  # 50개만 먼저 테스트
        occlusion_threshold=0.3,
        class_name="monkey_doll"
    )
    
    if results:
        print(f"\nGenerated {len(results)} YOLO images:")
        for r in results[:5]:  # 처음 5개만 출력
            print(f"  - {r.name}")
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more")
    else:
        print("\nNo images generated. Check the logs above for errors.")


if __name__ == "__main__":
    main()
