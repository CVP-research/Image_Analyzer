"""
Simple Paste + DoveNet 공격적 데이터 증강 파이프라인

1. 배경 랜덤 선택 (4배 업스케일)
2. 객체 랜덤 선택
3. 공격적 Simple Paste (4000장)
4. DoveNet 실행 (batch=10)
5. YOLO 데이터셋 구성

특징:
- 배경 4배 업스케일로 고해상도 유지
- 객체/배경 해상도 고려한 적절한 스케일링
- 마스크 미리 저장 후 DoveNet 결과와 매칭
"""

import sys
import subprocess
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict
import json

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 상위 디렉토리의 main.py 함수들을 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    get_all_objects,
    BASE_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    MASKED_FRAMES_DIR
)
from yolo_utils import save_yolo_annotation, create_yolo_dataset_structure


# 경로 설정
DATASET_DIR = Path("/home/rocknroll1397/Image_Analyzer/dataset/train")
DOVENET_ROOT = Path("/home/rocknroll1397/Image_Analyzer/Image-Harmonization-Dataset-iHarmony4/DoveNet")
DOVENET_VENV_PYTHON = Path("/home/rocknroll1397/Image_Analyzer/Image-Harmonization-Dataset-iHarmony4/venv/bin/python")
DOVENET_DATA_DIR = DOVENET_ROOT / "data" / "simple_paste"


def get_random_backgrounds(dataset_dir: Path, num_backgrounds: int = 500) -> List[Path]:
    """
    데이터셋에서 배경 이미지 랜덤 선택
    
    Args:
        dataset_dir: 데이터셋 디렉토리 (ADE20K 등)
        num_backgrounds: 선택할 배경 개수
    
    Returns:
        배경 이미지 경로 리스트
    """
    print(f"\n[Background Selection] Scanning {dataset_dir}...")
    
    # 모든 이미지 파일 찾기
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(dataset_dir.rglob(f"*{ext}"))
    
    print(f"  Found {len(all_images)} total images")
    
    # 랜덤 선택
    if len(all_images) > num_backgrounds:
        selected = random.sample(all_images, num_backgrounds)
    else:
        selected = all_images
    
    print(f"  Selected {len(selected)} backgrounds")
    return selected


def upscale_image(image: Image.Image, scale_factor: int = 4) -> Image.Image:
    """
    이미지 업스케일 (고품질 리샘플링)
    
    Args:
        image: 원본 이미지
        scale_factor: 업스케일 배율
    
    Returns:
        업스케일된 이미지
    """
    new_size = (image.width * scale_factor, image.height * scale_factor)
    return image.resize(new_size, Image.LANCZOS)


def simple_paste_composite(
    bg_image: Image.Image,
    obj_image: Image.Image,
    obj_scale_range: Tuple[float, float] = (0.2, 0.4),
    rotation_range: Tuple[float, float] = (-30, 30),
    position_margin: float = 0.1
) -> Tuple[Image.Image, np.ndarray]:
    """
    Simple Paste 합성 (해상도 고려)
    
    Args:
        bg_image: 배경 이미지 (업스케일된 고해상도)
        obj_image: 객체 이미지 (RGBA)
        obj_scale_range: 객체 크기 범위 (배경 대비 비율)
        rotation_range: 회전 각도 범위
        position_margin: 가장자리 여백 비율
    
    Returns:
        (합성 이미지, 객체 마스크)
    """
    bg_width, bg_height = bg_image.size
    obj_rgba = obj_image.convert("RGBA")
    
    # 1. 객체 크기 조정 (배경 크기 대비)
    bg_diag = np.sqrt(bg_width**2 + bg_height**2)
    obj_scale = random.uniform(*obj_scale_range)
    target_size = int(bg_diag * obj_scale)
    
    # 객체의 현재 크기
    obj_width, obj_height = obj_rgba.size
    obj_max_dim = max(obj_width, obj_height)
    
    # 리스케일
    scale_ratio = target_size / obj_max_dim
    new_obj_width = int(obj_width * scale_ratio)
    new_obj_height = int(obj_height * scale_ratio)
    obj_resized = obj_rgba.resize((new_obj_width, new_obj_height), Image.LANCZOS)
    
    # 2. 회전
    rotation_angle = random.uniform(*rotation_range)
    obj_rotated = obj_resized.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
    
    # 3. 랜덤 위치 (여백 고려)
    obj_rot_width, obj_rot_height = obj_rotated.size
    
    margin_x = int(bg_width * position_margin)
    margin_y = int(bg_height * position_margin)
    
    max_x = bg_width - obj_rot_width - margin_x
    max_y = bg_height - obj_rot_height - margin_y
    
    if max_x < margin_x or max_y < margin_y:
        # 객체가 너무 크면 중앙 배치
        x = (bg_width - obj_rot_width) // 2
        y = (bg_height - obj_rot_height) // 2
    else:
        x = random.randint(margin_x, max_x)
        y = random.randint(margin_y, max_y)
    
    # 4. 합성
    composite = bg_image.convert("RGBA").copy()
    composite.paste(obj_rotated, (x, y), obj_rotated)
    composite_rgb = composite.convert("RGB")
    
    # 5. 마스크 생성
    mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
    obj_alpha = np.array(obj_rotated.split()[-1])
    
    # 마스크 배치 위치 계산 (경계 체크)
    y_end = min(y + obj_rot_height, bg_height)
    x_end = min(x + obj_rot_width, bg_width)
    obj_h = y_end - y
    obj_w = x_end - x
    
    mask[y:y_end, x:x_end] = obj_alpha[:obj_h, :obj_w]

    # 1픽셀 침식(erosion) 적용
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    return composite_rgb, mask


def prepare_simple_paste_dataset(
    objects: List[Tuple[Image.Image, Dict]],
    backgrounds: List[Path],
    num_composites: int = 4000,
    upscale_factor: int = 4
) -> Tuple[Path, Dict]:
    """
    Simple Paste 데이터셋 준비 + 마스크 정보 저장
    
    Returns:
        (데이터셋 디렉토리, 마스크 정보 딕셔너리)
    """
    # 디렉토리 생성
    composite_dir = DOVENET_DATA_DIR / "composite_images"
    masks_dir = DOVENET_DATA_DIR / "masks"
    real_dir = DOVENET_DATA_DIR / "real_images"
    metadata_dir = DOVENET_DATA_DIR / "metadata"
    
    # 기존 데이터 삭제 후 새로 생성
    if DOVENET_DATA_DIR.exists():
        shutil.rmtree(DOVENET_DATA_DIR)
    
    composite_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Simple Paste] Creating {num_composites} composites...")
    print(f"  Backgrounds: {len(backgrounds)}")
    print(f"  Objects: {len(objects)}")
    print(f"  Upscale factor: {upscale_factor}x")
    
    image_list = []
    mask_metadata = {}  # {filename: mask_array}
    
    pbar = tqdm(total=num_composites, desc="Creating composites", unit="img")
    
    generated_count = 0
    attempts = 0
    max_attempts = num_composites * 3  # 실패 대비
    
    while generated_count < num_composites and attempts < max_attempts:
        attempts += 1
        try:
            # 랜덤 배경 선택
            bg_path = random.choice(backgrounds)
            bg_image = Image.open(bg_path).convert("RGB")

            # 업스케일 여부에 따라 배경 선택
            if upscale_factor > 1:
                bg_final = upscale_image(bg_image, upscale_factor)
            else:
                bg_final = bg_image

            # 랜덤 객체 선택
            obj_image, obj_meta = random.choice(objects)

            # Simple Paste 합성
            composite_img, obj_mask = simple_paste_composite(
                bg_image=bg_final,
                obj_image=obj_image,
                obj_scale_range=(0.4, 0.7),
                rotation_range=(-30, 30),
                position_margin=0.05
            )

            # 마스크가 비어있는지 확인
            if obj_mask.max() == 0:
                continue

            # 파일명 생성
            filename = f"composite_{generated_count:04d}"

            # 저장
            composite_img.save(composite_dir / f"{filename}.jpg", quality=95)

            # 마스크는 DoveNet용과 메타데이터용 분리
            mask_pil = Image.fromarray(obj_mask)
            mask_pil.save(masks_dir / f"{filename}.png")

            # 메타데이터에 마스크 저장 (나중에 YOLO 어노테이션용)
            mask_metadata[filename] = obj_mask.copy()

            # 배경 저장
            bg_final.save(real_dir / f"{filename}.jpg", quality=95)

            # 리스트에 추가
            image_list.append(f"composite_images/{filename}.jpg")

            generated_count += 1
            pbar.update(1)
        except Exception as e:
            continue
    
    pbar.close()
    
    # IHD_test.txt 생성
    test_list_file = DOVENET_DATA_DIR / "IHD_test.txt"
    with open(test_list_file, 'w') as f:
        f.write('\n'.join(image_list))
    
    # 마스크 메타데이터 저장
    print(f"\n  Saving mask metadata...")
    mask_metadata_file = metadata_dir / "masks_metadata.npz"
    np.savez_compressed(mask_metadata_file, **mask_metadata)
    
    print(f"\n✓ Simple Paste dataset prepared: {generated_count} images")
    print(f"  Composite images: {composite_dir}")
    print(f"  Masks: {masks_dir}")
    print(f"  Real images: {real_dir}")
    print(f"  Metadata: {mask_metadata_file}")
    
    return DOVENET_DATA_DIR, mask_metadata


def run_dovenet_batch(
    dataset_root: Path,
    experiment_name: str = "simple_paste",
    batch_size: int = 10,
    num_test: int = None
) -> Path:
    """
    DoveNet 배치 실행
    
    Returns:
        결과 이미지 디렉토리 경로
    """
    print(f"\n[DoveNet Harmonization] Running DoveNet with batch size {batch_size}...")
    
    # 생성된 이미지 개수 확인
    composite_images = list((dataset_root / "composite_images").glob("*.jpg"))
    if num_test is None:
        num_test = len(composite_images)
    
    print(f"  Processing {num_test} images...")
    
    # DoveNet 명령어 구성
    cmd = [
        str(DOVENET_VENV_PYTHON),
        "test.py",
        "--dataset_root", str(dataset_root)+"/",
        "--name", experiment_name,
        "--model", "dovenet",
        "--dataset_mode", "iharmony4",
        "--netG", "s2ad",
        "--is_train", "0",
        "--norm", "batch",
        "--batch_size", str(batch_size),
        "--no_flip",
        "--preprocess", "none",
        "--num_test", str(num_test)
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    
    # DoveNet 디렉토리에서 실행
    try:
        print(f"\n  Starting DoveNet harmonization...")
        
        # 실시간 출력
        process = subprocess.Popen(
            cmd,
            cwd=str(DOVENET_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
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
        if e.stdout:
            print(f"  Stdout: {e.stdout[-500:]}")  # 마지막 500자만
        if e.stderr:
            print(f"  Stderr: {e.stderr[-500:]}")
        return None


def create_yolo_dataset_from_dovenet(
    harmonized_dir: Path,
    mask_metadata: Dict[str, np.ndarray],
    class_name: str = "object"
) -> List[Path]:
    """
    DoveNet 결과 + 저장된 마스크로 YOLO 데이터셋 구성
    
    Args:
        harmonized_dir: DoveNet 결과 이미지 디렉토리
        mask_metadata: 저장된 마스크 딕셔너리
        class_name: YOLO 클래스 이름
    
    Returns:
        생성된 YOLO 이미지 경로 리스트
    """
    print(f"\n[YOLO Dataset] Creating YOLO dataset from DoveNet results...")
    
    # YOLO 데이터셋 구조 생성
    yolo_base = OUTPUT_DIR / "simple_paste_dovenet"
    images_train, images_val, labels_train, labels_val = create_yolo_dataset_structure(
        yolo_base, [class_name]
    )
    
    # harmonized 이미지 처리
    harmonized_images = sorted(harmonized_dir.glob("*.jpg"))
    
    output_paths = []
    train_count = 0
    val_count = 0
    
    pbar = tqdm(harmonized_images, desc="Creating YOLO dataset", unit="img")
    
    for idx, harm_img_path in enumerate(pbar):
        try:
            # 파일명에서 인덱스 추출 (composite_0000.jpg)
            base_name = harm_img_path.stem  # composite_0000
            
            # 마스크 로드 (메타데이터에서)
            if base_name not in mask_metadata:
                print(f"\nWarning: Mask not found for {base_name}")
                continue
            
            obj_mask = mask_metadata[base_name]
            
            # 마스크 확인
            if obj_mask.max() == 0:
                continue
            
            # train/val 분리 (80/20)
            is_train = (idx % 5) != 0
            image_dir = images_train if is_train else images_val
            label_dir = labels_train if is_train else labels_val
            
            # harmonized 이미지 로드 및 저장
            harm_image = Image.open(harm_img_path)
            output_filename = f"{base_name}.jpg"
            image_path = image_dir / output_filename
            harm_image.save(image_path, quality=95)
            
            # YOLO 어노테이션 저장
            label_path = label_dir / output_filename.replace('.jpg', '.txt')
            if save_yolo_annotation(obj_mask, harm_image.size[0], harm_image.size[1], label_path, class_id=0):
                output_paths.append(image_path)
                if is_train:
                    train_count += 1
                else:
                    val_count += 1
            
            pbar.set_postfix({"train": train_count, "val": val_count})
            
        except Exception as e:
            print(f"\nError processing {harm_img_path.name}: {e}")
            continue
    
    pbar.close()
    
    print(f"\n✓ YOLO dataset created: {len(output_paths)} images")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Output directory: {yolo_base}")
    
    return output_paths


def aggressive_simple_paste_pipeline(
    object_category: str = "object",
    num_backgrounds: int = 500,
    num_composites: int = 4000,
    upscale_factor: int = 4,
    batch_size: int = 10,
    class_name: str = "object"
) -> List[Path]:
    """
    공격적 Simple Paste + DoveNet 파이프라인
    
    Args:
        object_category: 객체 카테고리
        num_backgrounds: 사용할 배경 개수
        num_composites: 생성할 합성 이미지 개수
        upscale_factor: 배경 업스케일 배율
        batch_size: DoveNet 배치 크기
        class_name: YOLO 클래스 이름
    
    Returns:
        최종 YOLO 이미지 경로 리스트
    """
    print("=" * 70)
    print("Aggressive Simple Paste + DoveNet Pipeline")
    print(f"Target: {num_composites} images with {upscale_factor}x upscaled backgrounds")
    print(f"DoveNet batch size: {batch_size}")
    print("=" * 70)
    
    # Step 1: 배경 랜덤 선택
    print(f"\n[Step 1/5] Selecting random backgrounds...")
    backgrounds = get_random_backgrounds(DATASET_DIR, num_backgrounds)
    
    if len(backgrounds) == 0:
        print("Error: No backgrounds found!")
        return []
    
    # Step 2: 객체 로드
    print(f"\n[Step 2/5] Loading objects...")
    objects = get_all_objects(
        original_input_dir=INPUT_DIR,
        masked_frames_dir=MASKED_FRAMES_DIR
    )
    print(f"✓ Loaded {len(objects)} objects")
    
    if len(objects) == 0:
        print("Error: No objects found!")
        return []
    
    # Step 3: Simple Paste 데이터셋 생성
    print(f"\n[Step 3/5] Creating Simple Paste dataset...")
    dataset_dir, mask_metadata = prepare_simple_paste_dataset(
        objects=objects,
        backgrounds=backgrounds,
        num_composites=num_composites,
        upscale_factor=upscale_factor
    )
    
    if dataset_dir is None or len(mask_metadata) == 0:
        print("Error: Failed to create dataset!")
        return []
    
    # Step 4: DoveNet 실행
    print(f"\n[Step 4/5] Running DoveNet harmonization...")
    harmonized_dir = run_dovenet_batch(
        dataset_root=dataset_dir,
        experiment_name="simple_paste",
        batch_size=batch_size
    )
    
    if harmonized_dir is None:
        print("Error: DoveNet harmonization failed!")
        return []
    
    # Step 5: YOLO 데이터셋 생성
    print(f"\n[Step 5/5] Creating YOLO dataset...")
    output_paths = create_yolo_dataset_from_dovenet(
        harmonized_dir=harmonized_dir,
        mask_metadata=mask_metadata,
        class_name=class_name
    )
    
    print("\n" + "=" * 70)
    print(f"Pipeline completed successfully!")
    print(f"Generated {len(output_paths)} YOLO-ready images")
    print(f"Results: {OUTPUT_DIR / 'simple_paste_dovenet'}")
    print("=" * 70)
    
    return output_paths


# def save_yolo_dataset_only(
#     objects: List[Tuple[Image.Image, dict]],
#     backgrounds: List[Path],
#     num_composites: int = 4000,
#     upscale_factor: int = 4,
#     class_name: str = "object"
# ) -> List[Path]:
#     """
#     DOVENET 없이 YOLO 데이터셋만 생성 (합성+마스크)
#     """
#     # 디렉토리 생성
#     composite_dir = DOVENET_DATA_DIR / "composite_images"
#     masks_dir = DOVENET_DATA_DIR / "masks"
#     real_dir = DOVENET_DATA_DIR / "real_images"
#     metadata_dir = DOVENET_DATA_DIR / "metadata"
    
#     # 기존 데이터 삭제 후 새로 생성
#     if DOVENET_DATA_DIR.exists():
#         shutil.rmtree(DOVENET_DATA_DIR)
    
#     composite_dir.mkdir(parents=True, exist_ok=True)
#     masks_dir.mkdir(parents=True, exist_ok=True)
#     real_dir.mkdir(parents=True, exist_ok=True)
#     metadata_dir.mkdir(parents=True, exist_ok=True)
    
#     print(f"\n[Simple Paste] Creating {num_composites} composites...")
#     print(f"  Backgrounds: {len(backgrounds)}")
#     print(f"  Objects: {len(objects)}")
#     print(f"  Upscale factor: {upscale_factor}x")
    
#     image_list = []
#     mask_metadata = {}  # {filename: mask_array}
    
#     pbar = tqdm(total=num_composites, desc="Creating composites", unit="img")
    
#     generated_count = 0
#     attempts = 0
#     max_attempts = num_composites * 3  # 실패 대비
    
#     while generated_count < num_composites and attempts < max_attempts:
#         attempts += 1
#         try:
#             # 랜덤 배경 선택
#             bg_path = random.choice(backgrounds)
#             bg_image = Image.open(bg_path).convert("RGB")

#             # 업스케일 여부에 따라 배경 선택
#             if upscale_factor > 1:
#                 bg_final = upscale_image(bg_image, upscale_factor)
#             else:
#                 bg_final = bg_image

#             # 랜덤 객체 선택
#             obj_image, obj_meta = random.choice(objects)

#             # Simple Paste 합성
#             composite_img, obj_mask = simple_paste_composite(
#                 bg_image=bg_final,
#                 obj_image=obj_image,
#                 obj_scale_range=(0.1, 0.4),
#                 rotation_range=(-30, 30),
#                 position_margin=0.05
#             )

#             # 마스크가 비어있는지 확인
#             if obj_mask.max() == 0:
#                 continue

#             # 파일명 생성
#             filename = f"composite_{generated_count:04d}"

#             # 저장
#             composite_img.save(composite_dir / f"{filename}.jpg", quality=95)

#             # 마스크는 DoveNet용과 메타데이터용 분리
#             mask_pil = Image.fromarray(obj_mask)
#             mask_pil.save(masks_dir / f"{filename}.png")

#             # 메타데이터에 마스크 저장 (나중에 YOLO 어노테이션용)
#             mask_metadata[filename] = obj_mask.copy()

#             # 배경 저장
#             bg_final.save(real_dir / f"{filename}.jpg", quality=95)

#             # 리스트에 추가
#             image_list.append(f"composite_images/{filename}.jpg")

#             generated_count += 1
#             pbar.update(1)
#         except Exception as e:
#             continue
    
#     pbar.close()
    
#     # IHD_test.txt 생성
#     test_list_file = DOVENET_DATA_DIR / "IHD_test.txt"
#     with open(test_list_file, 'w') as f:
#         f.write('\n'.join(image_list))
    
#     # 마스크 메타데이터 저장
#     print(f"\n  Saving mask metadata...")
#     mask_metadata_file = metadata_dir / "masks_metadata.npz"
#     np.savez_compressed(mask_metadata_file, **mask_metadata)
    
#     print(f"\n✓ Simple Paste dataset prepared: {generated_count} images")
#     print(f"  Composite images: {composite_dir}")
#     print(f"  Masks: {masks_dir}")
#     print(f"  Real images: {real_dir}")
#     print(f"  Metadata: {mask_metadata_file}")
    
#     # YOLO 데이터셋 구조 생성
#     yolo_base = OUTPUT_DIR / "simple_paste_yolo"
#     images_train, images_val, labels_train, labels_val = create_yolo_dataset_structure(
#         yolo_base, [class_name]
#     )
#     composite_dir = dataset_dir / "composite_images"
#     masks_dir = dataset_dir / "masks"
#     harmonized_images = sorted(composite_dir.glob("*.jpg"))
    
#     output_paths = []
#     train_count = 0
#     val_count = 0
    
#     pbar = tqdm(harmonized_images, desc="Creating YOLO dataset", unit="img")
    
#     for idx, img_path in enumerate(pbar):
#         base_name = img_path.stem
#         if base_name not in mask_metadata:
#             continue
#         obj_mask = mask_metadata[base_name]
#         if obj_mask.max() == 0:
#             continue
#         is_train = (idx % 5) != 0
#         image_dir = images_train if is_train else images_val
#         label_dir = labels_train if is_train else labels_val
#         output_filename = f"{base_name}.jpg"
#         image_path = image_dir / output_filename
#         img = Image.open(img_path)
#         img.save(image_path, quality=95)
#         label_path = label_dir / output_filename.replace('.jpg', '.txt')
#         if save_yolo_annotation(obj_mask, img.size[0], img.size[1], label_path, class_id=0):
#             output_paths.append(image_path)
#             if is_train:
#                 train_count += 1
#             else:
#                 val_count += 1
#         pbar.set_postfix({"train": train_count, "val": val_count})
    
#     pbar.close()
    
#     print(f"\n✓ YOLO dataset created: {len(output_paths)} images")
#     print(f"  Train: {train_count} images")
#     print(f"  Val: {val_count} images")
#     print(f"  Output directory: {yolo_base}")
    
#     return output_paths


def move_yolo_split_to_folder(yolo_base: Path, target_dir: Path):
    """
    YOLO 데이터셋의 train/val 이미지를 target_dir/train, target_dir/val로 이동
    """
    import shutil
    train_img = yolo_base / "images" / "train"
    val_img = yolo_base / "images" / "val"
    train_lbl = yolo_base / "labels" / "train"
    val_lbl = yolo_base / "labels" / "val"
    (target_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (target_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
    for f in train_img.glob("*.jpg"):
        shutil.copy(f, target_dir / "train" / "images" / f.name)
    for f in val_img.glob("*.jpg"):
        shutil.copy(f, target_dir / "val" / "images" / f.name)
    for f in train_lbl.glob("*.txt"):
        shutil.copy(f, target_dir / "train" / "labels" / f.name)
    for f in val_lbl.glob("*.txt"):
        shutil.copy(f, target_dir / "val" / "labels" / f.name)
    print(f"YOLO split moved to {target_dir}/train and {target_dir}/val")


def main():
    """
    메인 엔트리 포인트
    """
    print("Aggressive Simple Paste + DoveNet Pipeline")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Objects: {INPUT_DIR}")
    print(f"  DoveNet: {DOVENET_ROOT}")
    print()
    
    # 파이프라인 실행
    results = aggressive_simple_paste_pipeline(
        object_category="monkey_doll",
        num_backgrounds=500,      # 500개 배경 사용
        num_composites=4000,      # 4000장 생성
        # num_backgrounds=5,
        # num_composites=20,
        upscale_factor=1,         # 4배 업스케일
        batch_size=10,            # 배치 크기 10
        class_name="monkey_doll"
    )
    
    if results:
        print(f"\n✓ Successfully generated {len(results)} YOLO images")
        print(f"\nSample outputs:")
        for r in results[:10]:
            print(f"  - {r.name}")
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more")
    else:
        print("\n✗ Pipeline failed. Check logs above.")


if __name__ == "__main__":
    main()
