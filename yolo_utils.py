"""
YOLO 데이터셋 생성 유틸리티

Segmentation mask를 YOLO format으로 변환하고 저장
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import yaml


def mask_to_yolo_polygon(
    mask: np.ndarray,
    img_width: int,
    img_height: int,
    class_id: int = 0
) -> str:
    """
    Binary mask를 YOLO segmentation format으로 변환
    
    YOLO format: <class_id> <x1> <y1> <x2> <y2> ... (normalized 0-1)
    
    Args:
        mask: Binary mask (H x W, bool or uint8)
        img_width: 이미지 너비
        img_height: 이미지 높이
        class_id: 클래스 ID (기본 0)
    
    Returns:
        YOLO format annotation string (한 줄)
        빈 문자열이면 valid한 polygon이 없음
    """
    # Mask를 uint8로 변환
    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)
    
    # Contour 찾기 (외곽선)
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,  # 외곽 contour만
        cv2.CHAIN_APPROX_SIMPLE  # 간단하게 압축
    )
    
    if len(contours) == 0:
        return ""
    
    # 가장 큰 contour 선택
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 너무 작은 contour는 무시 (최소 10 픽셀)
    if cv2.contourArea(largest_contour) < 10:
        return ""
    
    # Contour를 normalized coordinates로 변환
    points = largest_contour.reshape(-1, 2)
    normalized_points = []
    
    for x, y in points:
        # 0-1 범위로 정규화
        nx = x / img_width
        ny = y / img_height
        
        # 범위 체크
        nx = np.clip(nx, 0.0, 1.0)
        ny = np.clip(ny, 0.0, 1.0)
        
        normalized_points.append(f"{nx:.6f}")
        normalized_points.append(f"{ny:.6f}")
    
    # YOLO format: class_id x1 y1 x2 y2 x3 y3 ...
    yolo_line = f"{class_id} " + " ".join(normalized_points)
    
    return yolo_line


def save_yolo_annotation(
    mask: np.ndarray,
    img_width: int,
    img_height: int,
    label_path: Path,
    class_id: int = 0
) -> bool:
    """
    YOLO annotation 파일 저장
    
    Args:
        mask: Binary mask
        img_width: 이미지 너비
        img_height: 이미지 높이
        label_path: 저장할 label 파일 경로 (.txt)
        class_id: 클래스 ID
    
    Returns:
        성공 여부
    """
    yolo_line = mask_to_yolo_polygon(mask, img_width, img_height, class_id)
    
    if not yolo_line:
        return False
    
    # 디렉토리 생성
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 파일 저장
    with open(label_path, 'w') as f:
        f.write(yolo_line + '\n')
    
    return True


def create_yolo_dataset_structure(
    output_base: Path,
    class_names: List[str]
) -> Tuple[Path, Path, Path, Path]:
    """
    YOLO 데이터셋 디렉토리 구조 생성
    
    output_base/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
    
    Args:
        output_base: 출력 디렉토리 (예: output/result)
        class_names: 클래스 이름 리스트 (예: ['monkey_doll'])
    
    Returns:
        (images_train, images_val, labels_train, labels_val) 경로
    """
    # 디렉토리 생성
    images_train = output_base / "images" / "train"
    images_val = output_base / "images" / "val"
    labels_train = output_base / "labels" / "train"
    labels_val = output_base / "labels" / "val"
    
    images_train.mkdir(parents=True, exist_ok=True)
    images_val.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)
    
    # data.yaml 생성
    data_yaml = {
        'path': str(output_base.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = output_base / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✓ YOLO dataset structure created at: {output_base}")
    print(f"  Classes: {class_names}")
    print(f"  data.yaml: {yaml_path}")
    
    return images_train, images_val, labels_train, labels_val


def split_train_val(
    total_count: int,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Train/Val split indices 생성
    
    Args:
        total_count: 전체 데이터 개수
        val_ratio: Validation 비율 (기본 0.2 = 20%)
        seed: Random seed
    
    Returns:
        (train_indices, val_indices)
    """
    import random
    
    random.seed(seed)
    indices = list(range(total_count))
    random.shuffle(indices)
    
    val_count = int(total_count * val_ratio)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    
    return train_indices, val_indices
