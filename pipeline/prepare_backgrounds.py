"""
배경 이미지 준비 모듈

Places365 등의 대용량 이미지 데이터셋에서 특정 색상과 유사한 배경을 선택하고
YOLO 학습에 적합한 형태로 준비합니다.

기능:
- 색상 유사도 기반 이미지 선택
- 4x4 타일링으로 1024x1024 배경 생성 (선택적)
- Train/Val 분할
"""

import os
import glob
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class BackgroundConfig:
    """배경 준비 설정"""
    source_dir: str = "./dataset/val"
    dest_dir: str = "./output/backgrounds"

    # 색상 필터링
    target_rgb: Tuple[int, int, int] = (160, 110, 60)  # 목표 색상 (갈색)
    top_n: int = 32000  # 선택할 최대 이미지 수

    # 타일링 설정
    use_tiling: bool = True  # False면 단순 복사
    tile_size: int = 256
    grid_size: int = 4  # 4x4 = 16장 -> 1024x1024

    # 분할 비율
    val_split_ratio: float = 0.2


def calculate_color_distance(
    image_path: str,
    target_rgb: Tuple[int, int, int]
) -> float:
    """
    이미지의 평균 색상과 타겟 색상 사이의 거리를 계산합니다.

    Args:
        image_path: 이미지 파일 경로
        target_rgb: 목표 색상 (R, G, B)

    Returns:
        색상 거리 (낮을수록 유사)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return float('inf')

        # 속도 향상을 위해 작게 리사이즈
        img_small = cv2.resize(img, (64, 64))
        avg_bgr = np.mean(img_small, axis=(0, 1))
        target_bgr = np.array(target_rgb[::-1])  # RGB -> BGR

        distance = np.linalg.norm(avg_bgr - target_bgr)
        return distance
    except Exception:
        return float('inf')


def pick_top_color_images(
    image_folder: str,
    target_rgb: Tuple[int, int, int],
    top_n: int
) -> List[str]:
    """
    주어진 폴더에서 특정 색상과 가장 유사한 이미지들을 선택합니다.

    Args:
        image_folder: 이미지 폴더 경로
        target_rgb: 목표 색상 (R, G, B)
        top_n: 선택할 이미지 수

    Returns:
        선택된 이미지 경로 리스트
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))

    print(f"Found {len(image_files)} images. Analyzing colors...")

    scored_images = []
    random.shuffle(image_files)  # 다양성을 위해 셔플

    for img_path in tqdm(image_files, desc="Analyzing colors"):
        score = calculate_color_distance(img_path, target_rgb)
        if score != float('inf'):
            scored_images.append((score, img_path))

    # 색상 유사도 순 정렬
    scored_images.sort(key=lambda x: x[0])

    # 상위 N개 선택
    selected_paths = [path for score, path in scored_images[:top_n]]

    print(f"Selected {len(selected_paths)} images")
    if scored_images:
        print(f"Top 5 scores: {[round(s[0], 2) for s in scored_images[:5]]}")

    return selected_paths


def create_tiled_image(
    image_paths: List[str],
    tile_size: int = 256,
    grid_size: int = 4
) -> Optional[np.ndarray]:
    """
    이미지들을 grid_size x grid_size 타일로 합성합니다.

    Args:
        image_paths: 이미지 경로 리스트 (grid_size^2 개 필요)
        tile_size: 개별 타일 크기
        grid_size: 격자 크기 (4x4 = 16장)

    Returns:
        합성된 이미지 또는 None
    """
    tiles_needed = grid_size ** 2
    if len(image_paths) < tiles_needed:
        return None

    rows = []
    idx = 0

    for _ in range(grid_size):
        cols = []
        for _ in range(grid_size):
            path = image_paths[idx]
            img = cv2.imread(path)

            if img is None:
                img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

            img = cv2.resize(img, (tile_size, tile_size))
            cols.append(img)
            idx += 1

        rows.append(np.hstack(cols))

    return np.vstack(rows)


def prepare_backgrounds_tiled(config: BackgroundConfig) -> Tuple[int, int]:
    """
    타일링 방식으로 배경 이미지를 준비합니다.

    Returns:
        (train_count, val_count)
    """
    train_dir = os.path.join(config.dest_dir, 'train')
    val_dir = os.path.join(config.dest_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 색상 기반 이미지 선택
    selected_images = pick_top_color_images(
        config.source_dir,
        config.target_rgb,
        config.top_n
    )

    # 타일 수에 맞게 자르기
    tiles_per_bg = config.grid_size ** 2
    cutoff = (len(selected_images) // tiles_per_bg) * tiles_per_bg
    selected_images = selected_images[:cutoff]

    if len(selected_images) < tiles_per_bg:
        print(f"Not enough images for tiling (need at least {tiles_per_bg})")
        return 0, 0

    # 섞기
    print("Shuffling images for tiling...")
    random.shuffle(selected_images)

    # 청크로 분할
    chunks = [
        selected_images[i:i + tiles_per_bg]
        for i in range(0, len(selected_images), tiles_per_bg)
    ]
    print(f"Will create {len(chunks)} tiled backgrounds")

    # Train/Val 분할
    random.shuffle(chunks)
    split_idx = int(len(chunks) * (1 - config.val_split_ratio))
    train_chunks = chunks[:split_idx]
    val_chunks = chunks[split_idx:]

    def process_and_save(chunks: List[List[str]], save_dir: str, prefix: str) -> int:
        count = 0
        for chunk in tqdm(chunks, desc=f"Creating {prefix} backgrounds"):
            tiled_img = create_tiled_image(chunk, config.tile_size, config.grid_size)
            if tiled_img is not None:
                filename = f"bg_{prefix}_{count:04d}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, tiled_img)
                count += 1
        return count

    print("\nCreating train backgrounds...")
    train_count = process_and_save(train_chunks, train_dir, "train")

    print("\nCreating val backgrounds...")
    val_count = process_and_save(val_chunks, val_dir, "val")

    return train_count, val_count


def prepare_backgrounds_copy(config: BackgroundConfig) -> Tuple[int, int]:
    """
    단순 복사 방식으로 배경 이미지를 준비합니다.

    Returns:
        (train_count, val_count)
    """
    train_dir = os.path.join(config.dest_dir, 'train')
    val_dir = os.path.join(config.dest_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 색상 기반 이미지 선택
    selected_images = pick_top_color_images(
        config.source_dir,
        config.target_rgb,
        config.top_n
    )

    if not selected_images:
        print("No images selected")
        return 0, 0

    # 셔플 후 분할
    random.shuffle(selected_images)
    split_idx = int(len(selected_images) * (1 - config.val_split_ratio))
    train_images = selected_images[:split_idx]
    val_images = selected_images[split_idx:]

    def copy_images(image_list: List[str], dest_folder: str) -> int:
        for i, img_path in enumerate(tqdm(image_list, desc=f"Copying to {os.path.basename(dest_folder)}")):
            ext = os.path.splitext(img_path)[1]
            dest_path = os.path.join(dest_folder, f"bg_{i:04d}{ext}")
            shutil.copy(img_path, dest_path)
        return len(image_list)

    print("\nCopying train images...")
    train_count = copy_images(train_images, train_dir)

    print("\nCopying val images...")
    val_count = copy_images(val_images, val_dir)

    return train_count, val_count


def run(config: Optional[BackgroundConfig] = None):
    """배경 준비 실행"""
    if config is None:
        config = BackgroundConfig()

    print("=" * 50)
    print("Background Preparation")
    print(f"Source: {config.source_dir}")
    print(f"Destination: {config.dest_dir}")
    print(f"Mode: {'Tiling' if config.use_tiling else 'Copy'}")
    print("=" * 50)

    if config.use_tiling:
        train_count, val_count = prepare_backgrounds_tiled(config)
        resolution = config.tile_size * config.grid_size
        print(f"\nResolution: {resolution}x{resolution}")
    else:
        train_count, val_count = prepare_backgrounds_copy(config)

    print("\n" + "=" * 50)
    print("Done!")
    print(f"Train: {train_count} images")
    print(f"Val: {val_count} images")
    print("=" * 50)


if __name__ == "__main__":
    run()
