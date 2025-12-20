"""
네거티브 샘플 생성 및 데이터셋 추가

1. 랜덤 객체 10개 선택
2. 각 객체당 fake 2개 생성
3. 100장의 랜덤 배경에 합성
4. result_sum 폴더에 추가
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import random
import yaml
from modules.fake_object_generator import generate_fake_object


def load_dataset_info(dataset_path: Path) -> dict:
    """
    data.yaml에서 데이터셋 정보 로드
    """
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data


def get_random_objects(num_objects: int = 10) -> List[Path]:
    """
    masked_frames 폴더에서 랜덤으로 누끼 딴 객체 선택
    """
    masked_frames_dir = Path("output/dataset/masked_frames")
    
    if not masked_frames_dir.exists():
        print(f"Error: {masked_frames_dir} not found")
        return []
    
    all_images = list(masked_frames_dir.glob("*.png"))
    
    if len(all_images) < num_objects:
        print(f"Warning: Only {len(all_images)} objects available, selecting all")
        return all_images
    
    selected = random.sample(all_images, num_objects)
    return selected


def get_random_backgrounds(dataset_path: Path, num_backgrounds: int = 100) -> List[Path]:
    """
    데이터셋에서 랜덤으로 배경 이미지 선택
    """
    # dataset/train과 dataset/val에서 배경 수집
    bg_folders = []
    
    dataset_root = Path("dataset")
    
    for split in ["train", "val"]:
        split_path = dataset_root / split
        if split_path.exists():
            # 각 카테고리 폴더에서 이미지 수집
            for category_folder in split_path.iterdir():
                if category_folder.is_dir():
                    bg_folders.extend(list(category_folder.glob("*.jpg")))
    
    if len(bg_folders) < num_backgrounds:
        print(f"Warning: Only {len(bg_folders)} backgrounds available")
        return bg_folders
    
    selected = random.sample(bg_folders, num_backgrounds)
    return selected


def composite_simple(bg_image: Image.Image, overlay_image: Image.Image, 
                    overlay_position: Tuple[int, int]) -> Image.Image:
    """
    단순 알파 블렌딩 합성
    """
    overlay_image = overlay_image.convert("RGBA")
    
    bg_array = np.array(bg_image.convert("RGB"))
    overlay_array = np.array(overlay_image)
    
    h_bg, w_bg = bg_array.shape[:2]
    h_ov, w_ov = overlay_array.shape[:2]
    
    # 중심 기준으로 top-left 좌표 계산
    center_x, center_y = overlay_position
    x_offset = center_x - w_ov // 2
    y_offset = center_y - h_ov // 2
    
    # 알파 채널 추출
    alpha = overlay_array[:, :, 3] / 255.0
    overlay_rgba = overlay_array.astype(np.float32) / 255.0
    overlay_rgb = overlay_rgba[:, :, :3]
    overlay_rgb_premult = overlay_rgb * alpha[:, :, np.newaxis]
    
    bg_float = bg_array.astype(np.float32) / 255.0
    result = bg_float.copy()
    
    for i in range(h_ov):
        for j in range(w_ov):
            alpha_val = alpha[i, j]
            
            if alpha_val < 0.001:
                continue
            
            bg_y = y_offset + i
            bg_x = x_offset + j
            
            if 0 <= bg_x < w_bg and 0 <= bg_y < h_bg:
                result[bg_y, bg_x] = (
                    overlay_rgb_premult[i, j] +
                    bg_float[bg_y, bg_x] * (1.0 - alpha_val)
                )
    
    result_uint8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result_uint8)


def get_next_negative_index(dataset_path: Path, split: str) -> int:
    """
    다음 네거티브 샘플 인덱스 찾기
    """
    images_dir = dataset_path / "images" / split
    negative_files = list(images_dir.glob("negative_*.png"))
    
    if not negative_files:
        return 1
    
    max_idx = 0
    for f in negative_files:
        try:
            idx = int(f.stem.split("_")[1])
            max_idx = max(max_idx, idx)
        except:
            continue
    
    return max_idx + 1


def main():
    # 설정
    dataset_path = Path("output/dataset/result_sum")
    num_objects = 10
    num_fakes_per_object = 2
    num_backgrounds = 100
    
    print("=" * 60)
    print("네거티브 샘플 생성 시작")
    print("=" * 60)
    
    # 1. 랜덤 객체 선택
    print(f"\n1. {num_objects}개의 랜덤 객체 선택 중...")
    selected_objects = get_random_objects(num_objects)
    print(f"   선택된 객체: {len(selected_objects)}개")
    for obj in selected_objects:
        print(f"   - {obj.name}")
    
    # 2. 랜덤 배경 선택
    print(f"\n2. {num_backgrounds}개의 랜덤 배경 선택 중...")
    backgrounds = get_random_backgrounds(dataset_path, num_backgrounds)
    print(f"   선택된 배경: {len(backgrounds)}개")
    
    # 3. Fake 객체 생성 및 합성
    print(f"\n3. Fake 객체 생성 및 합성 중...")
    
    # train/val 비율 4:1
    train_count = int(len(backgrounds) * 0.8)
    train_backgrounds = backgrounds[:train_count]
    val_backgrounds = backgrounds[train_count:]
    
    train_negative_idx = get_next_negative_index(dataset_path, "train")
    val_negative_idx = get_next_negative_index(dataset_path, "val")
    
    total_created = 0
    
    for obj_idx, obj_path in enumerate(selected_objects, 1):
        print(f"\n   객체 {obj_idx}/{len(selected_objects)}: {obj_path.name}")
        
        # 객체 이미지 로드
        obj_img = Image.open(obj_path)
        
        # Fake 객체 2개 생성
        fakes = []
        for fake_idx in range(num_fakes_per_object):
            print(f"      - Fake {fake_idx + 1} 생성 중...")
            fake_img = generate_fake_object(obj_img)
            if fake_img is not None:
                fakes.append(fake_img)
            else:
                print(f"        경고: Fake 생성 실패")
        
        if not fakes:
            print(f"      경고: {obj_path.name}에 대한 fake 생성 실패, 건너뜀")
            continue
        
        # 각 fake를 모든 배경에 합성
        for fake_idx, fake_img in enumerate(fakes):
            print(f"      - Fake {fake_idx + 1} 합성 중...")
            
            # Train 배경에 합성
            for bg_path in train_backgrounds:
                bg_img = Image.open(bg_path).convert("RGB")
                
                # fake 객체가 배경보다 크면 리사이즈
                if fake_img.width > bg_img.width * 0.8 or fake_img.height > bg_img.height * 0.8:
                    scale = min(bg_img.width * 0.6 / fake_img.width, bg_img.height * 0.6 / fake_img.height)
                    new_w = int(fake_img.width * scale)
                    new_h = int(fake_img.height * scale)
                    resized_fake = fake_img.resize((new_w, new_h), Image.LANCZOS)
                else:
                    resized_fake = fake_img
                
                # 랜덤 위치
                w, h = bg_img.size
                max_x = w - resized_fake.width // 2
                max_y = h - resized_fake.height // 2
                min_x = resized_fake.width // 2
                min_y = resized_fake.height // 2
                
                if min_x >= max_x or min_y >= max_y:
                    continue  # 배경이 너무 작으면 건너뛰기
                
                pos_x = random.randint(min_x, max_x)
                pos_y = random.randint(min_y, max_y)
                
                # 합성
                composite_img = composite_simple(bg_img, resized_fake, (pos_x, pos_y))
                
                # 저장
                save_path = dataset_path / "images" / "train" / f"negative_{train_negative_idx:04d}.png"
                composite_img.save(save_path)
                
                # 빈 라벨 생성
                label_path = dataset_path / "labels" / "train" / f"negative_{train_negative_idx:04d}.txt"
                label_path.touch()
                
                train_negative_idx += 1
                total_created += 1
            
            # Val 배경에 합성
            for bg_path in val_backgrounds:
                bg_img = Image.open(bg_path).convert("RGB")
                
                # fake 객체가 배경보다 크면 리사이즈
                if fake_img.width > bg_img.width * 0.8 or fake_img.height > bg_img.height * 0.8:
                    scale = min(bg_img.width * 0.6 / fake_img.width, bg_img.height * 0.6 / fake_img.height)
                    new_w = int(fake_img.width * scale)
                    new_h = int(fake_img.height * scale)
                    resized_fake = fake_img.resize((new_w, new_h), Image.LANCZOS)
                else:
                    resized_fake = fake_img
                
                # 랜덤 위치
                w, h = bg_img.size
                max_x = w - resized_fake.width // 2
                max_y = h - resized_fake.height // 2
                min_x = resized_fake.width // 2
                min_y = resized_fake.height // 2
                
                if min_x >= max_x or min_y >= max_y:
                    continue  # 배경이 너무 작으면 건너뛰기
                
                pos_x = random.randint(min_x, max_x)
                pos_y = random.randint(min_y, max_y)
                
                # 합성
                composite_img = composite_simple(bg_img, resized_fake, (pos_x, pos_y))
                
                # 저장
                save_path = dataset_path / "images" / "val" / f"negative_{val_negative_idx:04d}.png"
                composite_img.save(save_path)
                
                # 빈 라벨 생성
                label_path = dataset_path / "labels" / "val" / f"negative_{val_negative_idx:04d}.txt"
                label_path.touch()
                
                val_negative_idx += 1
                total_created += 1
    
    print("\n" + "=" * 60)
    print(f"완료! 총 {total_created}개의 네거티브 샘플 생성")
    print(f"Train: negative_{get_next_negative_index(dataset_path, 'train') - 1:04d}까지")
    print(f"Val: negative_{get_next_negative_index(dataset_path, 'val') - 1:04d}까지")
    print("=" * 60)


if __name__ == "__main__":
    main()
