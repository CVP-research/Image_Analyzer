"""
Negative Sample Generator
객체와 비슷한 색상의 배경을 negative sample로 추가
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import random
from typing import List, Tuple

def extract_object_color_profile(object_images_dir: Path, num_samples: int = 50) -> dict:
    """
    객체 이미지들의 평균 색상 프로파일 추출
    
    Returns:
        dict: {
            'hsv_hist': HSV 히스토그램,
            'dominant_colors': 주요 색상 리스트,
            'color_ranges': HSV 범위
        }
    """
    print("Extracting object color profile...")
    
    object_paths = list(object_images_dir.glob("*.png"))[:num_samples]
    
    if not object_paths:
        print(f"⚠️ No object images found in {object_images_dir}")
        # 기본값 반환 (갈색 원숭이 가정)
        return {
            'avg_hue': 10,  # 갈색/주황색
            'avg_sat': 150,
            'hue_range': (0, 25),  # 빨강~주황~갈색
            'sat_range': (100, 200),
            'dominant_hues': [10],
            'dominant_sats': [150]
        }
    
    hsv_histograms = []
    dominant_hues = []
    dominant_sats = []
    
    for obj_path in object_paths:
        img = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
            
        # RGBA 체크
        if len(img.shape) != 3 or img.shape[2] != 4:
            # RGB 이미지면 전체를 마스크로
            if len(img.shape) == 3:
                mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            else:
                continue
        else:
            # 알파 채널로 마스크 생성
            alpha = img[:, :, 3]
            mask = (alpha > 128).astype(np.uint8) * 255
            
            # RGB to HSV
            rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
        
        # 마스크 영역의 히스토그램
        hist_h = cv2.calcHist([rgb], [0], mask, [180], [0, 180])
        hist_s = cv2.calcHist([rgb], [1], mask, [256], [0, 256])
        
        hsv_histograms.append((hist_h, hist_s))
        
        # 주요 색상 (마스크 영역만)
        masked_pixels = rgb[mask > 0]
        if len(masked_pixels) > 0:
            # Hue, Saturation 평균
            dominant_hues.append(np.mean(masked_pixels[:, 0]))
            dominant_sats.append(np.mean(masked_pixels[:, 1]))
    
    # 데이터가 충분하지 않으면 기본값
    if len(dominant_hues) == 0:
        print(f"⚠️ No valid object colors found, using default (brown)")
        return {
            'avg_hue': 10,
            'avg_sat': 150,
            'hue_range': (0, 25),
            'sat_range': (100, 200),
            'dominant_hues': [10],
            'dominant_sats': [150]
        }
    
    # 평균 색상 범위 계산
    avg_hue = np.mean(dominant_hues)
    avg_sat = np.mean(dominant_sats)
    std_hue = np.std(dominant_hues)
    std_sat = np.std(dominant_sats)
    
    color_profile = {
        'avg_hue': avg_hue,
        'avg_sat': avg_sat,
        'hue_range': (max(0, avg_hue - std_hue * 2), min(180, avg_hue + std_hue * 2)),
        'sat_range': (max(0, avg_sat - std_sat * 2), min(255, avg_sat + std_sat * 2)),
        'dominant_hues': dominant_hues,
        'dominant_sats': dominant_sats
    }
    
    print(f"  Object color profile:")
    print(f"    Avg Hue: {avg_hue:.1f} (range: {color_profile['hue_range']})")
    print(f"    Avg Saturation: {avg_sat:.1f} (range: {color_profile['sat_range']})")
    
    return color_profile

def calculate_color_similarity(bg_image: np.ndarray, color_profile: dict) -> float:
    """
    배경 이미지와 객체 색상의 유사도 계산 (0~1, 높을수록 유사)
    """
    # HSV 변환
    hsv = cv2.cvtColor(bg_image, cv2.COLOR_BGR2HSV)
    
    # 객체 색상 범위
    hue_min, hue_max = color_profile['hue_range']
    sat_min, sat_max = color_profile['sat_range']
    
    # 유사한 색상 픽셀 비율
    mask = cv2.inRange(hsv, 
                      np.array([hue_min, sat_min, 50]),
                      np.array([hue_max, sat_max, 255]))
    
    similarity = np.sum(mask > 0) / (bg_image.shape[0] * bg_image.shape[1])
    
    return similarity

def find_similar_backgrounds(
    backgrounds_dir: Path,
    color_profile: dict,
    num_negatives: int = 200,
    similarity_threshold: float = 0.1  # 10% 이상 유사 색상
) -> List[Path]:
    """
    객체와 색상이 비슷한 배경 찾기
    """
    print(f"\nFinding similar backgrounds (threshold: {similarity_threshold:.1%})...")
    
    # 모든 배경 카테고리
    categories = [d for d in backgrounds_dir.iterdir() if d.is_dir()]
    
    similar_bgs = []
    
    for cat in categories:
        bg_images = list(cat.glob("*.jpg"))[:400]  # 카테고리당 최대 400장
        
        for bg_path in bg_images:
            try:
                # 배경 로드
                bg = cv2.imread(str(bg_path))
                if bg is None:
                    continue
                
                # 640x640으로 리사이즈 (빠른 계산)
                bg_small = cv2.resize(bg, (640, 640))
                
                # 유사도 계산
                similarity = calculate_color_similarity(bg_small, color_profile)
                
                if similarity >= similarity_threshold:
                    print("Similar background found:",len(similar_bgs))
                    similar_bgs.append((bg_path, similarity))
                    
                if len(similar_bgs) >= num_negatives * 2:
                    print("Reached sufficient similar backgrounds, stopping search.")
                    break
                    
            except Exception as e:
                continue
        
        if len(similar_bgs) >= num_negatives * 2:
            print("Reached sufficient similar backgrounds, stopping search.")
            break
    
    # 유사도 높은 순으로 정렬
    similar_bgs.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 num_negatives개 선택
    selected = [path for path, sim in similar_bgs[:num_negatives]]
    
    print(f"  Found {len(selected)} similar backgrounds")
    if selected:
        print(f"  Similarity range: {similar_bgs[0][1]:.1%} ~ {similar_bgs[min(len(similar_bgs)-1, num_negatives-1)][1]:.1%}")
    
    return selected

def add_negative_samples(
    output_dir: Path,
    negative_bg_paths: List[Path],
    upscale_factor: int = 4
):
    """
    Negative sample을 학습 데이터에 추가
    """
    print(f"\nAdding {len(negative_bg_paths)} negative samples...")
    
    images_train = output_dir / "images" / "train"
    labels_train = output_dir / "labels" / "train"
    
    images_train.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)
    
    for idx, bg_path in enumerate(negative_bg_paths):
        try:
            # 배경 로드
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue
            
            # Upscale (4x, 256 → 1024)
            h, w = bg.shape[:2]
            new_h, new_w = h * upscale_factor, w * upscale_factor
            bg_upscaled = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 저장
            filename = f"negative_{idx:04d}.png"
            cv2.imwrite(str(images_train / filename), bg_upscaled)
            
            # 빈 라벨 파일 (객체 없음)
            (labels_train / f"negative_{idx:04d}.txt").touch()
            
            if (idx + 1) % 50 == 0:
                print(f"  Added {idx + 1}/{len(negative_bg_paths)} negative samples")
                
        except Exception as e:
            print(f"  Error processing {bg_path}: {e}")
            continue
    
    print(f"\n✅ Added {len(negative_bg_paths)} negative samples")
    print(f"  Images: {images_train}")
    print(f"  Labels: {labels_train} (empty files)")

def main():
    # 경로 설정
    object_images_dir = Path("output/dataset/masked_frames")
    backgrounds_dir = Path("dataset/train")
    output_dir = Path("output/dataset/result")
    
    # 1. 객체 색상 프로파일 추출
    color_profile = extract_object_color_profile(object_images_dir, num_samples=50)
    
    # 2. 유사한 배경 찾기
    similar_bgs = find_similar_backgrounds(
        backgrounds_dir,
        color_profile,
        num_negatives=200,  # 전체 데이터의 ~12%
        similarity_threshold=0.1  # 5% 이상 유사 (낮춤)
    )
    
    # 3. Negative samples 추가
    if similar_bgs:
        add_negative_samples(output_dir, similar_bgs, upscale_factor=4)
    else:
        print("⚠️ No similar backgrounds found")

if __name__ == "__main__":
    main()
