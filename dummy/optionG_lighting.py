"""
Option G: 밝기 + 색온도 조정 (고유 색상 유지)

수정 사항:
- 배경의 밝기 분석 → 객체 밝기 조정
- 배경의 색온도 분석 → 객체 색온도 살짝 조정
- HSV 색상(Hue)은 유지 → 빨간 인형은 빨간색 유지
- NORMAL_CLONE으로 테두리 처리

출력: output/dataset_optionG/
"""

import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image

# 상위 디렉토리의 main.py 함수들을 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    find_suitable_backgrounds,
    get_all_objects,
    BASE_DIR,
    INPUT_DIR,
    MASKED_FRAMES_DIR
)

# Option G 전용 출력 디렉토리
OUTPUT_DIR_G = BASE_DIR / "output" / "dataset_optionG"
OUTPUT_DIR_G.mkdir(parents=True, exist_ok=True)


def adjust_lighting_and_temperature(
    obj_cv: np.ndarray, 
    bg_cv: np.ndarray, 
    mask_bin: np.ndarray,
    brightness_strength: float = 0.3,
    temperature_strength: float = 0.2
) -> np.ndarray:
    """
    배경의 밝기와 색온도를 분석하여 객체에 적용 (고유 색상 유지)
    
    Args:
        obj_cv: 객체 이미지 (BGR)
        bg_cv: 배경 이미지 (BGR)
        mask_bin: 객체 마스크
        brightness_strength: 밝기 조정 강도 (0.3 = 30% 배경 밝기에 맞춤)
        temperature_strength: 색온도 조정 강도 (0.2 = 20% 배경 색온도에 맞춤)
    
    Returns:
        조정된 객체 이미지 (BGR)
    """
    # 1. 배경 밝기 분석 (Lab의 L 채널)
    bg_lab = cv2.cvtColor(bg_cv, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_brightness = np.mean(bg_lab[:, :, 0])  # L 채널 평균
    
    # 2. 배경 색온도 분석 (R-B 차이)
    bg_r = np.mean(bg_cv[:, :, 2])  # Red
    bg_b = np.mean(bg_cv[:, :, 0])  # Blue
    bg_temperature = (bg_r - bg_b) / 255.0  # -1(차가움) ~ 1(따뜻함)
    
    # 3. 객체를 HSV로 변환 (Hue 유지용)
    obj_hsv = cv2.cvtColor(obj_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
    obj_lab = cv2.cvtColor(obj_cv, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # 객체의 현재 밝기
    obj_mask_bool = mask_bin > 0
    if np.sum(obj_mask_bool) > 0:
        obj_brightness = np.mean(obj_lab[:, :, 0][obj_mask_bool[:obj_lab.shape[0], :obj_lab.shape[1]]])
    else:
        obj_brightness = np.mean(obj_lab[:, :, 0])
    
    # 4. 밝기 조정 (Lab의 L 채널만)
    brightness_diff = bg_brightness - obj_brightness
    obj_lab[:, :, 0] = np.clip(
        obj_lab[:, :, 0] + brightness_diff * brightness_strength,
        0, 255
    )
    
    # Lab → BGR 변환
    obj_adjusted = cv2.cvtColor(obj_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # 5. 색온도 조정 (R, B 채널만 살짝)
    if abs(bg_temperature) > 0.05:  # 배경이 확실히 따뜻하거나 차가울 때만
        # 따뜻한 배경 → R 증가, B 감소
        # 차가운 배경 → R 감소, B 증가
        temperature_shift = bg_temperature * temperature_strength * 50  # -15 ~ 15 정도
        
        obj_adjusted[:, :, 2] = np.clip(
            obj_adjusted[:, :, 2].astype(np.float32) + temperature_shift,
            0, 255
        ).astype(np.uint8)  # Red
        
        obj_adjusted[:, :, 0] = np.clip(
            obj_adjusted[:, :, 0].astype(np.float32) - temperature_shift,
            0, 255
        ).astype(np.uint8)  # Blue
    
    return obj_adjusted


def blend_object_optionG(
    obj_image: Image.Image, 
    bg_image: Image.Image,
    brightness_strength: float = 0.3,
    temperature_strength: float = 0.2,
    erode_iterations: int = 1
) -> Image.Image:
    """
    Option G: 밝기 + 색온도 조정 + 순수 알파 블렌딩
    
    Args:
        brightness_strength: 밝기 조정 강도
        temperature_strength: 색온도 조정 강도
        erode_iterations: 마스크 침식 반복 횟수 (테두리 제거용)
    """
    obj_rgba = obj_image.convert("RGBA")
    bg_rgba = bg_image.convert("RGBA")

    obj_cv = cv2.cvtColor(np.array(obj_rgba), cv2.COLOR_RGBA2BGRA)
    bg_cv = cv2.cvtColor(np.array(bg_rgba), cv2.COLOR_RGBA2BGRA)

    bg_h, bg_w = bg_cv.shape[:2]
    obj_h, obj_w = obj_cv.shape[:2]
    
    # 객체 리사이즈
    if obj_w > bg_w or obj_h > bg_h:
        scale = min(bg_w / obj_w, bg_h / obj_h) * 0.9
        obj_cv = cv2.resize(obj_cv, (int(obj_w * scale), int(obj_h * scale)), interpolation=cv2.INTER_AREA)
        obj_h, obj_w = obj_cv.shape[:2]

    # 마스크 생성
    if obj_cv.shape[2] == 4:
        mask = obj_cv[:, :, 3]
    else:
        mask = cv2.cvtColor(obj_cv, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    
    # 마스크 침식: 테두리 픽셀 제거 (흰색/검은색 경계선 문제 해결)
    if erode_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_bin = cv2.erode(mask_bin, kernel, iterations=erode_iterations)
    
    # 알파 채널 약간만 부드럽게 (경계를 선명하게 유지)
    mask_float = mask_bin.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (3, 3), 0.5)

    # 밝기 + 색온도 조정
    obj_adjusted = adjust_lighting_and_temperature(
        obj_cv[:, :, :3], 
        bg_cv[:, :, :3], 
        (mask_float * 255).astype(np.uint8),
        brightness_strength=brightness_strength,
        temperature_strength=temperature_strength
    )
    
    # 순수 알파 블렌딩 (Poisson 제거)
    result = bg_cv[:, :, :3].copy()
    y_offset = (bg_h - obj_h) // 2
    x_offset = (bg_w - obj_w) // 2
    
    for i in range(obj_h):
        for j in range(obj_w):
            alpha = mask_float[i, j]
            if alpha > 0.01:  # 거의 투명한 픽셀 무시
                by, bx = y_offset + i, x_offset + j
                if 0 <= by < bg_h and 0 <= bx < bg_w:
                    result[by, bx] = (
                        obj_adjusted[i, j] * alpha + 
                        result[by, bx] * (1 - alpha)
                    ).astype(np.uint8)
    
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def composite_naturally_optionG(
    objects: List[Tuple[Image.Image, Dict]],
    backgrounds: List[Dict],
    overlay_scale: float = 0.8,
    objects_per_bg: int = 2,
    brightness_strength: float = 0.3,
    temperature_strength: float = 0.2,
    erode_iterations: int = 1
) -> List[Path]:
    """
    Option G 합성: 밝기 + 색온도 조정
    """
    print(f"\n[Option G] Compositing with lighting and temperature adjustment...")
    print(f"  Backgrounds: {len(backgrounds)}")
    print(f"  Objects per background: {objects_per_bg}")
    print(f"  Brightness strength: {brightness_strength:.1%}")
    print(f"  Temperature strength: {temperature_strength:.1%}")
    print(f"  Edge erosion iterations: {erode_iterations} (remove white border)")
    print(f"  Method: Adjust brightness & warmth while preserving object color")
    
    output_paths = []
    
    for bg_idx, bg_info in enumerate(backgrounds):
        selected_objects = random.sample(objects, min(objects_per_bg, len(objects)))
        
        for obj_idx, (obj_image, obj_meta) in enumerate(selected_objects):
            try:
                # 원본 객체로 직접 합성
                blended_img = blend_object_optionG(
                    obj_image, 
                    bg_info["bg_image"],
                    brightness_strength=brightness_strength,
                    temperature_strength=temperature_strength,
                    erode_iterations=erode_iterations
                )
                
                output_filename = f"optionG_bg{bg_idx}_obj{obj_idx}_{bg_info['segment_label'].replace(' ', '_')}.png"
                output_path = OUTPUT_DIR_G / output_filename
                blended_img.save(output_path)
                output_paths.append(output_path)
                
                print(f"    ✓ Saved: {output_filename}")
            
            except Exception as e:
                print(f"    ✗ Error: bg{bg_idx} + obj{obj_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n[Option G] Generated {len(output_paths)} images")
    print(f"Output: {OUTPUT_DIR_G}")
    return output_paths


def main():
    print("=" * 70)
    print("Option G: Lighting & Temperature Adjustment")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR_G}")
    print("Strategy: Adjust brightness & warmth, preserve object color")
    print()
    
    backgrounds = find_suitable_backgrounds(
        object_category="monkey doll",
        semantic_locations=["shelf", "bed", "couch", "table", "toy box"],
        broad_categories=["home", "indoor", "living room", "bedroom"],
        max_backgrounds=5,
        similarity_threshold=0.8,
        max_workers=5
    )
    
    objects = get_all_objects(
        original_input_dir=INPUT_DIR,
        masked_frames_dir=MASKED_FRAMES_DIR
    )
    results = composite_naturally_optionG(
        objects=objects,
        backgrounds=backgrounds,
        overlay_scale=0.8,
        objects_per_bg=2,
        brightness_strength=0.3,
        temperature_strength=0.2,
        erode_iterations=1  # 테두리 1픽셀 제거
    )
    
    print("\n" + "=" * 70)
    print(f"✓ Option G completed: {len(results)} images")
    print("=" * 70)


if __name__ == "__main__":
    main()
