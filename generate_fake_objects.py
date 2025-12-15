"""
가짜 객체 생성기 - 객체의 색상과 모양을 추출해서 비슷한 blob 생성

실제 객체가 아닌, 평균 색상과 대략적인 모양만 가진 가짜 객체를 생성
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import random


def extract_color_and_shape(obj_img: Image.Image) -> dict:
    """
    객체의 평균 색상과 모양(contour) 추출
    """
    if obj_img.mode != 'RGBA':
        obj_img = obj_img.convert('RGBA')
    
    obj_array = np.array(obj_img)
    alpha = obj_array[:, :, 3]
    
    if alpha.max() == 0:
        return None
    
    # 마스크 생성
    mask = (alpha > 10).astype(np.uint8) * 255
    
    # 평균 색상 추출 (BGR)
    obj_bgr = cv2.cvtColor(obj_array[:, :, :3], cv2.COLOR_RGB2BGR)
    mean_color = cv2.mean(obj_bgr, mask=mask)[:3]  # (B, G, R)
    
    # 주요 색상 여러 개 추출 (K-means)
    pixels = obj_bgr[mask > 0].reshape(-1, 3)
    if len(pixels) > 100:
        # 샘플링
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
        
        # K-means로 5개 주요 색상
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = min(5, len(pixels))
        _, _, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        color_palette = centers.astype(np.uint8)
    else:
        color_palette = np.array([mean_color], dtype=np.uint8)
    
    # 외곽선 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    main_contour = max(contours, key=cv2.contourArea)
    
    # 외곽선 단순화 (덜 단순화 - epsilon 감소)
    epsilon = 0.005 * cv2.arcLength(main_contour, True)  # 0.02 -> 0.005 (더 자세하게)
    approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    
    return {
        'mean_color': mean_color,
        'color_palette': color_palette,
        'contour': approx_contour
    }


def generate_fake_object(obj_img: Image.Image) -> Image.Image:
    """
    객체의 평균 색상과 모양으로 가짜 blob 생성
    """
    features = extract_color_and_shape(obj_img)
    if features is None:
        return None
    
    mean_color = features['mean_color']
    color_palette = features['color_palette']
    contour = features['contour']
    
    # 캔버스 크기 (원본과 동일)
    h, w = obj_img.height, obj_img.width
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    
    # 1. 외곽선 변형 (노이즈 추가)
    contour_float = contour.astype(np.float32).reshape(-1, 2)
    
    # 랜덤 노이즈로 모양 왜곡 (더 작게)
    noise = np.random.normal(0, 1.5, contour_float.shape)  # 3 -> 1.5 (작은 노이즈)
    contour_deformed = contour_float + noise
    contour_deformed = contour_deformed.astype(np.int32).reshape(-1, 1, 2)
    
    # 회전
    angle = random.uniform(-30, 30)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 반전 (50% 확률)
    if random.random() < 0.5:
        contour_deformed[:, :, 0] = w - contour_deformed[:, :, 0]
    
    # 회전 적용
    ones = np.ones((contour_deformed.shape[0], 1, 1))
    contour_homogeneous = np.concatenate([contour_deformed, ones], axis=2)
    contour_rotated = np.matmul(M, contour_homogeneous.reshape(-1, 3).T).T
    contour_final = contour_rotated.astype(np.int32).reshape(-1, 1, 2)
    
    # 2. 색상 변형 (평균 색상에서 편차 - 줄임)
    color_variation = np.random.randint(-20, 20, 3)  # -40~40 -> -20~20
    fake_color = np.clip(np.array(mean_color) + color_variation, 0, 255).astype(np.uint8)
    
    # 3. blob 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour_final], -1, 255, -1)
    
    # 베이스 색상으로 채우기
    for c in range(3):
        canvas[:, :, c] = fake_color[c]
    
    # 그라디언트 효과 (약간의 입체감)
    gradient_y = np.linspace(0.85, 1.15, h).reshape(-1, 1)
    gradient_x = np.linspace(0.9, 1.1, w).reshape(1, -1)
    gradient = gradient_y * gradient_x
    
    for c in range(3):
        channel = canvas[:, :, c].astype(np.float32) * gradient
        canvas[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
    
    # 디테일 1: 객체의 주요 색상들로 패치 추가
    num_patches = random.randint(5, 10)
    for _ in range(num_patches):
        patch_color = color_palette[random.randint(0, len(color_palette)-1)]
        patch_color_shifted = np.clip(patch_color.astype(np.int16) + random.randint(-15, 15), 0, 255).astype(np.uint8)  # -25~25 -> -15~15
        
        patch_x = random.randint(0, w-1)
        patch_y = random.randint(0, h-1)
        patch_size = random.randint(15, 40)
        
        # 불규칙한 모양의 패치
        for _ in range(random.randint(2, 4)):
            offset_x = random.randint(-patch_size//2, patch_size//2)
            offset_y = random.randint(-patch_size//2, patch_size//2)
            radius = random.randint(patch_size//3, patch_size//2)
            cv2.circle(canvas, (patch_x + offset_x, patch_y + offset_y), radius, 
                      tuple(patch_color_shifted.tolist()), -1)
    
    # 디테일 2: 노이즈 (텍스처 느낌)
    noise = np.random.randint(-10, 10, (h, w, 3)).astype(np.int16)  # -15~15 -> -10~10
    for c in range(3):
        canvas[:, :, c] = np.clip(canvas[:, :, c].astype(np.int16) + noise[:, :, c], 0, 255).astype(np.uint8)
    
    # 디테일 3: 선/얼룩 (약하게)
    for _ in range(random.randint(2, 4)):
        line_color = color_palette[random.randint(0, len(color_palette)-1)]
        line_color_shifted = np.clip(line_color.astype(np.int16) + random.randint(-10, 10), 0, 255).astype(np.uint8)  # -20~20 -> -10~10
        
        pt1 = (random.randint(0, w), random.randint(0, h))
        pt2 = (random.randint(0, w), random.randint(0, h))
        cv2.line(canvas, pt1, pt2, tuple(line_color_shifted.tolist()), random.randint(1, 2))
    
    # 알파 채널
    canvas[:, :, 3] = mask
    
    # 부드럽게
    canvas[:, :, 3] = cv2.GaussianBlur(canvas[:, :, 3], (3, 3), 0.5)
    
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA))


def generate_fake_objects_from_image(
    obj_img_path: Path,
    output_dir: Path,
    num_variations: int = 4
) -> List[Path]:
    """
    하나의 객체 이미지로부터 여러 가짜 객체 생성 (변형/왜곡)
    
    Args:
        obj_img_path: 원본 객체 이미지 경로
        output_dir: 출력 디렉토리
        num_variations: 생성할 변형 개수
    
    Returns:
        생성된 가짜 객체 이미지 경로 리스트
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 원본 이미지 로드
    obj_img = Image.open(obj_img_path)
    
    # 가짜 객체 생성
    fake_paths = []
    base_name = obj_img_path.stem
    
    for i in range(num_variations):
        fake_obj = generate_fake_object(obj_img)
        
        if fake_obj is None:
            print(f"  ✗ Failed to generate fake object from {obj_img_path.name}")
            continue
        
        fake_path = output_dir / f"fake_{base_name}_var{i:02d}.png"
        fake_obj.save(fake_path)
        fake_paths.append(fake_path)
    
    print(f"  ✓ Generated {len(fake_paths)} fake objects from {obj_img_path.name}")
    return fake_paths


def main():
    """
    메인 함수: 테스트용
    MASKED_FRAMES_DIR의 객체들로부터 가짜 객체 생성
    """
    from main import MASKED_FRAMES_DIR, BASE_DIR
    
    output_dir = BASE_DIR / "output" / "fake_objects"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 객체 이미지 찾기
    masked_images = list(MASKED_FRAMES_DIR.glob("*.png"))
    
    if not masked_images:
        print(f"No masked images found in {MASKED_FRAMES_DIR}")
        return
    
    print(f"Found {len(masked_images)} object images")
    print(f"Output directory: {output_dir}")
    print()
    
    # 각 객체당 4개의 가짜 객체 생성 (테스트)
    all_fake_objects = []
    
    for obj_path in masked_images[:3]:  # 테스트: 처음 3개만
        fake_paths = generate_fake_objects_from_image(
            obj_path,
            output_dir,
            num_variations=4
        )
        all_fake_objects.extend(fake_paths)
    
    print(f"\n✓ Generated {len(all_fake_objects)} fake objects")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
