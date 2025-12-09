"""
기존 masked frame을 compact하게 crop하는 스크립트

투명 픽셀을 제거하고 객체만 있는 부분으로 잘라서 저장
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# 상위 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import MASKED_FRAMES_DIR


def crop_transparent_pixels(image_path: Path, output_dir: Path) -> Path:
    """
    이미지에서 투명 픽셀을 제거하고 객체만 crop
    
    Args:
        image_path: 원본 이미지 경로
        output_dir: 출력 디렉토리
    
    Returns:
        저장된 파일 경로
    """
    # 이미지 로드
    img_pil = Image.open(image_path)
    
    # RGBA로 변환
    if img_pil.mode != 'RGBA':
        print(f"  ⚠ {image_path.name} is not RGBA, skipping")
        return image_path
    
    # numpy array로 변환
    img_array = np.array(img_pil)
    alpha = img_array[:, :, 3]
    
    # 불투명 픽셀 찾기
    ys, xs = np.where(alpha > 0)
    
    if len(xs) == 0 or len(ys) == 0:
        print(f"  ⚠ {image_path.name} has no visible pixels, skipping")
        return image_path
    
    h, w = alpha.shape
    gap_threshold = 10
    
    # 왼쪽부터 스캔: 열 전체가 투명한 열이 10개 이상 연속되면 그 앞은 노이즈
    x_min = 0
    empty_count = 0
    for x in range(w):
        col = alpha[:, x]
        if np.all(col == 0):  # 이 열 전체가 투명
            empty_count += 1
        else:  # 불투명 픽셀 발견
            if empty_count >= gap_threshold:
                # 10개 이상 빈 열 후 처음 나타난 픽셀 → 여기서부터 시작
                x_min = x
                break
            else:
                # 10개 미만이면 아직 외곽 노이즈일 수 있음
                empty_count = 0
    
    # 오른쪽부터 스캔
    x_max = w - 1
    empty_count = 0
    for x in range(w - 1, -1, -1):
        col = alpha[:, x]
        if np.all(col == 0):
            empty_count += 1
        else:
            if empty_count >= gap_threshold:
                x_max = x
                break
            else:
                empty_count = 0
    
    # 위쪽부터 스캔: 행 전체가 투명한 행이 10개 이상 연속되면 그 위는 노이즈
    y_min = 0
    empty_count = 0
    for y in range(h):
        row = alpha[y, :]
        if np.all(row == 0):
            empty_count += 1
        else:
            if empty_count >= gap_threshold:
                y_min = y
                break
            else:
                empty_count = 0
    
    # 아래쪽부터 스캔
    y_max = h - 1
    empty_count = 0
    for y in range(h - 1, -1, -1):
        row = alpha[y, :]
        if np.all(row == 0):
            empty_count += 1
        else:
            if empty_count >= gap_threshold:
                y_max = y
                break
            else:
                empty_count = 0
    
    # Crop
    img_cropped = img_array[y_min:y_max+1, x_min:x_max+1]
    
    # 크기 비교
    original_h, original_w = img_array.shape[:2]
    cropped_h, cropped_w = img_cropped.shape[:2]
    
    # PIL로 변환
    img_pil_cropped = Image.fromarray(img_cropped)
    
    # 저장 (새 디렉토리에 같은 파일명으로)
    output_path = output_dir / image_path.name
    img_pil_cropped.save(output_path)
    
    # 크기가 같으면 이미 crop됨
    if original_h == cropped_h and original_w == cropped_w:
        print(f"  ✓ {image_path.name}: already cropped (no change)")
    else:
        reduction = (1 - (cropped_h * cropped_w) / (original_h * original_w)) * 100
        print(f"  ✓ {image_path.name}: {original_h}x{original_w} → {cropped_h}x{cropped_w} ({reduction:.1f}% reduced)")
    
    return output_path


def crop_all_masked_frames(masked_dir: Path = None, output_dir: Path = None):
    """
    디렉토리 내 모든 masked frame을 crop하여 새 디렉토리에 저장
    
    Args:
        masked_dir: masked frames 디렉토리 (기본: MASKED_FRAMES_DIR)
        output_dir: 출력 디렉토리 (기본: masked_frames_cropped)
    """
    if masked_dir is None:
        masked_dir = MASKED_FRAMES_DIR
    
    if output_dir is None:
        output_dir = masked_dir.parent / "masked_frames_cropped"
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Cropping Existing Masked Frames")
    print("=" * 60)
    print(f"Input directory: {masked_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 이미지 파일 찾기
    image_files = list(masked_dir.glob("*.png")) + list(masked_dir.glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"No images found in {masked_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print()
    
    # 각 이미지 처리
    processed = 0
    skipped = 0
    
    for img_path in image_files:
        try:
            result_path = crop_transparent_pixels(img_path, output_dir)
            processed += 1
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {e}")
            skipped += 1
            continue
    
    print()
    print("=" * 60)
    print(f"Completed! Processed: {processed}, Skipped: {skipped}")
    print("=" * 60)


def main():
    """
    메인 엔트리 포인트
    """
    output_dir = MASKED_FRAMES_DIR.parent / "masked_frames_cropped"
    
    print("This script will crop all masked frames to remove transparent pixels.")
    print(f"Input directory: {MASKED_FRAMES_DIR}")
    print(f"Output directory: {output_dir}")
    print()
    print("Original files will NOT be modified.")
    print()
    
    # 확인
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # 실행
    crop_all_masked_frames()
    
    print(f"\n✓ All masked frames have been cropped and saved to:")
    print(f"  {output_dir}")
    print("\nIf you like the results:")
    print(f"  1. Delete old directory: rm -rf {MASKED_FRAMES_DIR}")
    print(f"  2. Rename new directory: mv {output_dir} {MASKED_FRAMES_DIR}")


if __name__ == "__main__":
    main()
