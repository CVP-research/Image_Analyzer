import os
import glob
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm

def calculate_color_distance(image_path, target_rgb=(139, 69, 19)):
    """
    이미지의 평균 색상과 타겟 색상(갈색) 사이의 거리를 계산하는 함수
    
    Args:
        image_path (str): 이미지 파일 경로
        target_rgb (tuple): 목표로 하는 갈색의 (R, G, B) 값. 
                            기본값은 'SaddleBrown' (139, 69, 19)
    
    Returns:
        float: 색상 거리 점수 (0에 가까울수록 타겟 색상과 비슷함)
               이미지를 읽지 못하면 무한대(inf) 반환
    """
    # 1. 이미지 읽기 (OpenCV는 기본적으로 BGR 순서임)
    img = cv2.imread(image_path)
    
    if img is None:
        return float('inf') # 읽기 실패 시 제외

    # 2. 이미지의 평균 색상 계산 (BGR 순서)
    # axis=(0, 1)은 높이와 너비 축을 따라 평균을 낸다는 뜻
    avg_bgr = np.mean(img, axis=(0, 1))

    # 3. 타겟 색상을 RGB -> BGR로 변환 (비교를 위해)
    target_bgr = np.array(target_rgb[::-1]) # (R,G,B) -> (B,G,R)

    # 4. 유클리드 거리(Euclidean Distance) 계산
    # 두 색상 벡터 사이의 거리를 구함
    distance = np.linalg.norm(avg_bgr - target_bgr)

    return distance

def pick_top_brown_images(image_folder, top_n=100):
    """
    주어진 폴더에서 특정 갈색과 가장 유사한 이미지 상위 N개를 선택합니다.
    """
    # 1. 이미지 파일 리스트 가져오기 (jpg, png 등, 하위 폴더 포함)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))

    # 2. 모든 이미지에 대해 '갈색 점수' 계산
    print(f"총 {len(image_files)}장의 이미지를 분석 중...")
    
    scored_images = []
    for img_path in tqdm(image_files, desc="Analyzing image colors"):
        # 갈색(나무/박스 색) RGB 예시: (160, 110, 60)
        score = calculate_color_distance(img_path, target_rgb=(160, 110, 60))
        if score != float('inf'):
            scored_images.append((score, img_path))

    # 3. 점수가 낮은 순(거리가 가까운 순)으로 정렬
    scored_images.sort(key=lambda x: x[0])

    # 4. 상위 N개 선택
    selected_images = scored_images[:top_n]
    
    print(f"\n[결과] 갈색에 가장 가까운 상위 {len(selected_images)}장:")
    # 상세 출력을 위해 상위 10개만 프린트
    for score, path in selected_images[:10]:
        print(f"점수: {score:.2f} | 경로: {os.path.basename(path)}")
    if len(selected_images) > 10:
        print("...")

    return [path for score, path in selected_images]

def copy_images(image_list, destination_folder):
    """지정된 폴더로 이미지 목록을 복사합니다."""
    os.makedirs(destination_folder, exist_ok=True)
    for img_path in tqdm(image_list, desc=f"Copying to {os.path.basename(destination_folder)}"):
        shutil.copy(img_path, destination_folder)

def main():
    """메인 실행 함수"""
    # Directories
    source_dir = 'dataset/val'
    dest_dir = '/home/rocknroll1397/Image_Analyzer/output/dataset/libcom8'
    train_dir = os.path.join(dest_dir, 'images', 'train')
    val_dir = os.path.join(dest_dir, 'images', 'val')

    # 1. 색상 기반으로 700장 선택
    print("Step 1: Selecting images based on color similarity...")
    selected_images = pick_top_brown_images(source_dir, top_n=700)
    
    if not selected_images:
        print("이미지를 선택하지 못했습니다. 스크립트를 종료합니다.")
        return

    # 2. Train/Val 분할을 위해 셔플
    print("\nStep 2: Shuffling images for train/val split...")
    random.shuffle(selected_images)

    # 3. 80/20 비율로 분할
    train_split = int(0.8 * len(selected_images))
    train_images = selected_images[:train_split]
    val_images = selected_images[train_split:]

    # 4. 이미지 복사
    print("\nStep 3: Copying images to destination...")
    copy_images(train_images, train_dir)
    copy_images(val_images, val_dir)

    print("\nDone!")
    print(f"Copied {len(train_images)} images to {train_dir}")
    print(f"Copied {len(val_images)} images to {val_dir}")


if __name__ == "__main__":
    main()