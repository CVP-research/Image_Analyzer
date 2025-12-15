import os
import glob
import random
import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 설정 파라미터
# ==========================================
SOURCE_DIR = 'dataset/val'  # 원본 이미지가 있는 폴더
DEST_DIR = '/home/rocknroll1397/Image_Analyzer/output/dataset/libcom8' # 저장될 폴더
TILE_SIZE = 256             # 개별 타일 크기 (256x256)
GRID_SIZE = 4               # 격자 크기 (4x4 = 16장)
TARGET_RGB = (160, 110, 60) # 목표 갈색 (R, G, B)
TOP_N = 32000                # 분석할 최대 이미지 수 (16장씩 묶으므로 넉넉하게 잡음)
# 예상 결과물 수 = TOP_N / 16 (예: 3200장 -> 200장의 배경 생성)

def calculate_color_distance(image_path, target_rgb):
    """이미지 평균 색상과 타겟 색상 간 거리 계산"""
    try:
        img = cv2.imread(image_path)
        if img is None: return float('inf')
        
        # 속도 향상을 위해 이미지를 작게 줄여서 평균 계산
        img_small = cv2.resize(img, (64, 64)) 
        avg_bgr = np.mean(img_small, axis=(0, 1))
        target_bgr = np.array(target_rgb[::-1])
        
        distance = np.linalg.norm(avg_bgr - target_bgr)
        return distance
    except:
        return float('inf')

def pick_top_brown_images(image_folder, target_rgb, top_n):
    """갈색과 유사한 이미지 경로 리스트 반환"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, '**', ext), recursive=True))

    print(f"📂 총 {len(image_files)}장의 이미지를 찾았습니다. 색상 분석 시작...")
    
    scored_images = []
    # 셔플을 먼저 해서 매번 다른 이미지가 분석되도록 함 (전체가 너무 많을 경우 대비)
    random.shuffle(image_files)
    
    for img_path in tqdm(image_files, desc="Analyzing Colors"):
        score = calculate_color_distance(img_path, target_rgb)
        if score != float('inf'):
            scored_images.append((score, img_path))
            
    # 색상 유사도 순 정렬
    scored_images.sort(key=lambda x: x[0])
    
    # 상위 N개 선택
    selected_paths = [path for score, path in scored_images[:top_n]]
    
    print(f"\n✅ 상위 {len(selected_paths)}장 선택 완료 (상위 5개 점수: {[round(s[0], 2) for s in scored_images[:5]]})")
    return selected_paths

def create_tiled_image(image_paths, tile_size=256):
    """
    16개의 이미지 경로를 받아 4x4 (1024x1024) 이미지로 병합
    """
    if len(image_paths) < 16:
        return None

    rows = []
    idx = 0
    
    # 4행
    for _ in range(4):
        cols = []
        # 4열
        for _ in range(4):
            path = image_paths[idx]
            img = cv2.imread(path)
            
            # 읽기 실패 시 검은색 빈 이미지로 대체 (에러 방지)
            if img is None:
                img = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            
            # 타일 크기로 리사이징 (256x256)
            img = cv2.resize(img, (tile_size, tile_size))
            cols.append(img)
            idx += 1
        
        # 가로로 붙이기
        rows.append(np.hstack(cols))
    
    # 세로로 붙이기 (최종 1024x1024)
    final_img = np.vstack(rows)
    return final_img

def main():
    # 1. 저장 경로 생성
    train_dir = os.path.join(DEST_DIR, 'images', 'train')
    val_dir = os.path.join(DEST_DIR, 'images', 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 2. 색상 기반 이미지 선택
    selected_images = pick_top_brown_images(SOURCE_DIR, TARGET_RGB, TOP_N)
    
    # 16장씩 묶어야 하므로, 남는 자투리는 버림
    total_available = len(selected_images)
    cutoff = (total_available // 16) * 16
    selected_images = selected_images[:cutoff]
    
    if len(selected_images) < 16:
        print("❌ 이미지가 부족하여 타일을 만들 수 없습니다. (최소 16장 필요)")
        return

    # 3. 섞기 (비슷한 색끼리 뭉치지 않고 다양한 텍스처가 섞이도록)
    print("🔀 타일 구성을 위해 이미지를 섞습니다...")
    random.shuffle(selected_images)

    # 4. 16개씩 그룹화 (Chunking)
    chunks = [selected_images[i:i + 16] for i in range(0, len(selected_images), 16)]
    print(f"📦 총 {len(chunks)}개의 타일 배경을 생성할 예정입니다.")

    # 5. Train/Val 분할 (배경 이미지 단위로 분할)
    random.shuffle(chunks) # 그룹 자체도 순서 섞기
    train_split = int(0.8 * len(chunks))
    
    train_chunks = chunks[:train_split]
    val_chunks = chunks[train_split:]

    # 6. 생성 및 저장 함수
    def process_and_save(chunks, save_dir, prefix):
        count = 0
        for chunk in tqdm(chunks, desc=f"Saving to {prefix}"):
            # 16장 합치기
            tiled_img = create_tiled_image(chunk, tile_size=TILE_SIZE)
            
            if tiled_img is not None:
                filename = f"{prefix}_tiled_{count:04d}.jpg"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, tiled_img)
                count += 1
        return count

    # 실행
    print("\n🚀 Train 데이터 생성 중...")
    train_count = process_and_save(train_chunks, train_dir, "train")
    
    print("\n🚀 Val 데이터 생성 중...")
    val_count = process_and_save(val_chunks, val_dir, "val")

    print("\n" + "="*50)
    print(f"✅ 작업 완료!")
    print(f"📂 저장 경로: {DEST_DIR}")
    print(f"📊 생성된 배경 수 -> Train: {train_count}장 / Val: {val_count}장")
    print(f"🧱 해상도: {TILE_SIZE*GRID_SIZE}x{TILE_SIZE*GRID_SIZE} (1024x1024)")
    print("="*50)

if __name__ == "__main__":
    main()