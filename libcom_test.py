import os
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm
from libcom import image_harmonization

# ==========================================
# 1. 설정
# ==========================================
BG_ROOT = "./dataset"
OBJ_ROOT = "./output/dataset/masked_frames"
OUTPUT_ROOT = "./output/dataset/yolo_harmonized_data"

TARGET_SIZE = 1024        # 1024x1024 배경
NUM_BG_GROUPS = 600       # 만들 배경 수
IMGS_PER_GROUP = 16       # 그룹당 이미지 수
TOTAL_IMGS_NEEDED = NUM_BG_GROUPS * IMGS_PER_GROUP # 9600장 필요

REPEAT_PER_OBJECT = 5
VAL_SPLIT_RATIO = 0.2
CLASS_ID = 0

# ==========================================
# 2. 모델 로드
# ==========================================
print("🚀 모델 로드 중...")
try:
    net = image_harmonization.ImageHarmonizationModel(device=0)
except:
    net = image_harmonization.ImageHarmonizationModel(device=0)

# ==========================================
# 3. 핵심: 무식하게 읽어서 정렬하기 (제일 확실함)
# ==========================================
def create_sorted_background_pool(root_dir, total_needed):
    print(f"📂 1. 이미지 경로 {total_needed}개 수집 중...")
    
    paths = []
    # 폴더 뒤져서 경로만 수집 (빠름)
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                paths.append(os.path.join(subdir, f))
                if len(paths) >= total_needed: break # 필요한 만큼 모으면 탈출
        if len(paths) >= total_needed: break
    
    if len(paths) < 16:
        raise Exception("배경 이미지가 너무 적습니다.")

    print(f"🎨 2. 색상 계산 및 정렬 중 ({len(paths)}장)...")
    
    # (평균색상합계, 경로) 형태로 저장
    color_info = []
    for p in tqdm(paths, desc="Sorting Colors"):
        try:
            # 헤더만 읽고 1x1로 줄여서 평균색 빠르게 추출
            img = cv2.imread(p)
            if img is None: continue
            
            # 1x1 픽셀로 줄이면 그게 곧 평균색
            pixel = cv2.resize(img, (1, 1))
            mean_val = np.sum(pixel) # B+G+R 합계 (밝기/톤 유사도)
            color_info.append((mean_val, p))
        except:
            continue

    # ★ 핵심: 색상 값 기준으로 정렬 ★
    # 이러면 어두운 것 -> 밝은 것 순서로 쫘르륵 줄을 섭니다.
    # 옆에 있는 애들끼리는 색이 비슷할 수밖에 없습니다.
    color_info.sort(key=lambda x: x[0])
    
    sorted_paths = [x[1] for x in color_info]
    
    print("🧱 3. 타일 배경 생성 중...")
    ready_backgrounds = []
    
    # 정렬된 리스트를 16개씩 뚝뚝 끊어서 타일 생성
    for i in tqdm(range(0, len(sorted_paths), 16), desc="Tiling"):
        chunk = sorted_paths[i : i+16]
        if len(chunk) < 16: break # 자투리는 버림
        
        # 16장 합치기
        bg = create_1024_tiled_bg(chunk)
        ready_backgrounds.append(bg)
        
        if len(ready_backgrounds) >= NUM_BG_GROUPS:
            break
            
    print(f"✅ 배경 {len(ready_backgrounds)}개 준비 완료!")
    return ready_backgrounds

# ==========================================
# 4. 유틸리티
# ==========================================
def create_1024_tiled_bg(bg_paths):
    # 이미 정렬된 16장이 들어옴
    rows = []
    idx = 0
    for _ in range(4):
        cols = []
        for _ in range(4):
            img = cv2.imread(bg_paths[idx])
            img = cv2.resize(img, (256, 256))
            cols.append(img)
            idx += 1
        rows.append(np.hstack(cols))
    return np.vstack(rows)

def resize_object_smart(obj_img):
    h, w = obj_img.shape[:2]
    canvas_diag = np.sqrt(TARGET_SIZE**2 + TARGET_SIZE**2)
    obj_diag = np.sqrt(h**2 + w**2)
    target_diag = random.uniform(0.30, 0.60) * canvas_diag
    scale = target_diag / obj_diag
    return cv2.resize(obj_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# ==========================================
# 5. 메인 실행
# ==========================================
def main():
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', subset), exist_ok=True)
    
    # 1. 배경 600개 생성 (색상 정렬 방식)
    background_pool = create_sorted_background_pool(BG_ROOT, TOTAL_IMGS_NEEDED)
    
    if not background_pool:
        print("❌ 배경 생성 실패!")
        return

    # 2. 객체 합성 시작
    obj_files = glob.glob(os.path.join(OBJ_ROOT, "*.png"))
    random.shuffle(obj_files)
    print(f"🧸 객체 {len(obj_files)}개 처리 시작.")

    count = 0
    
    for obj_path in tqdm(obj_files, desc="Synthesizing"):
        obj_img_orig = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
        if obj_img_orig is None: continue
        
        for _ in range(REPEAT_PER_OBJECT):
            # 미리 만든 배경 중 랜덤 선택
            large_bg = random.choice(background_pool)
            
            obj_img = resize_object_smart(obj_img_orig)
            h_obj, w_obj = obj_img.shape[:2]
            if h_obj >= TARGET_SIZE or w_obj >= TARGET_SIZE: continue
            
            y_pos = random.randint(0, TARGET_SIZE - h_obj)
            x_pos = random.randint(0, TARGET_SIZE - w_obj)
            
            # 합성 & 마스크
            composite = large_bg.copy()
            mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
            
            if obj_img.shape[2] == 4:
                alpha_s = obj_img[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(3):
                    composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c] = \
                        (alpha_s * obj_img[:, :, c] + alpha_l * composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c])
                obj_mask_area = (obj_img[:, :, 3] > 0).astype(np.uint8) * 255
            else:
                composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = obj_img
                obj_mask_area = 255
                
            mask[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = obj_mask_area
            
            # 모델 실행
            harmonized_result = net(composite, mask)
            
            # Float -> Uint8 변환 (안전장치)
            if harmonized_result.dtype != np.uint8:
                harmonized_result = np.clip(harmonized_result * 255, 0, 255).astype(np.uint8)

            # 저장
            split = 'val' if random.random() < VAL_SPLIT_RATIO else 'train'
            filename = f"{split}_{count:06d}"
            
            cv2.imwrite(os.path.join(OUTPUT_ROOT, 'images', split, f"{filename}.jpg"), harmonized_result)
            
            cx = (x_pos + w_obj / 2) / TARGET_SIZE
            cy = (y_pos + h_obj / 2) / TARGET_SIZE
            nw = w_obj / TARGET_SIZE
            nh = h_obj / TARGET_SIZE
            
            with open(os.path.join(OUTPUT_ROOT, 'labels', split, f"{filename}.txt"), "w") as f:
                f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            
            count += 1

    print(f"\n✅ 진짜 완료! {count}장.")

if __name__ == "__main__":
    main()