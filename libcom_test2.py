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
OUTPUT_ROOT = "./output/dataset/yolo_harmonized_data5"

TARGET_SIZE = 1024        
NUM_BG_GROUPS = 600       
IMGS_PER_GROUP = 16       
REPEAT_PER_BG = 5         
VAL_SPLIT_RATIO = 0.2     # 배경 개수 기준으로 20%를 뚝 뗍니다.
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
# 3. 배경 준비 함수 (동일)
# ==========================================
def prepare_600_backgrounds(root_dir, target_count):
    print(f"📂 이미지 경로 수집 중...")
    paths = []
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                paths.append(os.path.join(subdir, f))
                if len(paths) >= target_count * 20: break
        if len(paths) >= target_count * 20: break
    
    if len(paths) < 16: raise Exception("❌ 배경 이미지가 너무 적습니다!")

    print(f"🎨 색상 정렬 중 ({len(paths)}장)...")
    color_info = []
    paths = paths[:20000]
    
    for p in tqdm(paths, desc="Sorting Colors"):
        try:
            img = cv2.imread(p)
            if img is None: continue
            pixel = cv2.resize(img, (1, 1))
            mean_val = np.sum(pixel)
            color_info.append((mean_val, p))
        except: continue

    color_info.sort(key=lambda x: x[0])
    sorted_paths = [x[1] for x in color_info]
    
    print("🧱 타일 배경 생성 중...")
    ready_backgrounds = []
    for i in tqdm(range(0, len(sorted_paths), 16), desc="Tiling"):
        chunk = sorted_paths[i : i+16]
        if len(chunk) < 16: break
        bg = create_1024_tiled_bg(chunk)
        ready_backgrounds.append(bg)
        if len(ready_backgrounds) >= target_count: break
            
    while len(ready_backgrounds) < target_count:
        ready_backgrounds.append(random.choice(ready_backgrounds))

    return ready_backgrounds

def create_1024_tiled_bg(bg_paths):
    rows = []
    idx = 0
    for _ in range(4):
        cols = []
        for _ in range(4):
            img = cv2.imread(bg_paths[idx])
            if img is None: img = np.zeros((256, 256, 3), np.uint8)
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
# 4. 메인 실행 (★완전히 분리된 생성 로직★)
# ==========================================
def main():
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', subset), exist_ok=True)
    
    # 1. 배경 600개 생성
    full_bg_pool = prepare_600_backgrounds(BG_ROOT, NUM_BG_GROUPS)
    
    # ★★★ [핵심] 배경 리스트를 섞은 뒤, Train용과 Val용으로 칼같이 나눔 ★★★
    random.shuffle(full_bg_pool)
    split_idx = int(len(full_bg_pool) * (1 - VAL_SPLIT_RATIO)) # 80% 지점
    
    train_bgs = full_bg_pool[:split_idx] # 480개
    val_bgs = full_bg_pool[split_idx:]   # 120개 (Train과 겹치는 배경 0개)
    
    print(f"✂️ 데이터 분할: Train 배경 {len(train_bgs)}개 vs Val 배경 {len(val_bgs)}개")
    
    obj_files = glob.glob(os.path.join(OBJ_ROOT, "*.png"))
    if not obj_files: return
    print(f"🧸 객체 파일 {len(obj_files)}개 발견.")
    
    count = 0
    
    # 함수 하나로 묶어서 처리 (Train/Val 반복 줄이기 위함)
    def generate_split_data(bg_list, split_name):
        nonlocal count
        for large_bg in tqdm(bg_list, desc=f"Generating {split_name}"):
            selected_objs = random.choices(obj_files, k=REPEAT_PER_BG)
                
            for obj_path in selected_objs:
                obj_img_orig = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
                if obj_img_orig is None: continue
                
                # 리사이징 & 위치
                obj_img = resize_object_smart(obj_img_orig)
                h_obj, w_obj = obj_img.shape[:2]
                if h_obj >= TARGET_SIZE or w_obj >= TARGET_SIZE: continue
                
                y_pos = random.randint(0, TARGET_SIZE - h_obj)
                x_pos = random.randint(0, TARGET_SIZE - w_obj)
                
                # 합성할 캔버스 준비
                composite = large_bg.copy()
                mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

                if obj_img.shape[2] == 4:
                    # 1. 알파 채널 가져오기
                    alpha_channel = obj_img[:, :, 3]
                    
                    # 2. 엄격한 기준 (Threshold): 반투명 노이즈 제거 (200 이상만 인정)
                    _, binary_mask = cv2.threshold(alpha_channel, 200, 255, cv2.THRESH_BINARY)
                    
                    # 3. ★핵심★ 마스크 깎기 (Erosion)
                    # iterations=2 -> 2픽셀만큼 안쪽으로 깎아냅니다. (이게 손/배경 분리 핵심)
                    kernel = np.ones((3, 3), np.uint8)
                    clean_mask = cv2.erode(binary_mask, kernel, iterations=2)
                    
                    # 4. 이미지의 알파 채널도 깎인 마스크에 맞춰서 잘라냄
                    # (이걸 해야 눈에 보이는 이미지랑 라벨이 일치함)
                    obj_img[:, :, 3] = cv2.bitwise_and(obj_img[:, :, 3], clean_mask)
                    
                    # 5. 최종 마스크 확정
                    obj_mask_area = clean_mask
                    
                    # 6. 알파 블렌딩 (수정된 깨끗한 알파값 사용)
                    alpha_s = obj_img[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(3):
                        composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c] = \
                            (alpha_s * obj_img[:, :, c] + alpha_l * composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c])
                            
                else:
                    # 알파 채널 없는 경우 (거의 없겠지만)
                    composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = obj_img
                    obj_mask_area = 255
                
                # 전체 마스크에 객체 마스크 등록
                mask[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = obj_mask_area
                
                # BGR -> RGB -> Model -> BGR
                composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                harmonized_rgb = net(composite_rgb, mask)
                if harmonized_rgb.max() <= 1.0: harmonized_rgb *= 255.0
                harmonized_rgb = np.clip(harmonized_rgb, 0, 255).astype(np.uint8)
                final_bgr = cv2.cvtColor(harmonized_rgb, cv2.COLOR_RGB2BGR)

                # 저장
                filename = f"{split_name}_{count:06d}"
                cv2.imwrite(os.path.join(OUTPUT_ROOT, 'images', split_name, f"{filename}.jpg"), final_bgr)
                
                # 라벨
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    
                    # ★★★ [수정 핵심] 0.002 -> 0.0001로 변경 ★★★
                    # 수치가 작을수록 더 세밀하게(원래 모양에 가깝게) 땁니다.
                    epsilon = 0.0001 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    # 만약 너무 단순화돼서 점이 4개 미만이면(삼각형 등), 그냥 원본 사용
                    if len(approx) < 4:
                        approx = cnt
                    
                    points_str = []
                    for point in approx:
                        x, y = point[0]
                        # 좌표 정규화 (0~1)
                        nx = min(max(x / TARGET_SIZE, 0), 1)
                        ny = min(max(y / TARGET_SIZE, 0), 1)
                        points_str.append(f"{nx:.6f} {ny:.6f}")
                    
                    lbl_path = os.path.join(OUTPUT_ROOT, 'labels', split_name, f"{filename}.txt")
                    with open(lbl_path, "w") as f:
                        f.write(f"{CLASS_ID} " + " ".join(points_str) + "\n")
                
                count += 1

    # 1. 학습 데이터 생성 (Train용 배경만 사용)
    generate_split_data(train_bgs, 'train')
    
    # 2. 검증 데이터 생성 (Val용 배경만 사용)
    generate_split_data(val_bgs, 'val')

    print(f"\n✅ 누수 없는 데이터셋 생성 완료! 총 {count}장.")

if __name__ == "__main__":
    main()