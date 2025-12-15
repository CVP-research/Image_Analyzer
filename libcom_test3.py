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
OUTPUT_ROOT = "./output/dataset/libcom7" # 경로 구분

TARGET_SIZE = 1024        
NUM_BG_GROUPS = 600       
IMGS_PER_GROUP = 16       
REPEAT_PER_BG = 5         
VAL_SPLIT_RATIO = 0.2     
CLASS_ID = 0              

# ★ 10% 확률로 빈 배경 생성 (과탐지 방지 핵심)
EMPTY_RATIO = 0

# ==========================================
# 2. 모델 로드
# ==========================================
print("🚀 모델 로드 중...")
net = image_harmonization.ImageHarmonizationModel(device=0)


# ==========================================
# 3. 배경 준비 함수
# ==========================================
def prepare_600_backgrounds(root_dir, target_count):
    # ... (이전과 동일하여 생략, 기존 코드 그대로 사용) ...
    # (전체 코드가 필요하면 말씀주세요, 위쪽 코드와 동일합니다)
    print(f"📂 이미지 경로 수집 중...")
    paths = []
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                paths.append(os.path.join(subdir, f))
                if len(paths) >= target_count * 20: break
        if len(paths) >= target_count * 20: break
    
    if len(paths) < 16: raise Exception("❌ 배경 이미지가 너무 적습니다!")

    print(f"🎨 색상 정렬 중...")
    color_info = []
    paths = paths[:20000]
    for p in paths:
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
# 4. 메인 실행
# ==========================================
def main():
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', subset), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', subset), exist_ok=True)
    
    full_bg_pool = prepare_600_backgrounds(BG_ROOT, NUM_BG_GROUPS)
    
    random.shuffle(full_bg_pool)
    split_idx = int(len(full_bg_pool) * (1 - VAL_SPLIT_RATIO))
    train_bgs = full_bg_pool[:split_idx]
    val_bgs = full_bg_pool[split_idx:]
    
    obj_files = glob.glob(os.path.join(OBJ_ROOT, "*.png"))
    if not obj_files: return
    
    count = 0
    
    def generate_split_data(bg_list, split_name):
        nonlocal count
        for large_bg in tqdm(bg_list, desc=f"Generating {split_name}"):
            
            for _ in range(REPEAT_PER_BG):
                filename = f"{split_name}_{count:06d}"
                img_save_path = os.path.join(OUTPUT_ROOT, 'images', split_name, f"{filename}.jpg")
                # 변수명 정의: lbl_save_path
                lbl_save_path = os.path.join(OUTPUT_ROOT, 'labels', split_name, f"{filename}.txt")

                # [1] 빈 배경 생성 (과탐지 방지용)
                if random.random() < EMPTY_RATIO:
                    cv2.imwrite(img_save_path, large_bg)
                    open(lbl_save_path, 'w').close()
                    count += 1
                    continue

                obj_path = random.choice(obj_files)
                obj_img_orig = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
                if obj_img_orig is None: continue
                
                obj_img = resize_object_smart(obj_img_orig)
                h_obj, w_obj = obj_img.shape[:2]
                if h_obj >= TARGET_SIZE or w_obj >= TARGET_SIZE: continue
                
                y_pos = random.randint(0, TARGET_SIZE - h_obj)
                x_pos = random.randint(0, TARGET_SIZE - w_obj)
                
                # [2] 마스크 생성 (Threshold만 유지)
                if obj_img.shape[2] == 4:
                    alpha_channel = obj_img[:, :, 3]
                    _, clean_mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
                    
                    obj_img[:, :, 3] = cv2.bitwise_and(obj_img[:, :, 3], clean_mask)
                    obj_mask_area = clean_mask
                else:
                    clean_mask = np.ones((h_obj, w_obj), dtype=np.uint8) * 255
                    obj_mask_area = 255

                # 합성
                composite = large_bg.copy()
                mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
                mask[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = clean_mask
                
                if obj_img.shape[2] == 4:
                    alpha_s = obj_img[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(3):
                        composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c] = \
                            (alpha_s * obj_img[:, :, c] + alpha_l * composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c])
                else:
                    composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = obj_img
                
                # Harmonizer
                composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                harmonized_rgb = net(composite_rgb, mask)
                if harmonized_rgb.max() <= 1.0: harmonized_rgb *= 255.0
                harmonized_rgb = np.clip(harmonized_rgb, 0, 255).astype(np.uint8)
                final_bgr = cv2.cvtColor(harmonized_rgb, cv2.COLOR_RGB2BGR)

                cv2.imwrite(img_save_path, final_bgr)
                
                # 라벨 생성 (0.0001 정밀도)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    epsilon = 0.0001 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    if len(approx) < 4: approx = cnt 

                    points_str = []
                    for point in approx:
                        x, y = point[0]
                        nx = min(max(x / TARGET_SIZE, 0), 1)
                        ny = min(max(y / TARGET_SIZE, 0), 1)
                        points_str.append(f"{nx:.6f} {ny:.6f}")
                    
                    # ★★★ [수정됨] lbl_path -> lbl_save_path ★★★
                    with open(lbl_save_path, "w") as f:
                        f.write(f"{CLASS_ID} " + " ".join(points_str) + "\n")
                
                count += 1

    generate_split_data(train_bgs, 'train')
    generate_split_data(val_bgs, 'val')

    print(f"\n✅ 완료! 총 {count}장.")

if __name__ == "__main__":
    main()