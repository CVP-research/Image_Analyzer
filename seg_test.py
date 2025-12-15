import os
import cv2
import numpy as np
import random
import glob
from libcom import image_harmonization

# ==========================================
# 1. 설정
# ==========================================
BG_ROOT = "./dataset"
OBJ_ROOT = "./output/dataset/masked_frames"
TARGET_SIZE = 1024

# ==========================================
# 2. 모델 로드
# ==========================================
print("🚀 모델 로드 중...")

net = image_harmonization.ImageHarmonizationModel(device=0)


# ==========================================
# 3. 유틸리티 함수
# ==========================================
def create_random_1024_bg(root_dir):
    # 배경 폴더에서 이미지 긁어오기
    files = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True) + \
            glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    
    if len(files) < 16:
        # 파일이 적으면 중복 허용해서 16개 뽑기
        selected = random.choices(files, k=16)
    else:
        # 파일이 많으면 중복 없이 16개 뽑기
        selected = random.sample(files, 16)
    
    rows = []
    idx = 0
    for _ in range(4):
        cols = []
        for _ in range(4):
            img = cv2.imread(selected[idx])
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
    print("🎨 배경 생성 중...")
    large_bg = create_random_1024_bg(BG_ROOT)
    
    print("🧸 객체 합성 중...")
    obj_files = glob.glob(os.path.join(OBJ_ROOT, "*.png"))
    obj_path = random.choice(obj_files) # 아무거나 하나 선택
    
    obj_img_orig = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
    obj_img = resize_object_smart(obj_img_orig)
    
    h_obj, w_obj = obj_img.shape[:2]
    y_pos = random.randint(0, TARGET_SIZE - h_obj)
    x_pos = random.randint(0, TARGET_SIZE - w_obj)
    
    # 합성
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
    
    # Harmonizer 실행 (BGR -> RGB -> Model -> BGR)
    print("✨ Harmonizer 적용 중...")
    composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    harmonized_rgb = net(composite_rgb, mask)
    
    if harmonized_rgb.max() <= 1.0: harmonized_rgb *= 255.0
    harmonized_rgb = np.clip(harmonized_rgb, 0, 255).astype(np.uint8)
    final_bgr = cv2.cvtColor(harmonized_rgb, cv2.COLOR_RGB2BGR)

    # 결과 저장
    cv2.imwrite("test_result_final.jpg", final_bgr)
    print("✅ 최종 이미지 저장 완료: test_result_final.jpg")
    
    # ---------------------------------------------------------
    # ★ 라벨 시각화 (눈으로 확인하는 부분) ★
    # ---------------------------------------------------------
    print("🔍 라벨 시각화 중...")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        
        # ★★★ 여기가 핵심: epsilon = 0.0001 (초정밀) ★★★
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 이미지에 그리기 (빨간색 선, 두께 2)
        vis_img = final_bgr.copy()
        cv2.drawContours(vis_img, [approx], -1, (0, 0, 255), 2)
        
        cv2.imwrite("test_result_label_vis.jpg", vis_img)
        print("✅ 라벨 시각화 저장 완료: test_result_label_vis.jpg")
        print("   -> 빨간 선이 인형 외곽선에 딱 붙어있는지 확인하세요!")

if __name__ == "__main__":
    main()