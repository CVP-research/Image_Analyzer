import os
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm
from libcom import image_harmonization

# ==========================================
# 1. 설정 (디버깅 모드)
# ==========================================
BG_ROOT = "./dataset"
OBJ_ROOT = "./output/dataset/masked_frames"
# 디버깅 결과물을 저장할 폴더
DEBUG_DIR = "./debug_output"

TARGET_SIZE = 1024
# 디버깅을 위해 배경 1개, 객체 2개만 테스트합니다.
NUM_BG_GROUPS = 1
IMGS_PER_GROUP = 16
REPEAT_PER_BG = 2
CLASS_ID = 0

# ==========================================
# 2. 모델 로드
# ==========================================
print("🚀 (디버깅) 모델 로드 중...")
try:
    net = image_harmonization.ImageHarmonizationModel(device=0)
except:
    print("⚠️ CDTNet 실패, 기본 모델 사용")
    net = image_harmonization.ImageHarmonizationModel(device=0)

# ==========================================
# 3. 유틸리티 함수 (그대로 유지)
# ==========================================
def prepare_background_debug(root_dir):
    print(f"📂 (디버깅) 이미지 경로 수집 중...")
    paths = []
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                paths.append(os.path.join(subdir, f))
                if len(paths) >= 50: break
        if len(paths) >= 50: break
    
    if len(paths) < 16: raise Exception("❌ 배경 이미지가 너무 적습니다!")

    print(f"🎨 (디버깅) 색상 정렬 중...")
    color_info = []
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
    
    # 딱 1개의 배경만 만듦
    chunk = sorted_paths[:16]
    rows = []
    idx = 0
    for _ in range(4):
        cols = []
        for _ in range(4):
            img = cv2.imread(chunk[idx])
            if img is None: img = np.zeros((256, 256, 3), np.uint8)
            img = cv2.resize(img, (256, 256))
            cols.append(img)
            idx += 1
        rows.append(np.hstack(cols))
    bg = np.vstack(rows)
    return [bg]

def resize_object_smart(obj_img):
    h, w = obj_img.shape[:2]
    canvas_diag = np.sqrt(TARGET_SIZE**2 + TARGET_SIZE**2)
    obj_diag = np.sqrt(h**2 + w**2)
    target_diag = random.uniform(0.30, 0.60) * canvas_diag
    scale = target_diag / obj_diag
    return cv2.resize(obj_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# ==========================================
# 4. 메인 실행 (★단계별 저장 추가★)
# ==========================================
def main():
    if os.path.exists(DEBUG_DIR):
        import shutil
        shutil.rmtree(DEBUG_DIR)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"🐞 디버깅 시작! 결과물은 {DEBUG_DIR} 폴더에 저장됩니다.")
    
    # 1. 배경 1개 준비
    background_pool = prepare_background_debug(BG_ROOT)
    large_bg = background_pool[0]

    # [디버그 저장 1] 배경 이미지 확인
    cv2.imwrite(os.path.join(DEBUG_DIR, "step1_background.jpg"), large_bg)
    print("✅ step1_background.jpg 저장 완료")
    
    # 2. 객체 준비
    obj_files = glob.glob(os.path.join(OBJ_ROOT, "*.png"))
    if not obj_files: return
    
    # 테스트용 객체 2개만 선택
    selected_objs = random.sample(obj_files, min(len(obj_files), REPEAT_PER_BG))
            
    for idx, obj_path in enumerate(selected_objs):
        print(f"\n--- 객체 {idx+1} 처리 중 ---")
        obj_img_orig = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
        if obj_img_orig is None: continue
        
        # 리사이징 & 위치 선정
        obj_img = resize_object_smart(obj_img_orig)
        h_obj, w_obj = obj_img.shape[:2]
        y_pos = random.randint(0, TARGET_SIZE - h_obj)
        x_pos = random.randint(0, TARGET_SIZE - w_obj)
        
        # 합성 & 마스크
        composite = large_bg.copy()
        mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
        
        # Alpha Blending (여기가 의심 구간 1)
        if obj_img.shape[2] == 4:
            alpha_s = obj_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                # float 연산 후 uint8로 안전하게 변환하는지 확인 필요
                blended = (alpha_s * obj_img[:, :, c] + alpha_l * composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c])
                composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c] = np.clip(blended, 0, 255).astype(np.uint8)
                
            obj_mask_area = (obj_img[:, :, 3] > 0).astype(np.uint8) * 255
        else:
            composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = obj_img
            obj_mask_area = 255
            
        mask[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = obj_mask_area
        
        # [디버그 저장 2 & 3] 모델 들어가기 전 합성본과 마스크 확인
        cv2.imwrite(os.path.join(DEBUG_DIR, f"step2_composite_raw_{idx}.jpg"), composite)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"step3_mask_{idx}.jpg"), mask)
        print(f"✅ step2 & step3 (객체 {idx}) 저장 완료")

        # ★ Harmonizer 실행 ★
        harmonized_result = net(composite, mask)
        
        # [수정] 결과물 타입 안전 변환 (여기가 의심 구간 2)
        if harmonized_result.dtype != np.uint8:
            # 혹시 float 0~1 범위면 255 곱하기
            if harmonized_result.max() <= 1.00001:
                 harmonized_result = harmonized_result * 255.0
            harmonized_result = np.clip(harmonized_result, 0, 255).astype(np.uint8)

        # [디버그 저장 4] 최종 결과물 확인
        cv2.imwrite(os.path.join(DEBUG_DIR, f"step4_harmonized_final_{idx}.jpg"), harmonized_result)
        print(f"✅ step4 (객체 {idx}) 저장 완료")
        
        # Segmentation 라벨 확인용 시각화
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # 원본 이미지에 폴리곤 그리기 (빨간색)
            check_img = harmonized_result.copy()
            cv2.drawContours(check_img, [approx], -1, (0, 0, 255), 3)
            
            # [디버그 저장 5] 라벨링 시각화 확인
            cv2.imwrite(os.path.join(DEBUG_DIR, f"step5_contour_check_{idx}.jpg"), check_img)
            print(f"✅ step5 (객체 {idx}) 저장 완료")

    print(f"\n🐞 디버깅 완료! {DEBUG_DIR} 폴더를 확인해주세요.")

if __name__ == "__main__":
    main()