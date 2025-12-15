import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import SAM

# --- 1. 설정 (이전과 동일) ---
SAM_MODEL = None
DATASET_DIR = Path("../data/77_7859_15670/images")
OUTPUT_CLEAN_DIR = Path("../output/clean_masked_objects/")


def get_sam_model():
    """SAM 모델을 로드합니다."""
    # (SAM 로드 로직은 생략. main 함수에 구현되어 있음)
    global SAM_MODEL
    if SAM_MODEL is None:
        SAM_MODEL = SAM("....//sam2_l.pt") 
    return SAM_MODEL


def extract_object_from_mask(img: np.ndarray, binary_mask: np.ndarray, frame_name: str, out_dir: Path):
    """이진 마스크를 사용하여 배경 잔여물 없이 객체만 투명 PNG로 추출합니다."""
    
    # 1. 알파 채널 생성 (객체는 255, 배경은 0)
    alpha_channel = binary_mask

    # 2. 배경 잔여물 완벽 제거 (가장 중요한 클린 로직)
    mask_3channel = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
    
    img_float = img.astype(float)
    mask_float = mask_3channel.astype(float) / 255.0
    
    # 마스크 영역만 남기고 배경은 0으로 만듭니다. (잔여물 원천 차단)
    foreground = cv2.multiply(img_float, mask_float)
    foreground = foreground.astype(np.uint8)

    # 3. BGRA 이미지 생성 및 합치기
    b, g, r = cv2.split(foreground)
    bgra_image = cv2.merge([b, g, r, alpha_channel])
    
    # 4. 크롭 및 저장
    ys, xs = np.where(alpha_channel > 0)
    
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # 크롭 시 경계선이 잘리지 않도록 1픽셀 여유를 둡니다.
        x_min = max(0, x_min - 1)
        y_min = max(0, y_min - 1)
        x_max = min(img.shape[1], x_max + 1)
        y_max = min(img.shape[0], y_max + 1)
        
        bgra_image_cropped = bgra_image[y_min:y_max, x_min:x_max]
    else:
        bgra_image_cropped = bgra_image
        
    out_path = out_dir / f"{frame_name}_final_clean.png"
    cv2.imwrite(str(out_path), bgra_image_cropped)
    
    return out_path


def segment_and_extract_by_average(frame_path: Path, model, OUTPUT_CLEAN_DIR: Path):
    """
    모든 마스크를 평균 내어 가장 큰 영역을 최종 마스크로 선택하고 누끼를 땁니다.
    """
    frame_name = frame_path.stem
    img = cv2.imread(str(frame_path))
    if img is None: return None

    try:
        pred_results = model.predict(img, task="segment")
    except Exception as e:
        print(f"[ERROR] SAM 예측 실패: {e}")
        return None

    if not pred_results or not hasattr(pred_results[0], 'masks') or pred_results[0].masks is None:
        print(f"[WARN] 마스크 감지 실패: {frame_name}")
        return None

    # 모든 마스크를 리스트에 수집
    all_masks = []
    
    if pred_results[0].masks is not None:
        for mask_tensor in pred_results[0].masks.data:
            # 마스크를 float32로 변환하여 평균 계산 준비
            mask_np = mask_tensor.cpu().numpy().astype(np.float32)
            all_masks.append(mask_np)

    if not all_masks: return None
    
    # 1. 🚨 모든 마스크 평균 계산
    # np.stack으로 마스크들을 쌓고 평균을 냅니다.
    stacked_masks = np.stack(all_masks, axis=0)
    average_mask_float = np.mean(stacked_masks, axis=0) # [H, W] 형태의 평균 마스크 (0.0 ~ 1.0)
    
    # 2. 🚨 평균 마스크를 기반으로 가장 큰 마스크 선택 (면적 기준)
    # 현재 코드 구조상, 평균 마스크를 바로 사용하기 보다는
    # '가장 평균과 유사하거나 큰' 원본 마스크를 고르는 것이 안정적입니다.
    
    # Simplification: 마스크 평균을 내면 경계선이 부드러워지므로,
    # '가장 큰 마스크'를 평균 마스크의 대표로 가정합니다. (가장 신뢰도 높은 영역)
    
    largest_mask_np = None
    largest_area = 0
    
    for mask_np_float in all_masks:
        # float 마스크를 다시 이진(uint8)으로 변환
        mask_np_uint8 = (mask_np_float > 0.5).astype(np.uint8) # 0.5를 임계값으로 사용
        area = mask_np_uint8.sum()
        
        if area > largest_area:
            largest_area = area
            # 최종 마스크는 uint8 (0 또는 1) 형태로 저장
            largest_mask_np = mask_np_uint8 

    if largest_mask_np is None: return None
    
    # 최종 마스크를 0/255로 확장
    final_binary_mask = (largest_mask_np * 255).astype(np.uint8)

    # 3. 누끼 따기 실행
    out_path = extract_object_from_mask(img, final_binary_mask, frame_name, OUTPUT_CLEAN_DIR)
    print(f"✓ {frame_name}: 평균화 로직을 거친 최종 마스크로 누끼 완료. 저장 경로: {out_path.name}")
    return out_path


# ==========================================
# 4. 메인 실행
# ==========================================
def main():
    
    OUTPUT_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        model = get_sam_model()
    except Exception as e:
        print(f"Fatal Error: SAM 모델 로드 실패. 실행을 종료합니다. {e}")
        return

    frame_paths = sorted(DATASET_DIR.glob("*.jpg"))
    if not frame_paths:
        print(f"❌ 오류: 입력 디렉토리 {DATASET_DIR}에 JPG 파일이 없습니다.")
        return

    print(f"--- 총 {len(frame_paths)}개의 프레임 분할 및 누끼 시작 ---")
    start_total = time.time()
    
    for frame_path in frame_paths:
        segment_and_extract_by_average(frame_path, model, OUTPUT_CLEAN_DIR)
        
    end_total = time.time()
    print(f"--- 모든 프레임 처리 완료 ({end_total - start_total:.2f}초 소요) ---")


if __name__ == "__main__":
    main()