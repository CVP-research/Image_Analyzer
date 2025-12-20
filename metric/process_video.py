import os
import cv2
import numpy as np
from ultralytics import SAM
from pathlib import Path
import time
from tqdm import tqdm

# ==========================================
# 1. 설정
# ==========================================

# 🚨 사용자 설정 필요 🚨
VIDEO_PATH = Path("/home/rocknroll1397/Image_Analyzer/new/data/77_7859_15670/test3.mp4")  # 처리할 비디오 파일 경로
OUTPUT_DIR = Path("./output/video_dataset")
SIMILARITY_THRESHOLD = 1000  # 프레임 유사도 임계값. 낮을수록 더 많은 프레임을 처리.

# 모델 경로
SAM_MODEL_PATH = "../sam2_l.pt"

# ==========================================
# 2. 핵심 유틸리티 함수
# ==========================================

def are_frames_different(frame1: np.ndarray, frame2: np.ndarray, threshold: float) -> bool:
    """두 프레임의 유사도를 계산하여 다른지 여부를 반환."""
    if frame1 is None or frame2 is None:
        return True
    
    # 프레임을 그레이스케일로 변환하여 계산 단순화
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 평균 절대차(Mean Absolute Difference) 계산
    abs_diff = cv2.absdiff(gray1, gray2)
    mean_abs_diff = np.mean(abs_diff)
    
    return mean_abs_diff > threshold

def get_sam_model(model_path: str):
    """SAM 모델을 로드하고 반환."""
    try:
        print("Loading SAM model...")
        model = SAM(model_path)
        print("SAM model loaded.")
        return model
    except Exception as e:
        print(f"❌ FATAL ERROR: SAM 모델 로드 실패. 경로 확인: {e}")
        return None

def get_sam_masks(model: SAM, image_np: np.ndarray) -> list:
    """SAM 모델을 실행하고 마스크 리스트 반환"""
    if model is None: return []
    pred_results = model.predict(image_np, task="segment", verbose=False)
    if not pred_results or pred_results[0].masks is None: return []
    return [m.cpu().numpy().astype(np.uint8) for m in pred_results[0].masks.data]

def save_masks(frame_index: int, masks: list, output_dir: Path):
    """추출된 마스크들을 개별 파일로 저장"""
    if not masks:
        return
        
    for i, mask_np in enumerate(masks):
        # 파일명 형식: {비디오이름}_{프레임번호}_{마스크인덱스}.png
        output_filename = f"{VIDEO_PATH.stem}_{frame_index:05d}_{i:02d}.png"
        output_path = output_dir / "masks" / output_filename
        
        try:
            # 마스크를 0 또는 255 값을 가지는 그레이스케일 이미지로 저장
            mask_to_save = (mask_np * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), mask_to_save)
        except Exception as e:
            print(f"❌ 마스크 저장 실패 {output_path}: {e}")

def save_original_frame(frame_index: int, frame: np.ndarray, output_dir: Path):
    """원본 프레임을 저장"""
    output_filename = f"{VIDEO_PATH.stem}_{frame_index:05d}.jpg"
    output_path = output_dir / "images" / output_filename
    try:
        cv2.imwrite(str(output_path), frame)
    except Exception as e:
        print(f"❌ 원본 프레임 저장 실패 {output_path}: {e}")

# ==========================================
# 3. 비디오 처리 메인 함수
# ==========================================

def process_video():
    """비디오를 프레임 단위로 읽고, 유사하지 않은 프레임만 분할하여 데이터셋 생성"""
    print("\n--- 🚀 비디오 프레임 분할 및 데이터셋 생성 시작 ---")
    
    # 출력 폴더 생성
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

    # SAM 모델 로드
    sam_model = get_sam_model(SAM_MODEL_PATH)
    if sam_model is None:
        return

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"❌ 오류: 비디오를 열 수 없습니다. 경로 확인: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ 비디오 정보: {total_frames} frames, {cap.get(cv2.CAP_PROP_FPS):.2f} FPS")
    print(f"✅ 처리 설정: 유사도 임계값={SIMILARITY_THRESHOLD}")

    start_time = time.time()
    
    prev_frame = None
    processed_count = 0
    
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 이전 프레임과 현재 프레임의 유사도 체크
            if are_frames_different(prev_frame, frame, SIMILARITY_THRESHOLD):
                # 원본 프레임 저장
                save_original_frame(frame_index, frame, OUTPUT_DIR)
                
                # SAM으로 마스크 생성
                masks_list = get_sam_masks(sam_model, frame)
                
                # 생성된 마스크들 저장
                save_masks(frame_index, masks_list, OUTPUT_DIR)
                
                processed_count += 1
                prev_frame = frame.copy() # 다음 비교를 위해 현재 프레임 저장

            frame_index += 1
            pbar.update(1)

    cap.release()
    end_time = time.time()
    
    print(f"\n--- ✅ 비디오 처리 완료! ({end_time - start_time:.2f}초) ---")
    print(f"📁 총 {total_frames}개 프레임 중 {processed_count}개의 고유 프레임을 처리했습니다.")
    print(f"📁 데이터셋 저장 위치: {OUTPUT_DIR.resolve()}")

# ==========================================
# 4. 스크립트 메인 실행부
# ==========================================
if __name__ == "__main__":
    process_video()
