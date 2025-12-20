
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import os

# ==========================================
# 🚨 사용자 설정 영역 🚨
# ==========================================

# 1. 처리할 동영상 파일 경로
INPUT_VIDEO_PATH = Path("/home/rocknroll1397/Image_Analyzer/new/data/77_7859_15670/test3.mp4")

# 2. 'veo3' 및 'no veo3' 모델의 .pt 파일 경로
VEO3_MODEL_PATH = Path("/home/rocknroll1397/Image_Analyzer/runs/segment/train_veo3/weights/best.pt")
NO_VEO3_MODEL_PATH = Path("/home/rocknroll1397/Image_Analyzer/runs/segment/train_no_veo3/weights/best.pt")

# 3. 결과물을 저장할 폴더 경로
OUTPUT_DIR = Path("output/video_segmentation_results")

# 4. 프레임 유사도 임계값 (SSIM)
#    - 값이 1에 가까울수록 두 프레임이 유사하다는 의미입니다.
#    - 0.95는 약간의 변화가 있는 프레임도 처리하도록 설정한 값입니다.
SIMILARITY_THRESHOLD = 0.95

# ==========================================
# 핵심 유틸리티 함수
# ==========================================

def get_frame_similarity(frame1, frame2):
    """두 프레임 간의 구조적 유사성(SSIM)을 계산합니다."""
    if frame1 is None or frame2 is None:
        return 1.0  # 첫 프레임의 경우 항상 다르다고 판단
    
    # SSIM 계산을 위해 그레이스케일로 변환
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    return ssim(gray1, gray2)

def visualize_segmentation_comparison(image, veo3_masks, no_veo3_masks, output_path):
    """'veo3'와 'no veo3' 세그멘테이션 결과를 비교하여 시각화합니다."""
    vis_image = image.copy()

    # 'no veo3' 마스크 외곽선 (초록색)
    for mask in no_veo3_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

    # 'veo3' 마스크 외곽선 (빨간색)
    for mask in veo3_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 0, 255), 2)

    cv2.imwrite(str(output_path), vis_image)

# ==========================================
# 메인 비디오 처리 로직
# ==========================================

def main():
    """메인 비디오 처리 및 세그멘테이션 프로세스를 실행합니다."""
    print("🚀 비디오 처리 및 세그멘테이션을 시작합니다...")

    # 1. 설정 확인 및 폴더 생성
    if not INPUT_VIDEO_PATH.exists():
        print(f"❌ 오류: 비디오 파일을 찾을 수 없습니다: {INPUT_VIDEO_PATH}")
        return
    if not VEO3_MODEL_PATH.exists() or not NO_VEO3_MODEL_PATH.exists():
        print(f"❌ 오류: 세그멘테이션 모델 파일을 확인해주세요.")
        return
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 2. 모델 로드
    print("🔍 모델 로딩 중...")
    veo3_model = YOLO(VEO3_MODEL_PATH)
    no_veo3_model = YOLO(NO_VEO3_MODEL_PATH)
    
    # 3. 비디오 캡처 및 처리
    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    if not cap.isOpened():
        print(f"❌ 오류: 비디오를 열 수 없습니다: {INPUT_VIDEO_PATH}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame = None
    processed_frame_count = 0
    
    print(f"📹 총 {frame_count}개의 프레임을 처리합니다...")
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 유사도 체크
        if prev_frame is not None:
            similarity = get_frame_similarity(prev_frame, frame)
            if similarity > SIMILARITY_THRESHOLD:
                continue # 유사하면 건너뛰기
            
        print(f"  - 프레임 {i}: 세그멘테이션 진행...")
        
        h, w, _ = frame.shape
        
        # 'veo3' 모델 예측
        veo3_masks = []
        veo3_seg_results = veo3_model.predict(frame, conf=0.25, verbose=False)
        if veo3_seg_results[0].masks is not None:
            for mask_tensor in veo3_seg_results[0].masks.data:
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                veo3_masks.append(cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST))

        # 'no veo3' 모델 예측
        no_veo3_masks = []
        no_veo3_seg_results = no_veo3_model.predict(frame, conf=0.25, verbose=False)
        if no_veo3_seg_results[0].masks is not None:
            for mask_tensor in no_veo3_seg_results[0].masks.data:
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                no_veo3_masks.append(cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST))
        
        # 두 모델 중 하나라도 마스크가 있다면 결과 저장
        if veo3_masks or no_veo3_masks:
            output_path = OUTPUT_DIR / f"frame_{i:04d}_segmented.jpg"
            print("    - 시각화 및 저장 중...")
            visualize_segmentation_comparison(frame, veo3_masks, no_veo3_masks, output_path)
            processed_frame_count += 1

        prev_frame = frame

    cap.release()
    print(f"\n✅ 비디오 처리 완료!")
    print(f"  - 총 {processed_frame_count}개의 유효 프레임을 처리하여 저장했습니다.")
    print(f"  - 결과 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
