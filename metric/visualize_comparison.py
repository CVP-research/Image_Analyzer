
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import glob

# ==========================================
# 🚨 사용자 설정 영역 🚨
# ==========================================

# 1. 'veo3' 및 'no veo3' 모델의 .pt 파일 경로
VEO3_MODEL_PATH = Path("/home/rocknroll1397/Image_Analyzer/runs/segment/train_veo3/weights/best.pt")
NO_VEO3_MODEL_PATH = Path("/home/rocknroll1397/Image_Analyzer/runs/segment/train_no_veo3/weights/best.pt")

# 2. 평가할 원본 이미지들이 있는 폴더 경로
INPUT_IMAGES_DIR = Path("data/77_7859_15670/images_real")

# 3. 결과물을 저장할 폴더 경로
OUTPUT_DIR = Path("output/segmentation_visualization")

# ==========================================
# 핵심 시각화 함수
# ==========================================

def visualize_segmentation_comparison(image, veo3_masks, no_veo3_masks, output_path):
    """
    'veo3'와 'no veo3' 세그멘테이션 결과를 비교하여 시각화합니다.
    - 'no veo3' 마스크: 초록색 외곽선
    - 'veo3' 마스크: 빨간색 외곽선
    """
    vis_image = image.copy()

    # 'no veo3' 마스크 외곽선 그리기 (초록색)
    for mask in no_veo3_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

    # 'veo3' 마스크 외곽선 그리기 (빨간색)
    for mask in veo3_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 0, 255), 2)

    cv2.imwrite(str(output_path), vis_image)

# ==========================================
# 메인 로직
# ==========================================

def main():
    """메인 시각화 프로세스를 실행합니다."""
    print("🚀 세그멘테이션 결과 비교 시각화를 시작합니다...")

    # 1. 설정 확인 및 폴더 생성
    if not VEO3_MODEL_PATH.exists():
        print(f"❌ 오류: 'veo3' 모델 파일을 찾을 수 없습니다: {VEO3_MODEL_PATH}")
        return
    if not NO_VEO3_MODEL_PATH.exists():
        print(f"❌ 오류: 'no veo3' 모델 파일을 찾을 수 없습니다: {NO_VEO3_MODEL_PATH}")
        return
    if not INPUT_IMAGES_DIR.exists():
        print(f"❌ 오류: 이미지 폴더를 찾을 수 없습니다: {INPUT_IMAGES_DIR}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 2. 모델 로드
    print(f"🔍 'veo3' 모델 로딩 중: {VEO3_MODEL_PATH}")
    veo3_model = YOLO(VEO3_MODEL_PATH)
    print(f"🔍 'no veo3' 모델 로딩 중: {NO_VEO3_MODEL_PATH}")
    no_veo3_model = YOLO(NO_VEO3_MODEL_PATH)

    # 3. 이미지 순회 및 시각화
    image_paths = sorted(list(INPUT_IMAGES_DIR.glob("*.jpg")) + list(INPUT_IMAGES_DIR.glob("*.png")))

    print(f"🖼️ 총 {len(image_paths)}개의 이미지에 대해 시각화를 진행합니다.")
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape

        # 'veo3' 모델 예측
        veo3_results = veo3_model.predict(image, conf=0.25)
        veo3_masks = []
        if veo3_results[0].masks is not None:
            for mask_tensor in veo3_results[0].masks.data:
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                if mask_np.shape[:2] != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                veo3_masks.append(mask_np)

        # 'no veo3' 모델 예측
        no_veo3_results = no_veo3_model.predict(image, conf=0.25)
        no_veo3_masks = []
        if no_veo3_results[0].masks is not None:
            for mask_tensor in no_veo3_results[0].masks.data:
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                if mask_np.shape[:2] != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                no_veo3_masks.append(mask_np)

        # 시각화 결과 저장
        output_path = OUTPUT_DIR / f"{image_path.stem}_comparison.jpg"
        visualize_segmentation_comparison(image, veo3_masks, no_veo3_masks, output_path)

    print("\n✅ 시각화 완료!")
    print(f"  - 결과 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
