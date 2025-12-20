import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm

# ==========================================
# 🚨 사용자 설정 영역 🚨
# 스크립트 실행 전, 아래 경로들을 올바르게 수정해주세요.
# ==========================================

# 학습 폴더명
# TRAIN_NAME = "train_no_veo3_2"
TRAIN_NAME = "train_veo3"
# TRAIN_NAME = "train_veo3_v1"
# TRAIN_NAME = "train_veo3_v2"
# 1. 파인튜닝된 YOLO segmentation 모델의 .pt 파일 경로
FINE_TUNED_MODEL_PATH = Path(f"/home/rocknroll1397/Image_Analyzer/runs/segment/{TRAIN_NAME}/weights/best.pt")

VAL_DATA_YAML = f"/home/rocknroll1397/Image_Analyzer/new/dataset/{TRAIN_NAME}/data.yaml"

# 2. 평가할 원본 이미지들이 있는 폴더 경로
INPUT_IMAGES_DIR = Path("data/77_7859_15670/images_real")

# 3. 실제 정답 마스크(.png)들이 있는 폴더 경로
#    - segment_server_real.py에서 저장한 마스크들이 있는 폴더를 지정합니다.
#    - 파일명 형식: {원본_이미지명}_{인덱스}_mask.png (예: image_01_00_mask.png)
GROUND_TRUTH_MASKS_DIR = Path("output/masked_frames_real")

# 4. 결과물을 저장할 폴더 경로 (시각화 이미지, AP 결과 텍스트 파일)
OUTPUT_DIR = Path(f"output/evaluation_results_{TRAIN_NAME}")

# 5. AP 계산을 위한 IoU 임계값
IOU_THRESHOLD = 0.75

# ==========================================
# 메인 평가 로직
# ==========================================

def main():
    """메인 평가 프로세스를 실행합니다."""
    print("🚀 모델 평가를 시작합니다...")
    
    # 1. 설정 확인 및 폴더 생성
    if not FINE_TUNED_MODEL_PATH.exists():
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {FINE_TUNED_MODEL_PATH}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    vis_dir = OUTPUT_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # 2. 모델 로드
    print(f"🔍 모델 로딩 중: {FINE_TUNED_MODEL_PATH}")
    model = YOLO(FINE_TUNED_MODEL_PATH)

    #     # 시각화 결과 저장
    #     visualize_results(image, pred_masks, gt_masks, vis_dir / f"{image_path.stem}_result.jpg")

    val_results = model.val(
        data=VAL_DATA_YAML,
        imgsz=1024,
        conf=0.001,
        iou=IOU_THRESHOLD,
        device=0,
        split="test",
        save_json=True,
        verbose=True
    )

    print(f"\n✅ 평가 완료!")
    print(f"Precision: {val_results.seg.p}")
    print(f"Recall   : {val_results.seg.r}")
    print(f"F1 Score : {val_results.seg.f1}")
    print(f"mAP50-95: {val_results.seg.map}")
    print(f"mAP50    : {val_results.seg.map50}")
    print(f"mAP75    : {val_results.seg.map75}")

if __name__ == "__main__":
    main()
