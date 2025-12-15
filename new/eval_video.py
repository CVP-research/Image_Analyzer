import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 🚨 사용자 설정 영역 🚨
# 스크립트 실행 전, 아래 경로들을 올바르게 수정해주세요.
# ==========================================

# 1. 비교할 모델들의 경로
# 사용자가 직접 모델의 실제 경로로 수정해야 합니다.
MODEL_PATHS = {
    "veo3": Path("/home/rocknroll1397/Image_Analyzer/runs/segment/train_veo3/weights/best.pt"),
    "veo3_v1": Path("/home/rocknroll1397/Image_Analyzer/runs/segment/train_veo3_v1/weights/best.pt"),
    "veo3_v2": Path("/home/rocknroll1397/Image_Analyzer/runs/segment/train_veo3_v2/weights/best.pt"),
}

# 2. 각 모델의 시각화 색상 (BGR 순서)
MODEL_COLORS = {
    "veo3": (255, 0, 0),    # 파란색 (BGR)
    "veo3_v1": (0, 0, 255),   # 빨간색
    "veo3_v2": (0, 255, 0),   # 초록색
}

# 3. 처리할 입력 비디오 파일 경로
INPUT_VIDEO_PATH = Path("/home/rocknroll1397/Image_Analyzer/new/data/77_7859_15670/test4.mp4")

# 4. 결과 비교 이미지를 저장할 디렉토리
OUTPUT_FRAMES_DIR = Path("output/fake_test8/")

# ==========================================
# 메인 비교 로직
# ==========================================

def main():
    """
    여러 YOLO 모델의 segmentation 결과를 비디오의 각 프레임에 시각화하여
    이미지 파일로 저장합니다.
    """
    print("🚀 다중 모델 비디오 분할 비교를 시작합니다...")

    # 1. 설정 확인 및 폴더 생성
    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            print(f"❌ 경고: '{model_name}' 모델을 찾을 수 없습니다: {model_path}")
            print("   -> 해당 모델 없이 진행합니다. 모델 경로를 확인해주세요.")
    
    if not INPUT_VIDEO_PATH.exists():
        print(f"❌ 오류: 입력 비디오 파일을 찾을 수 없습니다: {INPUT_VIDEO_PATH}")
        return

    OUTPUT_FRAMES_DIR.mkdir(exist_ok=True)

    # 2. 모델 로드
    loaded_models = {}
    for model_name, model_path in MODEL_PATHS.items():
        if model_path.exists():
            print(f"🔍 '{model_name}' 모델 로딩 중...")
            loaded_models[model_name] = YOLO(model_path)
    
    if not loaded_models:
        print("❌ 오류: 로드할 수 있는 모델이 하나도 없습니다. 스크립트를 종료합니다.")
        return

    # 3. 비디오 캡처 설정
    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    if not cap.isOpened():
        print(f"❌ 오류: 비디오를 열 수 없습니다: {INPUT_VIDEO_PATH}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 비디오 처리 중: {INPUT_VIDEO_PATH.name} ({w}x{h} @ {fps:.2f}fps)")
    count_results = {
        "veo3": 0,
        "veo3_v1": 0,
        "veo3_v2": 0
    }
    # 4. 비디오 프레임 순회 및 처리
    frame_number = 0
    with tqdm(total=frame_count, desc="비교 이미지 생성 중") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            vis_frame = frame.copy()

            # 각 모델에 대해 예측 및 시각화
            temp = []
            for model_name, model in loaded_models.items():
                # 예측 수행
                results = model.predict(frame, verbose=False)
                
                if results[0].masks is None:
                    continue

                temp.append(results[0].masks)
                count_results[model_name] += 1

                # 윤곽선 그리기
                color = MODEL_COLORS.get(model_name, (255, 255, 255)) # 기본값 흰색
                for mask in results[0].masks.data:
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    # 모델이 리사이즈된 이미지에서 마스크를 생성했을 수 있으므로, 원본 프레임 크기로 리사이즈
                    if mask_np.shape[:2] != (h, w):
                        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_frame, contours, -1, color, 2)
            if not temp:
                pbar.update(1)
                continue
            # 결과 프레임 이미지로 저장
            output_path = OUTPUT_FRAMES_DIR / f"frame_{frame_number:05d}.jpg"
            cv2.imwrite(str(output_path), vis_frame)
            
            frame_number += 1
            pbar.update(1)

    # 5. 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

    print(count_results)

    print("\n✅ 비교 이미지 생성이 완료되었습니다!")
    print(f"  -> 결과물 저장 위치: {OUTPUT_FRAMES_DIR.resolve()}")

if __name__ == "__main__":
    main()