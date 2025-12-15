import os
import cv2
import numpy as np
from pathlib import Path
import glob
from PIL import Image

# 🚨 주의: SAM 모델 로드를 위한 함수입니다. 실제 환경에 맞게 구현하거나 수정해야 합니다.
# 예시: YOLOv8-SAM 통합 모델을 로드하는 함수
def get_sam_model():
    # from ultralytics import YOLO
    # model = YOLO('yolov8n-seg.pt')
    # return model
    raise NotImplementedError("SAM 모델 로드 함수를 실제 환경에 맞게 구현해주세요.")

# --- 1. 설정 변수 (사용자 수정 필요) ---
# CO3D 데이터셋 구조를 가정합니다.
BASE_DATA_DIR = Path("./data/77_7859_15670")
IMAGE_SUBDIR = "images" 
MASK_SUBDIR = "masks"
OUTPUT_MASKED_DIR = Path("./output/masked_frames/")

# --- 2. 주요 함수: 누끼 따기 (객체만 분리) ---

def process_frame(frame_path: Path, mask_path: Path, model):
    """
    단일 이미지 프레임과 해당 마스크를 처리하여 객체만 추출합니다.
    (SAM 모델을 사용하여 마스크가 없는 프레임을 처리할 수도 있지만,
    여기서는 CO3D 마스크 파일을 우선적으로 사용합니다.)
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"[WARN] 이미지 파일 읽기 실패: {frame_path.name}")
        return None

    # 1. 마스크 파일 읽기 및 안정화 (CO3D 비표준 값 처리)
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            # 마스크 파일이 없거나 읽을 수 없으면 SAM 모델로 폴백(fallback)
            print(f"[INFO] 마스크 파일 없음/읽기 실패. SAM 모델로 분할 시도: {frame_path.name}")
            return segment_with_sam(img, frame_path.name, model)
        
        # 🚨 마스크 값 안정화: 0보다 큰 모든 값을 객체로 처리 (깊이/ID 무시)
        # 이진 마스크 (0 또는 255) 생성
        binary_mask = ((mask > 0) * 255).astype(np.uint8)

        # 마스크 내 객체 픽셀 수 확인 (전체 픽셀 수가 0이면 객체 없음)
        if np.sum(binary_mask) == 0:
            print(f"[WARN] 마스크 파일이 모두 0입니다 (객체 없음). SAM 모델로 재시도: {frame_path.name}")
            return segment_with_sam(img, frame_path.name, model)
            
    except Exception as e:
        print(f"[ERROR] 마스크 처리 중 오류 발생: {e}. SAM 모델로 폴백.")
        return segment_with_sam(img, frame_path.name, model)

    # 2. 객체 추출 및 PNG 생성 (마스크가 정상적으로 준비됨)
    return extract_object_from_mask(img, binary_mask, frame_path.name)


def segment_with_sam(img: np.ndarray, frame_name: str, model):
    """
    마스크 파일을 찾을 수 없을 때 SAM 모델을 사용하여 객체를 분할합니다.
    가장 큰 마스크를 선택하는 로직을 구현합니다.
    """
    try:
        pred_results = model.predict(img, task="segment")
    except Exception as e:
        print(f"[ERROR] SAM 예측 실패. Skipping. Error: {e}")
        return None

    if not pred_results or not hasattr(pred_results[0], 'masks') or pred_results[0].masks is None:
        return None

    # 🚨 핵심 로직: 가장 많은 픽셀을 가진 마스크 선택
    largest_mask_np = None
    largest_area = 0
    
    # YOLO/SAM 출력 구조를 가정하여 반복
    for mask_tensor in pred_results[0].masks.data:
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
        area = mask_np.sum()
        
        # 흰색 픽셀(1)의 개수가 가장 많은 마스크를 선택
        if area > largest_area:
            largest_area = area
            largest_mask_np = mask_np

    if largest_mask_np is None:
        return None
        
    # 마스크가 1/0 값일 경우 255/0으로 확장
    binary_mask = (largest_mask_np * 255).astype(np.uint8)
    
    print(f"[INFO] SAM 모델로 {frame_name} 객체 분리 완료. 마스크 영역: {largest_area} 픽셀.")
    return extract_object_from_mask(img, binary_mask, frame_name)


def extract_object_from_mask(img: np.ndarray, binary_mask: np.ndarray, frame_name: str):
    """이진 마스크를 사용하여 객체만 투명 PNG로 추출 및 크롭합니다."""
    
    # 1. RGBA 이미지 생성 (객체 영역은 255, 배경은 0)
    bgra_image = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra_image[:, :, 3] = binary_mask

    # 2. 투명 픽셀 제거를 위한 크롭
    alpha_channel = binary_mask
    ys, xs = np.where(alpha_channel > 0)
    
    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        # 1픽셀 여유를 주어 경계선이 잘리지 않도록 함
        bgra_image_cropped = bgra_image[y_min:y_max+1, x_min:x_max+1]
    else:
        # 객체가 없거나 너무 작으면 크롭하지 않음
        bgra_image_cropped = bgra_image

    # 3. 저장
    out_path = OUTPUT_MASKED_DIR / f"{Path(frame_name).stem}_masked.png"
    cv2.imwrite(str(out_path), bgra_image_cropped)
    
    original_size = bgra_image.shape[:2]
    cropped_size = bgra_image_cropped.shape[:2]
    print(f"✓ Segmented and saved: {out_path.name} ({original_size[0]}x{original_size[1]} → {cropped_size[0]}x{cropped_size[1]})")
    
    return out_path


# --- 3. 메인 실행 함수 ---

def main():
    """메인 실행 흐름: 이미지와 마스크 파일을 순회하며 객체를 분리합니다."""
    
    OUTPUT_MASKED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 파일 경로 정의
    image_dir = BASE_DATA_DIR / IMAGE_SUBDIR
    mask_dir = BASE_DATA_DIR / MASK_SUBDIR
    
    if not image_dir.exists():
        print(f"❌ 오류: 이미지 폴더를 찾을 수 없습니다: {image_dir}")
        return

    # 🚨 SAM 모델 로드 (실패 시 종료)
    try:
        model = get_sam_model()
    except NotImplementedError:
        print("❌ 오류: SAM 모델 로드 함수가 구현되지 않았습니다. SAM 기능을 사용할 수 없습니다.")
        # SAM 기능 없이 마스크 파일만 처리하는 것으로 진행
        model = None
    except Exception as e:
        print(f"❌ 오류: SAM 모델 로드 중 예기치 않은 오류 발생: {e}")
        model = None
        
    frames = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    if not frames:
        print(f"❌ 오류: 이미지 폴더에 처리할 프레임이 없습니다.")
        return

    print(f"--- {len(frames)}개의 프레임 분할 시작 ---")
    
    for frame_path in frames:
        frame_stem = frame_path.stem
        # 마스크 파일 경로 생성: 이미지 파일 이름과 동일하고 확장자만 .png로 가정
        mask_path = mask_dir / f"{frame_stem}.png" 
        
        # 객체 분리 및 저장
        process_frame(frame_path, mask_path, model)


if __name__ == "__main__":
    main()