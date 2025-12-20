import cv2
import numpy as np
from pathlib import Path
import hashlib
from PIL import Image
from ultralytics import SAM
import sys

# ==========================================
# 🚨 사용자 설정 영역 🚨
# 스크립트 실행 전, 아래 경로들을 올바르게 수정해주세요.
# ==========================================

# 1. 프레임을 추출할 입력 비디오 파일 경로
# 예: "input/my_video.mp4"
INPUT_VIDEO_PATH = Path("/home/rocknroll1397/Image_Analyzer/new/data/77_7859_15670/video.mp4")

# 2. 추출된 프레임이 저장될 중간 폴더 경로
#    - 이 폴더는 스크립트 실행 시 자동으로 생성됩니다.
EXTRACTED_FRAMES_DIR = Path("/home/rocknroll1397/Image_Analyzer/new/output/extracted_frames")

# 3. 최종적으로 배경이 제거된 객체 이미지가 저장될 폴더 경로
#    - 이 폴더는 스크립트 실행 시 자동으로 생성됩니다.
MASKED_OBJECTS_DIR = Path("/home/rocknroll1397/Image_Analyzer/new/output/masked_video_objects")
# 4. SAM 모델 가중치 파일 경로
#    - sam2_l.pt 또는 다른 SAM 모델 가중치 파일의 위치를 지정합니다.
SAM_MODEL_PATH = Path("/home/rocknroll1397/Image_Analyzer/sam2_l.pt")


# ==========================================
# 헬퍼 함수 (utils.py, segment.py, main.py에서 추출)
# ==========================================

def frame_from_video(video_path: Path, output_dir: Path) -> None:
    """
    비디오에서 고유한 프레임을 추출하여 저장합니다.
    동일한 프레임이 반복될 경우 중복 저장하지 않습니다.
    """
    print(f"[*] 비디오에서 프레임 추출을 시작합니다: {video_path.name}")
    if not video_path.exists():
        print(f"[오류] 비디오 파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    saved_idx = 0
    seen_hashes = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 해시를 계산하여 중복 여부 확인
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        if frame_hash in seen_hashes:
            frame_idx += 1
            continue  # 이미 처리된 프레임이면 건너뛰기

        seen_hashes.add(frame_hash)
        frame_filename = output_dir / f"frame_{saved_idx:04d}.png"
        cv2.imwrite(str(frame_filename), frame)
        # print(f"  - 고유 프레임 저장: {frame_filename.name}")
        frame_idx += 1
        saved_idx += 1

    cap.release()
    print(f"[*] 총 {saved_idx}개의 고유 프레임을 {output_dir}에 저장했습니다.")

SAM_MODEL_INSTANCE = None
def get_sam_model(model_path: Path):
    """
    SAM 모델을 로드하고 전역 변수에 캐싱합니다.
    """
    global SAM_MODEL_INSTANCE
    if SAM_MODEL_INSTANCE is None:
        print("[*] SAM 모델을 로딩합니다... (시간이 소요될 수 있습니다)")
        if not model_path.exists():
            print(f"[오류] SAM 모델 파일을 찾을 수 없습니다: {model_path}")
            print("SAM_MODEL_PATH 변수를 올바른 .pt 파일 경로로 설정해주세요.")
            sys.exit(1)
        try:
            SAM_MODEL_INSTANCE = SAM(str(model_path))
            print("[*] SAM 모델 로딩 완료.")
        except Exception as e:
            print(f"[오류] SAM 모델 로딩 중 에러가 발생했습니다: {e}")
            sys.exit(1)
            
    return SAM_MODEL_INSTANCE

def segment_objects(frame_dir: Path, output_dir: Path, model_path: Path) -> None:
    """
    주어진 폴더의 모든 프레임에서 가장 큰 객체를 분할(segment)하고,
    배경을 제거한(누끼) 이미지를 PNG로 저장합니다.
    """
    print(f"[*] {frame_dir} 폴더의 프레임들에서 객체 분할을 시작합니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = get_sam_model(model_path)
    
    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    if not frames:
        print(f"[경고] {frame_dir}에서 분할할 프레임을 찾지 못했습니다.")
        return

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"[경고] {frame_path.name} 파일을 읽을 수 없어 건너뜁니다.")
            continue

        # SAM 모델로 예측
        pred_results = model.predict(img, task="segment", verbose=False)

        if not pred_results or not hasattr(pred_results[0], 'masks') or pred_results[0].masks is None:
            print(f"[경고] {frame_path.name}에서 객체를 찾지 못했습니다.")
            continue

        # 이미지 중앙 픽셀을 포함하는 마스크 선택
        h, w, _ = img.shape
        image_center_x, image_center_y = w // 2, h // 2

        center_pixel_mask = None
        for mask in pred_results[0].masks.data:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            if mask_np[image_center_y, image_center_x] > 0:
                center_pixel_mask = mask_np
                break  # 첫 번째로 찾은 마스크를 사용

        # 중앙 픽셀을 포함하는 마스크가 없으면, 가장 큰 마스크를 선택
        if center_pixel_mask is None:
            print(f"[정보] {frame_path.name}: 중앙 픽셀을 포함하는 객체가 없어 가장 큰 객체를 선택합니다.")
            largest_area = 0
            for mask in pred_results[0].masks.data:
                mask_np = mask.cpu().numpy().astype(np.uint8)
                area = np.sum(mask_np)
                if area > largest_area:
                    largest_area = area
                    center_pixel_mask = mask_np

        if center_pixel_mask is None:
            print(f"[경고] {frame_path.name}에서 유효한 마스크를 찾지 못했습니다.")
            continue

        # 마스크 가장자리를 부드럽게 다듬기 (작은 노이즈 제거)
        kernel = np.ones((3, 3), np.uint8)
        largest_mask_eroded = cv2.erode(center_pixel_mask, kernel, iterations=1)
        
        # 원본 이미지와 마스크를 사용하여 배경이 투명한 RGBA 이미지 생성
        alpha_channel = (largest_mask_eroded * 255).astype(np.uint8)
        b, g, r = cv2.split(img)
        rgba_image = cv2.merge([b, g, r, alpha_channel])
        
        # 객체 영역만 타이트하게 잘라내기 (crop)
        ys, xs = np.where(alpha_channel > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cropped_rgba = rgba_image[y_min:y_max+1, x_min:x_max+1]
        else:
            # 마스크에 감지된 픽셀이 없으면 빈 이미지 대신 원본 RGBA 사용
            cropped_rgba = rgba_image

        # OpenCV(BGRA) -> PIL(RGBA) 변환 후 저장
        final_image_pil = Image.fromarray(cv2.cvtColor(cropped_rgba, cv2.COLOR_BGRA2RGBA))

        out_path = output_dir / f"{frame_path.stem}_masked.png"
        final_image_pil.save(out_path)
        # print(f"  - 객체 저장 완료: {out_path.name}")
        
    print(f"[*] 객체 분할 및 저장이 완료되었습니다. 결과는 {output_dir} 폴더를 확인하세요.")


# ==========================================
# 메인 실행부
# ==========================================
def main():
    """
    전체 파이프라인을 실행합니다:
    1. 비디오에서 프레임 추출
    2. 추출된 프레임에서 객체 분할 및 배경 제거 후 저장
    """
    print("="*50)
    print("비디오 객체 추출 파이프라인 시작")
    print(f"  - 입력 비디오: {INPUT_VIDEO_PATH}")
    print(f"  - 프레임 저장 위치: {EXTRACTED_FRAMES_DIR}")
    print(f"  - 최종 결과물 위치: {MASKED_OBJECTS_DIR}")
    print("="*50)

    # 1. 비디오 -> 프레임 추출
    frame_from_video(INPUT_VIDEO_PATH, EXTRACTED_FRAMES_DIR)
    
    # 2. 프레임 -> 배경 제거된 객체 추출
    segment_objects(EXTRACTED_FRAMES_DIR, MASKED_OBJECTS_DIR, SAM_MODEL_PATH)
    
    print("\n[성공] 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
