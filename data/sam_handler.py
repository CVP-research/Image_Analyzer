import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import SAM
import sys

class SAMHandler:
    """
    SAM 모델 전문 핸들러

    - 모델 로드 및 캐싱
    - 이미지 분할 및 마스크 처리
    - 중앙 픽셀 기반 또는 최대 영역 객체 선택
    """

    SAM_MODEL_INSTANCE = None  # 클래스 단위 캐싱

    def __init__(self, sam_model_path: Path):
        self.sam_model_path = sam_model_path
        self.model = self._load_model()

    # -------------------- 모델 로드 --------------------
    def _load_model(self):
        if SAMHandler.SAM_MODEL_INSTANCE is None:
            print("[*] SAM 모델 로딩 중...")
            if not self.sam_model_path.exists():
                print(f"[오류] SAM 모델 파일이 존재하지 않습니다: {self.sam_model_path}")
                sys.exit(1)
            try:
                SAMHandler.SAM_MODEL_INSTANCE = SAM(str(self.sam_model_path))
                print("[*] SAM 모델 로딩 완료")
            except Exception as e:
                print(f"[오류] SAM 모델 로딩 중 에러 발생: {e}")
                sys.exit(1)
        return SAMHandler.SAM_MODEL_INSTANCE

    # -------------------- 단일 이미지 분할 --------------------
    def segment_image(self, img: np.ndarray) -> np.ndarray:
        """
        입력 이미지에서 가장 큰 객체 또는 중앙 픽셀 객체를 분할하여 RGBA 마스크 이미지 반환
        """
        pred_results = self.model.predict(img, task="segment", verbose=False)

        if not pred_results or not hasattr(pred_results[0], 'masks') or pred_results[0].masks is None:
            raise RuntimeError("객체를 감지할 수 없습니다.")

        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2

        selected_mask = None
        for mask in pred_results[0].masks.data:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            if mask_np[center_y, center_x] > 0:
                selected_mask = mask_np
                break

        # 중앙 픽셀 마스크 없으면 최대 영역 선택
        if selected_mask is None:
            largest_area = 0
            for mask in pred_results[0].masks.data:
                mask_np = mask.cpu().numpy().astype(np.uint8)
                area = np.sum(mask_np)
                if area > largest_area:
                    largest_area = area
                    selected_mask = mask_np

        if selected_mask is None:
            raise RuntimeError("유효한 객체 마스크를 찾지 못했습니다.")

        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(selected_mask, kernel, iterations=1)

        # RGBA 이미지 생성
        alpha = (mask_eroded * 255).astype(np.uint8)
        b, g, r = cv2.split(img)
        rgba = cv2.merge([b, g, r, alpha])

        # 객체 영역만 타이트하게 crop
        ys, xs = np.where(alpha > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cropped = rgba[y_min:y_max+1, x_min:x_max+1]
        else:
            cropped = rgba

        # OpenCV(BGRA) -> PIL(RGBA) 변환
        final_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA))
        return final_image

    # -------------------- 폴더 단위 분할 --------------------
    def segment_folder(self, input_dir: Path, output_dir: Path, verbose: bool = False):
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if not frames:
            print(f"[경고] {input_dir}에 처리할 이미지가 없습니다.")
            return

        for frame_path in frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"[경고] {frame_path.name} 읽기 실패, 건너뜀")
                continue
            try:
                segmented_img = self.segment_image(img)
                out_path = output_dir / f"{frame_path.stem}_masked.png"
                segmented_img.save(out_path)
                if verbose:
                    print(f"  - 저장 완료: {out_path.name}")
            except RuntimeError as e:
                print(f"[경고] {frame_path.name}: {e}")

        print(f"[*] 폴더 분할 완료. 결과: {output_dir}")
