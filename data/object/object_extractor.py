import cv2
import numpy as np
from pathlib import Path
import hashlib
from PIL import Image
from ultralytics import SAM
from tqdm import tqdm
import sys

class VideoObjectExtractor:
    """
    비디오에서 프레임을 추출하고 SAM 모델을 사용하여 객체를 분할 및 배경 제거
    """

    SAM_MODEL_INSTANCE = None  # 클래스 단위 캐시

    def __init__(self,
                 input_video_path: Path,
                 extracted_frames_dir: Path,
                 masked_objects_dir: Path,
                 sam_model_path: Path):
        self.input_video_path = input_video_path
        self.extracted_frames_dir = extracted_frames_dir
        self.masked_objects_dir = masked_objects_dir
        self.sam_model_path = sam_model_path
        
        # 통계
        self.success_count = 0
        self.failed_count = 0
        self.failed_frames = []

        # 디렉토리 생성
        self.extracted_frames_dir.mkdir(parents=True, exist_ok=True)
        self.masked_objects_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- 프레임 추출 --------------------
    def extract_frames(self):
        print(f"[*] 비디오에서 프레임 추출을 시작합니다: {self.input_video_path.name}")
        if not self.input_video_path.exists():
            print(f"[오류] 비디오 파일을 찾을 수 없습니다: {self.input_video_path}")
            sys.exit(1)

        cap = cv2.VideoCapture(str(self.input_video_path))
        frame_idx = 0
        saved_idx = 0
        seen_hashes = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 해시 확인 (중복 제거)
            frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
            if frame_hash in seen_hashes:
                frame_idx += 1
                continue

            seen_hashes.add(frame_hash)
            frame_filename = self.extracted_frames_dir / f"frame_{saved_idx:04d}.png"
            cv2.imwrite(str(frame_filename), frame)
            frame_idx += 1
            saved_idx += 1

        cap.release()
        print(f"[*] 총 {saved_idx}개의 고유 프레임을 {self.extracted_frames_dir}에 저장했습니다.")

    # -------------------- SAM 모델 로드 --------------------
    def _get_sam_model(self):
        if VideoObjectExtractor.SAM_MODEL_INSTANCE is None:
            print("[*] SAM 모델을 로딩합니다... (시간이 소요될 수 있습니다)")
            if not self.sam_model_path.exists():
                print(f"[오류] SAM 모델 파일을 찾을 수 없습니다: {self.sam_model_path}")
                sys.exit(1)
            try:
                VideoObjectExtractor.SAM_MODEL_INSTANCE = SAM(str(self.sam_model_path))
                print("[*] SAM 모델 로딩 완료.")
            except Exception as e:
                print(f"[오류] SAM 모델 로딩 중 에러 발생: {e}")
                sys.exit(1)
        return VideoObjectExtractor.SAM_MODEL_INSTANCE

    # -------------------- 객체 분할 --------------------
    def segment_objects(self):
        print(f"[*] {self.extracted_frames_dir} 폴더의 프레임에서 객체 분할 시작")
        model = self._get_sam_model()

        frames = sorted([p for p in self.extracted_frames_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if not frames:
            print(f"[경고] {self.extracted_frames_dir}에서 처리할 프레임이 없습니다.")
            return

        for frame_path in tqdm(frames, desc="🔄 객체 분할 중", unit="프레임", ncols=80):
            img = cv2.imread(str(frame_path))
            if img is None:
                self.failed_count += 1
                self.failed_frames.append(frame_path.name)
                continue

            pred_results = model.predict(img, task="segment", verbose=False)
            if not pred_results or not hasattr(pred_results[0], 'masks') or pred_results[0].masks is None:
                self.failed_count += 1
                self.failed_frames.append(frame_path.name)
                continue

            # 중앙 픽셀 포함 마스크 선택
            h, w, _ = img.shape
            center_x, center_y = w // 2, h // 2
            center_mask = None
            for mask in pred_results[0].masks.data:
                mask_np = mask.cpu().numpy().astype(np.uint8)
                if mask_np[center_y, center_x] > 0:
                    center_mask = mask_np
                    break

            # 중앙 픽셀 마스크 없으면 가장 큰 마스크 사용
            if center_mask is None:
                largest_area = 0
                for mask in pred_results[0].masks.data:
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    area = np.sum(mask_np)
                    if area > largest_area:
                        largest_area = area
                        center_mask = mask_np

            if center_mask is None:
                self.failed_count += 1
                self.failed_frames.append(frame_path.name)
                continue

            # 마스크 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            mask_eroded = cv2.erode(center_mask, kernel, iterations=1)

            # RGBA 이미지 생성
            alpha = (mask_eroded * 255).astype(np.uint8)
            b, g, r = cv2.split(img)
            rgba = cv2.merge([b, g, r, alpha])

            # 타이트하게 crop
            ys, xs = np.where(alpha > 0)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                cropped = rgba[y_min:y_max+1, x_min:x_max+1]
            else:
                self.failed_count += 1
                self.failed_frames.append(frame_path.name)
                continue

            # OpenCV(BGRA) -> PIL(RGBA) 저장
            final_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA))
            out_path = self.masked_objects_dir / f"{frame_path.stem}_masked.png"
            final_image.save(out_path)
            self.success_count += 1

        # 결과 요약
        print(f"\n[*] 객체 분할 완료!")
        print(f"    ✅ 성공: {self.success_count}개")
        print(f"    ❌ 실패 (자동 스킵): {self.failed_count}개")
        if self.failed_frames and len(self.failed_frames) <= 10:
            print(f"    실패 프레임: {', '.join(self.failed_frames)}")
        print(f"    결과 저장 위치: {self.masked_objects_dir}")

    # -------------------- 전체 실행 --------------------
    def run_pipeline(self):
        self.extract_frames()
        self.segment_objects()
