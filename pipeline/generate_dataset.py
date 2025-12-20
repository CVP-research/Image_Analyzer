"""
YOLO 세그멘테이션 데이터셋 생성기

객체 이미지를 배경에 합성하여 YOLO 학습용 데이터셋을 생성합니다.
- 다중 객체 폴더 지원 (리스트 또는 단일 경로)
- 이미지 조화(harmonization) 옵션 지원 (libcom 설치 시)
- 타일링 배경 생성
"""

import os
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Union, Optional

# libcom은 optional dependency
try:
    from libcom import image_harmonization
    LIBCOM_AVAILABLE = True
except ImportError:
    LIBCOM_AVAILABLE = False


# ----------------------------------------------------
# 1. 설정 (Configuration)
# ----------------------------------------------------
@dataclass
class DatasetConfig:
    """데이터셋 생성 설정"""
    bg_root: str = "./dataset"
    obj_root: Union[str, List[str]] = field(default_factory=lambda: ["./output/masked_frames"])
    output_root: str = "./dataset/train"

    target_size: int = 1024
    tile_size: int = 256
    num_bg_groups: int = 3000
    num_negative_samples: int = 2000
    val_split_ratio: float = 0.2
    class_id: int = 0

    # 객체 크기 범위 (캔버스 대각선 대비 비율)
    obj_size_min: float = 0.15
    obj_size_max: float = 0.35

    # 이미지 조화 사용 여부 (libcom 필요)
    use_harmonization: bool = True
    device: int = 0  # GPU device ID


# ----------------------------------------------------
# 2. DatasetBuilder 클래스 (핵심 로직)
# ----------------------------------------------------
class DatasetBuilder:
    def __init__(self, config: DatasetConfig):
        self.cfg = config
        self.harmonization_net = None

        # libcom harmonization 초기화
        if self.cfg.use_harmonization:
            if LIBCOM_AVAILABLE:
                print("Loading image harmonization model...")
                self.harmonization_net = image_harmonization.ImageHarmonizationModel(
                    device=self.cfg.device
                )
            else:
                print("Warning: libcom not installed, skipping harmonization")

        self._setup_folders()

        # 객체 경로 정규화 (문자열이면 리스트로 변환)
        obj_roots = self.cfg.obj_root
        if isinstance(obj_roots, str):
            obj_roots = [obj_roots]

        self.object_files = self._load_files(obj_roots, ('*.png',))
        self.bg_paths = self._load_files([self.cfg.bg_root], ('*.jpg', '*.png', '*.jpeg'))

        print(f"Found {len(self.object_files)} object files")
        print(f"Found {len(self.bg_paths)} background files")

        if not self.object_files:
            print("Warning: No object files found, will skip positive samples")

    def _load_files(self, root_dirs: List[str], extensions: tuple) -> List[str]:
        """지정된 확장자를 가진 파일 경로를 로드합니다."""
        files = []
        for root_dir in root_dirs:
            for ext in extensions:
                files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        return files

    def _setup_folders(self):
        """출력 폴더 구조를 설정합니다."""
        for subset in ['train', 'val']:
            os.makedirs(os.path.join(self.cfg.output_root, 'images', subset), exist_ok=True)
            os.makedirs(os.path.join(self.cfg.output_root, 'labels', subset), exist_ok=True)

    # --- 배경 관련 메서드 ---

    def _create_tiled_bg(self, bg_paths_chunk: List[str]) -> Optional[np.ndarray]:
        """배경 이미지들을 받아 타일링된 배경을 생성합니다."""
        rows = []
        idx = 0
        tile_size = self.cfg.tile_size
        tiles_per_row = self.cfg.target_size // tile_size

        for _ in range(tiles_per_row):
            cols = []
            for _ in range(tiles_per_row):
                if idx >= len(bg_paths_chunk):
                    break

                img = cv2.imread(bg_paths_chunk[idx])
                if img is None:
                    img = np.zeros((tile_size, tile_size, 3), np.uint8)

                img = cv2.resize(img, (tile_size, tile_size))
                cols.append(img)
                idx += 1

            if cols:
                rows.append(np.hstack(cols))

        if not rows:
            return None
        return np.vstack(rows)

    def _get_tiled_backgrounds(self, target_count: int) -> List[np.ndarray]:
        """타일링된 배경 리스트를 생성합니다."""
        tiles_needed = (self.cfg.target_size // self.cfg.tile_size) ** 2
        total_needed = target_count * tiles_needed

        if len(self.bg_paths) < total_needed:
            raise Exception(
                f"Not enough background images! Need {total_needed}, but only have {len(self.bg_paths)}"
            )

        random.shuffle(self.bg_paths)
        ready_backgrounds = []
        required_paths = self.bg_paths[:total_needed]

        print(f"Creating {target_count} tiled backgrounds...")
        for i in tqdm(range(0, len(required_paths), tiles_needed), desc="Tiling backgrounds"):
            chunk = required_paths[i:i + tiles_needed]
            if len(chunk) == tiles_needed:
                bg = self._create_tiled_bg(chunk)
                if bg is not None:
                    ready_backgrounds.append(bg)

        return ready_backgrounds

    # --- 객체 합성 및 라벨링 관련 메서드 ---

    def _resize_object_smart(self, obj_img_orig: np.ndarray) -> np.ndarray:
        """객체를 캔버스 크기에 맞춰 랜덤하게 리사이징합니다."""
        h, w = obj_img_orig.shape[:2]
        canvas_diag = np.sqrt(self.cfg.target_size**2 + self.cfg.target_size**2)
        obj_diag = np.sqrt(h**2 + w**2)

        target_diag = random.uniform(self.cfg.obj_size_min, self.cfg.obj_size_max) * canvas_diag
        scale = target_diag / obj_diag

        return cv2.resize(
            obj_img_orig,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    def _get_yolo_segmentation_label(self, mask: np.ndarray) -> Optional[str]:
        """마스크를 YOLO Segmentation 형식의 문자열로 변환합니다."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 가장 큰 컨투어만 선택
        cnt = max(contours, key=cv2.contourArea)

        # 컨투어를 단순화
        epsilon = 0.0001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            return None

        # 정규화된 좌표 문자열로 변환
        points_str = [
            f"{(p[0][0] / self.cfg.target_size):.6f} {(p[0][1] / self.cfg.target_size):.6f}"
            for p in approx
        ]

        return f"{self.cfg.class_id} " + " ".join(points_str)

    def _process_and_save(
        self,
        large_bg: np.ndarray,
        split_name: str,
        filename: str,
        is_positive: bool
    ) -> int:
        """이미지 합성, 조화, 라벨링, 저장을 처리하는 통합 메서드."""

        img_save_path = os.path.join(self.cfg.output_root, 'images', split_name, f"{filename}.jpg")
        lbl_save_path = os.path.join(self.cfg.output_root, 'labels', split_name, f"{filename}.txt")

        if is_positive:
            # --- 포지티브 샘플 처리 ---
            obj_path = random.choice(self.object_files)
            obj_img_orig = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            if obj_img_orig is None:
                return 0

            obj_img = self._resize_object_smart(obj_img_orig)
            h_obj, w_obj = obj_img.shape[:2]

            if h_obj >= self.cfg.target_size or w_obj >= self.cfg.target_size:
                return 0

            y_pos = random.randint(0, self.cfg.target_size - h_obj)
            x_pos = random.randint(0, self.cfg.target_size - w_obj)

            composite = large_bg.copy()

            # 마스크 생성
            mask = np.zeros((self.cfg.target_size, self.cfg.target_size), dtype=np.uint8)
            alpha_channel = obj_img[:, :, 3]
            _, clean_mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
            mask[y_pos:y_pos + h_obj, x_pos:x_pos + w_obj] = clean_mask

            # 객체 합성 (alpha blending)
            alpha_s = obj_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                composite[y_pos:y_pos + h_obj, x_pos:x_pos + w_obj, c] = (
                    alpha_s * obj_img[:, :, c] +
                    alpha_l * composite[y_pos:y_pos + h_obj, x_pos:x_pos + w_obj, c]
                )

            # 이미지 조화 (Harmonization) 적용
            if self.harmonization_net is not None:
                composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
                harmonized_rgb = self.harmonization_net(composite_rgb, mask)

                if harmonized_rgb.max() <= 1.0:
                    harmonized_rgb *= 255.0
                final_bgr = cv2.cvtColor(
                    np.clip(harmonized_rgb, 0, 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR
                )
            else:
                final_bgr = composite.astype(np.uint8)

            # 라벨링
            label_data = self._get_yolo_segmentation_label(mask)

            if label_data:
                cv2.imwrite(img_save_path, final_bgr)
                with open(lbl_save_path, "w") as f:
                    f.write(label_data + "\n")
                return 1
            return 0

        else:
            # --- 네거티브 샘플 처리 ---
            cv2.imwrite(img_save_path, large_bg)
            open(lbl_save_path, 'w').close()
            return 1

    # --- 메인 실행 메서드 ---

    def generate_datasets(self):
        """전체 데이터셋 생성 프로세스를 실행합니다."""

        total_generated = 0

        # 1. 포지티브 샘플 생성
        if self.object_files:
            positive_bgs = self._get_tiled_backgrounds(self.cfg.num_bg_groups)
            random.shuffle(positive_bgs)

            pos_split_idx = int(len(positive_bgs) * (1 - self.cfg.val_split_ratio))
            train_pos_bgs = positive_bgs[:pos_split_idx]
            val_pos_bgs = positive_bgs[pos_split_idx:]

            print(f"\n--- Generating positive samples (train: {len(train_pos_bgs)}, val: {len(val_pos_bgs)}) ---")

            for bg_list, split_name in [(train_pos_bgs, 'train'), (val_pos_bgs, 'val')]:
                for large_bg in tqdm(bg_list, desc=f"Generating positive {split_name}"):
                    filename = f"positive_{total_generated:06d}"
                    total_generated += self._process_and_save(large_bg, split_name, filename, is_positive=True)

        # 2. 네거티브 샘플 생성
        if self.cfg.num_negative_samples > 0:
            negative_bgs = self._get_tiled_backgrounds(self.cfg.num_negative_samples)
            random.shuffle(negative_bgs)

            neg_split_idx = int(len(negative_bgs) * (1 - self.cfg.val_split_ratio))
            train_neg_bgs = negative_bgs[:neg_split_idx]
            val_neg_bgs = negative_bgs[neg_split_idx:]

            print(f"\n--- Generating negative samples (train: {len(train_neg_bgs)}, val: {len(val_neg_bgs)}) ---")

            for bg_list, split_name in [(train_neg_bgs, 'train'), (val_neg_bgs, 'val')]:
                for large_bg in tqdm(bg_list, desc=f"Generating negative {split_name}"):
                    filename = f"negative_{total_generated:06d}"
                    total_generated += self._process_and_save(large_bg, split_name, filename, is_positive=False)

        print(f"\nDone! Generated {total_generated} images.")


# ----------------------------------------------------
# 3. 메인 실행 (Entry Point)
# ----------------------------------------------------

def run(config: Optional[DatasetConfig] = None):
    """데이터셋 생성 실행"""
    if config is None:
        config = DatasetConfig()

    builder = DatasetBuilder(config)
    builder.generate_datasets()


if __name__ == "__main__":
    run()
