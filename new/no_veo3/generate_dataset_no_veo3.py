import os
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm
from libcom import image_harmonization

# ----------------------------------------------------
# 1. 설정 (Configuration)
# ----------------------------------------------------
class Config:
    BG_ROOT = "/home/rocknroll1397/Image_Analyzer/dataset"
    OBJ_ROOT = "../output/masked_frames"
    OUTPUT_ROOT = "../dataset/train_no_veo3_2"
    
    TARGET_SIZE = 1024
    TILE_SIZE = 256
    NUM_BG_GROUPS = 3000
    NUM_NEGATIVE_SAMPLES = 2000
    REPEAT_PER_BG = 1
    VAL_SPLIT_RATIO = 0.2
    CLASS_ID = 0

# ----------------------------------------------------
# 2. DatasetBuilder 클래스 (핵심 로직)
# ----------------------------------------------------
class DatasetBuilder:
    def __init__(self, config):
        self.cfg = config
        print("🚀 이미지 조화 모델 로드 중...")
        # device=0는 GPU 사용을 가정합니다.
        self.harmonization_net = image_harmonization.ImageHarmonizationModel(device=0)
        self._setup_folders()
        
        self.object_files = self._load_files(self.cfg.OBJ_ROOT, ('*.png'))
        self.bg_paths = self._load_files(self.cfg.BG_ROOT, ('*.jpg', '*.png', '*.jpeg'))
        
        if not self.object_files:
            print("⚠️ 객체 파일이 없어 포지티브 샘플 생성을 건너뜁니다.")

    def _load_files(self, root_dir, extensions):
        """지정된 확장자를 가진 파일 경로를 로드합니다."""
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        return files

    def _setup_folders(self):
        """출력 폴더 구조를 설정합니다."""
        for subset in ['train', 'val']:
            os.makedirs(os.path.join(self.cfg.OUTPUT_ROOT, 'images', subset), exist_ok=True)
            os.makedirs(os.path.join(self.cfg.OUTPUT_ROOT, 'labels', subset), exist_ok=True)

    # --- 배경 관련 메서드 ---

    def _create_1024_tiled_bg(self, bg_paths_chunk):
        """16장의 배경 경로를 받아 1024x1024 타일 배경을 생성합니다."""
        rows = []
        idx = 0
        tile_size = self.cfg.TILE_SIZE
        
        for _ in range(self.cfg.TARGET_SIZE // tile_size):
            cols = []
            for _ in range(self.cfg.TARGET_SIZE // tile_size):
                if idx >= len(bg_paths_chunk): break
                
                img = cv2.imread(bg_paths_chunk[idx])
                if img is None: 
                    # 파일 로드 실패 시 검은색 타일 사용
                    img = np.zeros((tile_size, tile_size, 3), np.uint8)
                
                img = cv2.resize(img, (tile_size, tile_size))
                cols.append(img)
                idx += 1
            if cols:
                rows.append(np.hstack(cols))
            
        if not rows: return None
        return np.vstack(rows)

    def _get_tiled_backgrounds(self, target_count):
        """타일링된 배경 리스트를 생성합니다."""
        if len(self.bg_paths) < target_count * 16:
            raise Exception(f"❌ 배경 이미지가 부족합니다! {target_count*16}개 필요한데, {len(self.bg_paths)}개밖에 없습니다.")
            
        random.shuffle(self.bg_paths)
        ready_backgrounds = []
        required_paths = self.bg_paths[:target_count * 16]

        print(f"🧱 타일 배경 {target_count}개 생성 중...")
        for i in tqdm(range(0, len(required_paths), 16), desc="Tiling backgrounds"):
            chunk = required_paths[i : i+16]
            if len(chunk) == 16:
                bg = self._create_1024_tiled_bg(chunk)
                if bg is not None:
                    ready_backgrounds.append(bg)
        
        return ready_backgrounds

    # --- 객체 합성 및 라벨링 관련 메서드 ---

    def _resize_object_smart(self, obj_img_orig):
        """객체를 캔버스 크기에 맞춰 랜덤하게 리사이징합니다."""
        h, w = obj_img_orig.shape[:2]
        canvas_diag = np.sqrt(self.cfg.TARGET_SIZE**2 + self.cfg.TARGET_SIZE**2)
        obj_diag = np.sqrt(h**2 + w**2)
        
        # 객체 크기 범위 조정 (원래 30%~60% -> 15%~35%로 조정)
        target_diag = random.uniform(0.15, 0.35) * canvas_diag 
        scale = target_diag / obj_diag
        
        return cv2.resize(obj_img_orig, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _get_yolo_segmentation_label(self, mask):
        """마스크를 YOLO Segmentation 형식의 문자열로 변환합니다."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return None
        
        # 가장 큰 컨투어만 선택
        cnt = max(contours, key=cv2.contourArea)
        
        # 컨투어를 단순화하여 점의 개수 줄이기 (파일 크기 및 처리 속도 개선)
        epsilon = 0.0001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) < 3: return None
        
        # 정규화된 좌표 문자열로 변환
        points_str = [f"{(p[0][0]/self.cfg.TARGET_SIZE):.6f} {(p[0][1]/self.cfg.TARGET_SIZE):.6f}" 
                      for p in approx]
        
        return f"{self.cfg.CLASS_ID} " + " ".join(points_str)

    def _process_and_save(self, large_bg, split_name, filename, is_positive):
        """이미지 합성, 조화, 라벨링, 저장을 처리하는 통합 메서드."""
        
        img_save_path = os.path.join(self.cfg.OUTPUT_ROOT, 'images', split_name, f"{filename}.jpg")
        lbl_save_path = os.path.join(self.cfg.OUTPUT_ROOT, 'labels', split_name, f"{filename}.txt")

        if is_positive:
            # --- 포지티브 샘플 처리 ---
            obj_path = random.choice(self.object_files)
            obj_img_orig = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            if obj_img_orig is None: return 0

            obj_img = self._resize_object_smart(obj_img_orig)
            h_obj, w_obj = obj_img.shape[:2]
            
            if h_obj >= self.cfg.TARGET_SIZE or w_obj >= self.cfg.TARGET_SIZE: return 0

            y_pos, x_pos = random.randint(0, self.cfg.TARGET_SIZE - h_obj), random.randint(0, self.cfg.TARGET_SIZE - w_obj)

            composite = large_bg.copy()
            
            # 마스크 및 합성 로직
            mask = np.zeros((self.cfg.TARGET_SIZE, self.cfg.TARGET_SIZE), dtype=np.uint8)
            alpha_channel = obj_img[:, :, 3]
            _, clean_mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
            
            # 마스크 업데이트
            mask[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj] = clean_mask
            
            # 객체 합성 (alpha blending)
            alpha_s = obj_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c] = \
                    (alpha_s * obj_img[:, :, c] + alpha_l * composite[y_pos:y_pos+h_obj, x_pos:x_pos+w_obj, c])

            # 이미지 조화 (Harmonization) 적용
            composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            harmonized_rgb = self.harmonization_net(composite_rgb, mask)
            
            if harmonized_rgb.max() <= 1.0: harmonized_rgb *= 255.0
            final_bgr = cv2.cvtColor(np.clip(harmonized_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

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
            open(lbl_save_path, 'w').close() # 빈 라벨 파일 생성
            return 1

    # --- 메인 실행 메서드 ---

    def generate_datasets(self):
        """전체 데이터셋 생성 프로세스를 실행합니다."""
        
        total_generated = 0
        
        # 1. 포지티브 샘플 생성
        if self.object_files:
            positive_bgs = self._get_tiled_backgrounds(self.cfg.NUM_BG_GROUPS)
            random.shuffle(positive_bgs)

            # 데이터 분리
            pos_split_idx = int(len(positive_bgs) * (1 - self.cfg.VAL_SPLIT_RATIO))
            train_pos_bgs, val_pos_bgs = positive_bgs[:pos_split_idx], positive_bgs[pos_split_idx:]

            print(f"\n--- 포지티브 샘플 생성 시작 (훈련: {len(train_pos_bgs)}, 검증: {len(val_pos_bgs)}) ---")
            
            for bg_list, split_name in [(train_pos_bgs, 'train'), (val_pos_bgs, 'val')]:
                for large_bg in tqdm(bg_list, desc=f"Generating positive {split_name}"):
                    filename = f"positive_{total_generated:06d}"
                    total_generated += self._process_and_save(large_bg, split_name, filename, is_positive=True)
                    
        # 2. 네거티브 샘플 생성
        if self.cfg.NUM_NEGATIVE_SAMPLES > 0:
            negative_bg_count = self.cfg.NUM_NEGATIVE_SAMPLES
            negative_bgs = self._get_tiled_backgrounds(negative_bg_count)
            random.shuffle(negative_bgs)

            neg_split_idx = int(len(negative_bgs) * (1 - self.cfg.VAL_SPLIT_RATIO))
            train_neg_bgs, val_neg_bgs = negative_bgs[:neg_split_idx], negative_bgs[neg_split_idx:]

            print(f"\n--- 네거티브 샘플 생성 시작 (훈련: {len(train_neg_bgs)}, 검증: {len(val_neg_bgs)}) ---")

            for bg_list, split_name in [(train_neg_bgs, 'train'), (val_neg_bgs, 'val')]:
                for large_bg in tqdm(bg_list, desc=f"Generating negative {split_name}"):
                    filename = f"negative_{total_generated:06d}"
                    total_generated += self._process_and_save(large_bg, split_name, filename, is_positive=False)


        print(f"\n✅✅✅ 작업 완료! 총 {total_generated}개의 이미지가 생성되었습니다. ✅✅✅")


# ----------------------------------------------------
# 3. 메인 실행 (Entry Point)
# ----------------------------------------------------

if __name__ == "__main__":
    builder = DatasetBuilder(Config())
    builder.generate_datasets()