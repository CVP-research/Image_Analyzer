import cv2
from pathlib import Path
import shutil

# TRAIN_NAME = "train_no_veo3_2"
# TRAIN_NAME = "train_veo3"
# TRAIN_NAME = "train_veo3_v1"
TRAIN_NAME = "train_veo3_v2"


# ===============================
# 사용자 설정
# ===============================
IMAGE_DIR = Path("data/77_7859_15670/images_real")          # 원본 이미지 폴더
MASK_DIR = Path("output/masked_frames_real")             # 0/255 GT mask 폴더

OUT_IMG_DIR = Path(f"/home/rocknroll1397/Image_Analyzer/new/dataset/{TRAIN_NAME}/images/test")     # YOLO용 이미지 저장 폴더
OUT_LBL_DIR = Path(f"/home/rocknroll1397/Image_Analyzer/new/dataset/{TRAIN_NAME}/labels/test")     # YOLO용 라벨 저장 폴더

CLASS_ID = 0   # segmentation class id

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# 유틸
# ===============================
def mask_to_yolo_polygons(mask):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        cnt = cnt.squeeze(1)
        poly = []
        for x, y in cnt:
            poly.append(x / w)
            poly.append(y / h)
        polygons.append(poly)
    return polygons

# ===============================
# 메인 처리
# ===============================
print(list(IMAGE_DIR.glob("*")))

for img_path in IMAGE_DIR.glob("*"):
    if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
        print(2)
        continue

    mask_paths = sorted(MASK_DIR.glob(f"{img_path.stem}_*_mask.png"))
    if len(mask_paths) == 0:
        print(1)
        continue

    polygons = []

    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        polygons.extend(mask_to_yolo_polygons(mask))

    if len(polygons) == 0:
        continue

    # 이미지 복사
    shutil.copy(img_path, OUT_IMG_DIR / img_path.name)

    # 라벨 저장
    label_path = OUT_LBL_DIR / f"{img_path.stem}.txt"
    with open(label_path, "w") as f:
        for poly in polygons:
            line = str(CLASS_ID) + " " + " ".join([f"{v:.6f}" for v in poly])
            f.write(line + "\n")

print("✅ YOLO segmentation validation 데이터 생성 완료")
