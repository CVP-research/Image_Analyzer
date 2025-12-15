import cv2
import torch
import numpy as np
from realesrgan import RealESRGAN
from PIL import Image
import os

# =========================
# 1. 상수 설정
# =========================
IMAGE_PATH = "/home/rocknroll1397/Image_Analyzer/dataset/train/amusement_park/00000053.jpg"   # 👈 비교할 저해상도 이미지
OUTPUT_DIR = "./upscale_test"
UPSCALE_FACTOR = 4                  # x4 업스케일

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 이미지 로드
# =========================
img_bgr = cv2.imread(IMAGE_PATH)
assert img_bgr is not None, "이미지 로드 실패"

h, w = img_bgr.shape[:2]
new_size = (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR)

# =========================
# 3. OpenCV (LANCZOS) 업스케일
# =========================
opencv_upscaled = cv2.resize(
    img_bgr,
    new_size,
    interpolation=cv2.INTER_LANCZOS4
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "upscale_opencv_lanczos.png"),
    opencv_upscaled
)

# =========================
# 4. Real-ESRGAN 업스케일
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RealESRGAN(device, scale=UPSCALE_FACTOR)
model.load_weights("weights/RealESRGAN_x4plus.pth")

img_pil = Image.open(IMAGE_PATH).convert("RGB")
esrgan_upscaled = model.predict(img_pil)

esrgan_upscaled.save(
    os.path.join(OUTPUT_DIR, "upscale_realesrgan.png")
)

# =========================
# 5. 비교용 나란히 저장
# =========================
opencv_rgb = cv2.cvtColor(opencv_upscaled, cv2.COLOR_BGR2RGB)
opencv_pil = Image.fromarray(opencv_rgb)

combined = Image.new(
    "RGB",
    (opencv_pil.width + esrgan_upscaled.width, opencv_pil.height)
)

combined.paste(opencv_pil, (0, 0))
combined.paste(esrgan_upscaled, (opencv_pil.width, 0))

combined.save(
    os.path.join(OUTPUT_DIR, "compare_opencv_vs_esrgan.png")
)

print("✅ 업스케일 비교 완료")
print(" - OpenCV:", "upscale_opencv_lanczos.png")
print(" - Real-ESRGAN:", "upscale_realesrgan.png")
print(" - 비교:", "compare_opencv_vs_esrgan.png")
