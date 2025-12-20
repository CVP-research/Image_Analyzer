import os
import cv2
import numpy as np
import random
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from ultralytics import SAM
from io import BytesIO
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import json 
import uvicorn
from tqdm import tqdm

# ==========================================
# 1. 설정 및 전역 변수
# ==========================================

# 🚨 사용자 설정 필요 🚨
INPUT_VIDEO_PATH = Path("/home/rocknroll1397/Image_Analyzer/new/data/77_7859_15670/test2.mp4")  # 처리할 비디오 파일 경로
MASKED_FRAMES_DIR = Path("../output/veo3/masked_frames")

os.makedirs(MASKED_FRAMES_DIR, exist_ok=True)

# 메모리 캐시
MASK_CACHE: Dict[int, Dict[str, Any]] = {}
FRAME_ID_COUNTER = 0
SAM_MODEL = None


# ==========================================
# 2. 핵심 유틸리티 함수
# ==========================================

def get_sam_model():
    """SAM 모델을 로드하고 반환."""
    global SAM_MODEL
    if SAM_MODEL is None:
        try:
            print("Loading SAM model...")
            SAM_MODEL = SAM("../sam2_l.pt") 
            print("SAM model loaded.")
        except Exception as e:
            print(f"❌ FATAL ERROR: SAM 모델 로드 실패. 경로 확인: {e}")
            return None
    return SAM_MODEL

def get_sam_masks(image_np: np.ndarray) -> List[np.ndarray]:
    """SAM 모델을 실행하고 마스크 리스트 반환"""
    model = get_sam_model()
    if model is None: return []
    # verbose=False를 추가하여 predict 로그 숨김
    pred_results = model.predict(image_np, task="segment", verbose=False)
    if not pred_results or pred_results[0].masks is None: return []
    return [m.cpu().numpy().astype(np.uint8) for m in pred_results[0].masks.data]

def get_mask_contour_data(mask_np: np.ndarray) -> List[List[int]]:
    """마스크를 외곽선 좌표로 변환"""
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    # 가장 큰 외곽선만 반환
    return max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()

def extract_clean_object(img: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """마스크를 사용해 배경 없는 BGRA 이미지 생성 및 크롭"""
    alpha_channel = (binary_mask * 255).astype(np.uint8)
    b, g, r = cv2.split(img)
    bgra_image = cv2.merge([b, g, r, alpha_channel])
    
    ys, xs = np.where(alpha_channel > 0)
    if len(xs) == 0 or len(ys) == 0: return bgra_image
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return bgra_image[y_min:y_max, x_min:x_max]


# ==========================================
# 3. 사전 분할 및 캐싱 (비디오용)
# ==========================================

def pre_segment_video_frames(frame_limit: Optional[int] = None, is_random: bool = False):
    """시작 시 비디오의 프레임을 분할하고 캐싱합니다. is_random 플래그에 따라 순차 또는 랜덤 샘플링을 수행합니다."""
    global FRAME_ID_COUNTER
    print("\n--- 🚀 비디오 프레임 분할 작업 시작 ---")
    if get_sam_model() is None: return

    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    if not cap.isOpened():
        print(f"❌ 오류: 비디오를 열 수 없습니다. 경로 확인: {INPUT_VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ 비디오 정보: {total_frames} frames, {cap.get(cv2.CAP_PROP_FPS):.2f} FPS")
    
    frame_indices_to_process = []
    if is_random:
        print(f"🎲 랜덤 모드: {frame_limit}개의 프레임을 무작위로 선택합니다.")
        limit = min(total_frames, frame_limit if frame_limit is not None else total_frames)
        frame_indices_to_process = sorted(random.sample(range(total_frames), limit))
    else:
        print(f"sequential 모드: 처음부터 최대 {frame_limit}개의 프레임을 순차적으로 처리합니다.")
        limit = total_frames if frame_limit is None else min(total_frames, frame_limit)
        frame_indices_to_process = list(range(limit))

    start_time = time.time()
    
    with tqdm(total=len(frame_indices_to_process), desc="🎥 비디오 프레임 처리 중") as pbar:
        for frame_index in frame_indices_to_process:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                pbar.update(1)
                continue
            
            masks_list = get_sam_masks(frame)
            if not masks_list:
                pbar.update(1)
                continue
                
            mask_data = [{"index": i, "points": get_mask_contour_data(m)} for i, m in enumerate(masks_list)]
            frame_name = f"{INPUT_VIDEO_PATH.stem}_frame_{frame_index:05d}"
            
            # FRAME_ID_COUNTER를 사용하여 고유 ID 보장
            MASK_CACHE[FRAME_ID_COUNTER] = {
                "image_np": frame, 
                "masks": masks_list,
                "mask_data": mask_data, 
                "frame_name": frame_name
            }
            FRAME_ID_COUNTER += 1
            pbar.update(1)

    cap.release()
    print(f"--- ✅ 분할 완료! 총 {FRAME_ID_COUNTER}개 프레임 캐싱 완료 ({time.time() - start_time:.2f}초) ---")


# ==========================================
# 4. FastAPI 애플리케이션 정의
# ==========================================

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행: 폴더 생성 및 사전 분할"""
    print("--- FastAPI App 시작, 사전 분할 실행 ---")
    MASKED_FRAMES_DIR.mkdir(exist_ok=True)
    pre_segment_video_frames(frame_limit=5, is_random=True)  # 최대 5프레임만 랜덤으로 사전 분할
    if FRAME_ID_COUNTER == 0:
        print("\n--- ⚠️ 경고: 캐시된 프레임이 없습니다. ---")

@app.get("/", response_class=HTMLResponse)
async def root_redirect(request: Request):
    """첫 페이지. 사용 가능한 첫 프레임으로 리디렉션"""
    if not MASK_CACHE:
        return templates.TemplateResponse("error.html", {"request": request, "message": "사전 분할된 데이터가 없습니다. 스크립트 실행 로그를 확인하세요."}, status_code=503)
    
    first_frame_id = min(MASK_CACHE.keys())
    return RedirectResponse(url=f"/select/{first_frame_id}", status_code=302)

@app.get("/select/{frame_id}", response_class=HTMLResponse)
async def select_object_page(request: Request, frame_id: int):
    """지정된 프레임의 분할된 객체를 보여주는 선택 페이지"""
    if frame_id not in MASK_CACHE:
        return templates.TemplateResponse("error.html", {"request": request, "message": f"프레임 ID {frame_id}를 찾을 수 없습니다."}, status_code=404)
    
    cache = MASK_CACHE[frame_id]
    
    frame_ids = sorted(MASK_CACHE.keys())
    current_index = frame_ids.index(frame_id)
    prev_id = frame_ids[current_index - 1] if current_index > 0 else None
    next_id = frame_ids[current_index + 1] if current_index < len(frame_ids) - 1 else None

    return templates.TemplateResponse("select.html", {
        "request": request,
        "frame_id": frame_id,
        "image_url": f"/image/{frame_id}",
        "mask_data_json": json.dumps(cache["mask_data"]),
        "frame_name": cache["frame_name"],
        "total_frames": len(MASK_CACHE),
        "current_frame_num": current_index + 1,
        "prev_id": prev_id,
        "next_id": next_id
    })

@app.get("/image/{frame_id}")
async def get_image(frame_id: int):
    """원본 프레임 이미지를 JPEG로 스트리밍"""
    if frame_id not in MASK_CACHE: return JSONResponse({"error": "Frame not found"}, 404)
    image_np = MASK_CACHE[frame_id]['image_np']
    _, buffer = cv2.imencode('.jpg', image_np)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.post("/save/{frame_id}/{mask_index}")
async def save_object(frame_id: int, mask_index: int):
    """선택된 객체를 서버의 'masked_frames' 폴더에 PNG로 저장"""
    if frame_id not in MASK_CACHE:
        return JSONResponse({"status": "error", "message": "Frame ID not found"}, status_code=404)
    
    cache = MASK_CACHE[frame_id]
    if mask_index >= len(cache['masks']):
        return JSONResponse({"status": "error", "message": "Mask index out of range"}, status_code=404)
    
    original_image = cache['image_np']
    selected_mask = cache['masks'][mask_index]
    
    output_filename = f"{cache['frame_name']}_mask_{mask_index}.png"
    output_path = MASKED_FRAMES_DIR / output_filename
    
    try:
        bgra_cropped = extract_clean_object(original_image, selected_mask)
        cv2.imwrite(str(output_path), bgra_cropped)
        print(f"✅ Mask saved to: {output_path}")
        return JSONResponse({"status": "success", "path": str(output_path)})
    except Exception as e:
        print(f"❌ Failed to save mask: {e}")
        return JSONResponse({"status": "error", "message": "Failed to save file."}, status_code=500)

# ==========================================
# 5. 스크립트 메인 실행부
# ==========================================
if __name__ == "__main__":
    # uvicorn new.segment_server_video:app --reload
    print("--- Uvicorn 서버 직접 실행 ---")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
