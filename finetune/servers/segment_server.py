import os
import cv2
import numpy as np
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

# ==========================================
# 1. 설정 및 전역 변수
# ==========================================

# 🚨 사용자 설정 필요 🚨
INPUT_FRAMES_DIR = Path("./data/77_7859_15670/images") 
MASKED_FRAMES_DIR = Path("./output/masked_frames")

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
    pred_results = model.predict(image_np, task="segment", verbose=False)
    if not pred_results or pred_results[0].masks is None: return []
    return [m.cpu().numpy().astype(np.uint8) for m in pred_results[0].masks.data]

def get_mask_contour_data(mask_np: np.ndarray) -> List[List[int]]:
    """마스크를 외곽선 좌표로 변환"""
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
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
# 3. 사전 분할 및 캐싱
# ==========================================

def pre_segment_all_frames():
    """시작 시 모든 프레임을 분할하고 캐싱"""
    global FRAME_ID_COUNTER
    print("\n--- 🚀 모든 프레임 사전 분할 작업 시작 ---")
    if get_sam_model() is None: return

    frame_paths = sorted(INPUT_FRAMES_DIR.glob("*.jpg"))
    if not frame_paths:
        print(f"❌ 오류: {INPUT_FRAMES_DIR}에서 프레임을 찾을 수 없습니다.")
        return

    print(f"✅ 총 {len(frame_paths)}개의 프레임을 순차적으로 분할합니다.")
    start_time = time.time()
    
    for frame_path in frame_paths:
        image_np = cv2.imread(str(frame_path))
        if image_np is None: continue
        
        masks_list = get_sam_masks(image_np)
        if not masks_list: continue
            
        mask_data = [{"index": i, "points": get_mask_contour_data(m)} for i, m in enumerate(masks_list)]

        MASK_CACHE[FRAME_ID_COUNTER] = {
            "image_np": image_np, "masks": masks_list,
            "mask_data": mask_data, "frame_name": frame_path.stem
        }
        FRAME_ID_COUNTER += 1
        
    print(f"--- ✅ 사전 분할 완료! 총 {FRAME_ID_COUNTER}개 프레임 캐싱 ({time.time() - start_time:.2f}초) ---")


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
    pre_segment_all_frames()
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
    
    # 순차 탐색을 위한 ID 계산
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
        "total_frames": FRAME_ID_COUNTER,
        "current_frame_num": current_index + 1,
        "prev_id": prev_id,
        "next_id": next_id
    })

@app.get("/image/{frame_id}")
async def get_image(frame_id: int):
    """원본 프레임 이미지를 JPEG로 스트리밍"""
    if frame_id not in MASK_CACHE: return JSONResponse({"error": "Frame ID not found"}, 404)
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
    
    # 파일명은 원본 이름과 마스크 인덱스를 조합
    output_filename = f"{cache['frame_name']}.png"
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
    # `uvicorn new.segment_server:app --reload` 명령어로 실행하는 것을 권장
    print("--- Uvicorn 서버 직접 실행 ---")
    uvicorn.run("segment_server:app", host="0.0.0.0", port=8000, log_level="info", reload=True)