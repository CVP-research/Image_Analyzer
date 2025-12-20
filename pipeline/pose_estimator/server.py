"""
Pose Estimator Server

카메라 포즈 추정 및 관리를 위한 FastAPI 서버
- input 폴더의 이미지 관리
- 카메라 포즈 저장/로드
- 웹 UI 제공
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import json
import shutil
from pathlib import Path

app = FastAPI(title="Pose Estimator", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리 설정
BASE_DIR = Path(__file__).parent                    # pose_estimator/
ROOT_DIR = BASE_DIR.parent                          # Image_Analyzer/
INPUT_DIR = ROOT_DIR / "input"                      # 공통 input 폴더
POSE_DIR = INPUT_DIR / "pose"                       # 포즈 데이터 폴더
POSES_FILE = POSE_DIR / "camera_poses.json"         # 포즈 파일

INPUT_DIR.mkdir(exist_ok=True)
POSE_DIR.mkdir(exist_ok=True)

# Pydantic 모델
class CameraPose(BaseModel):
    """카메라 포즈 데이터 모델"""
    position: List[float]
    look_at: List[float]
    up: List[float]

class PoseData(BaseModel):
    """전체 포즈 데이터"""
    poses: Dict[str, CameraPose]


# ============================================================
# 웹 UI
# ============================================================

@app.get("/")
async def root():
    """메인 페이지 (index.html)"""
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        content={"status": "ok", "message": "Pose Estimator API Server"},
        status_code=200
    )


@app.get("/app.js")
async def get_app_js():
    """JavaScript 파일 제공"""
    js_file = BASE_DIR / "app.js"
    if js_file.exists():
        return FileResponse(str(js_file), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")

# ============================================================
# API: 이미지 관리
# ============================================================

@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    이미지 업로드 (input 폴더에 저장)
    
    Returns:
        업로드된 파일 목록 및 개수
    """
    uploaded_files = []

    for file in files:
        if not file.filename:
            continue

        file_path = INPUT_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_files.append(file.filename)

    return {"uploaded": uploaded_files, "count": len(uploaded_files)}


@app.get("/api/images")
async def get_images():
    """
    input 폴더의 이미지 목록 반환
    
    Returns:
        이미지 파일 목록 (png, jpg, jpeg만)
    """
    if not INPUT_DIR.exists():
        return {"images": []}

    images = [
        f.name for f in INPUT_DIR.iterdir() 
        if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ]
    return {"images": sorted(images)}


@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """
    이미지 파일 반환
    
    Args:
        filename: 이미지 파일명
    """
    file_path = INPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

# ============================================================
# API: 카메라 포즈 관리
# ============================================================

@app.post("/api/poses/save")
async def save_poses(data: PoseData):
    """
    카메라 포즈 저장
    
    Args:
        data: 포즈 데이터 (filename -> pose)
    """
    poses_dict = {
        filename: pose.dict()
        for filename, pose in data.poses.items()
    }

    with open(POSES_FILE, "w", encoding="utf-8") as f:
        json.dump(poses_dict, f, indent=2, ensure_ascii=False)

    return {"status": "success", "saved": len(poses_dict)}


@app.get("/api/poses/load")
async def load_poses():
    """
    저장된 카메라 포즈 로드
    
    Returns:
        포즈 데이터 딕셔너리
    """
    if not POSES_FILE.exists():
        return {"poses": {}}

    with open(POSES_FILE, "r", encoding="utf-8") as f:
        poses = json.load(f)

    return {"poses": poses}


@app.get("/api/poses/download")
async def download_poses():
    """
    포즈 JSON 파일 다운로드
    """
    if not POSES_FILE.exists():
        raise HTTPException(status_code=404, detail="No poses saved")
    return FileResponse(
        POSES_FILE,
        filename="camera_poses.json",
        media_type="application/json"
    )

# ============================================================
# API: 유틸리티
# ============================================================

@app.delete("/api/reset")
async def reset_all():
    """
    포즈 데이터 초기화 (이미지는 유지)
    
    Returns:
        초기화 상태
    """
    if POSES_FILE.exists():
        POSES_FILE.unlink()

    return {"status": "reset complete", "note": "images preserved"}


# ============================================================
# 서버 실행
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Pose Estimator Server")
    print("=" * 60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Pose directory: {POSE_DIR}")
    print(f"Poses file: {POSES_FILE}")
    print(f"Server: http://0.0.0.0:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
