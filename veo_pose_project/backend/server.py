"""
FastAPI 백엔드 서버
- 이미지 업로드 및 관리
- 카메라 포즈 저장/로드 API
- 정적 파일(프런트엔드) 서빙
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import json
import os
import shutil
from pathlib import Path

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 필요시 특정 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 디렉토리 설정
BASE_DIR = Path(__file__).parent              # backend/
PROJECT_DIR = BASE_DIR.parent                 # veo_pose_project/
UPLOAD_DIR = BASE_DIR / "uploads"
POSES_FILE = BASE_DIR / "camera_poses.json"
FRONTEND_DIR = PROJECT_DIR / "frontend"       # index.html, app.js 위치

UPLOAD_DIR.mkdir(exist_ok=True)

# Pydantic 모델
class CameraPose(BaseModel):
    position: List[float]
    look_at: List[float]
    up: List[float]

class PoseData(BaseModel):
    poses: Dict[str, CameraPose]


# -----------------------------
# 정적 파일 & 프런트엔드 서빙
# -----------------------------

# /static/*  →  frontend/ 안의 파일들 서빙 (app.js 포함)
if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static"
    )
else:
    print(f"[WARN] FRONTEND_DIR not found: {FRONTEND_DIR}")


@app.get("/")
async def root():
    """
    메인 페이지: frontend/index.html 반환
    """
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    # index.html 없을 때는 단순 JSON 응답
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "message": "frontend/index.html not found. API server is running."
        },
    )


# -----------------------------
# API: 이미지 업로드 / 관리
# -----------------------------

@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """이미지 업로드"""
    uploaded_files = []

    # 기존 업로드 디렉토리 정리
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(exist_ok=True)

    for file in files:
        if not file.filename:
            continue

        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_files.append(file.filename)

    return {"uploaded": uploaded_files, "count": len(uploaded_files)}


@app.get("/api/images")
async def get_images():
    """업로드된 이미지 목록"""
    if not UPLOAD_DIR.exists():
        return {"images": []}

    images = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"images": sorted(images)}


@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """이미지 파일 반환"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


# -----------------------------
# API: 카메라 포즈 저장 / 로드
# -----------------------------

@app.post("/api/poses/save")
async def save_poses(data: PoseData):
    """카메라 포즈 저장"""
    poses_dict = {
        filename: pose.dict()
        for filename, pose in data.poses.items()
    }

    with open(POSES_FILE, "w", encoding="utf-8") as f:
        json.dump(poses_dict, f, indent=2, ensure_ascii=False)

    return {"status": "success", "saved": len(poses_dict)}


@app.get("/api/poses/load")
async def load_poses():
    """저장된 카메라 포즈 로드"""
    if not POSES_FILE.exists():
        return {"poses": {}}

    with open(POSES_FILE, "r", encoding="utf-8") as f:
        poses = json.load(f)

    return {"poses": poses}


@app.get("/api/poses/download")
async def download_poses():
    """포즈 JSON 파일 다운로드"""
    if not POSES_FILE.exists():
        raise HTTPException(status_code=404, detail="No poses saved")
    return FileResponse(
        POSES_FILE,
        filename="camera_poses.json",
        media_type="application/json"
    )


# -----------------------------
# API: 전체 리셋
# -----------------------------

@app.delete("/api/reset")
async def reset_all():
    """모든 데이터 초기화"""
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(exist_ok=True)

    if POSES_FILE.exists():
        POSES_FILE.unlink()

    return {"status": "reset complete"}


# -----------------------------
# 개발용 실행
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
