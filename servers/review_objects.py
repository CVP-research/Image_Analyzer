"""
객체 검수 서버

Veo3 영상에서 추출한 객체들을 사용자가 검수하고,
잘못된 객체는 삭제하고 좋은 객체만 남기는 웹 UI 서버

- 승인 (A / ←): 그대로 유지
- 버리기 (D / →): 파일 삭제
"""

import os
import signal
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ==========================================
# 1. 설정 및 전역 변수
# ==========================================
# 환경 변수로 경로 받기
OBJECTS_DIR = Path(os.environ.get("REVIEW_OBJECTS_DIR", "./output/masked_frames"))

# 상태 관리
object_files = []
current_index = 0
approved_count = 0
discarded_count = 0


# ==========================================
# 2. FastAPI 애플리케이션
# ==========================================

app = FastAPI(title="객체 검수 서버")


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 객체 파일 목록 로드"""
    global object_files
    
    print("\n" + "=" * 60)
    print("🔍 객체 검수 서버 시작")
    print("=" * 60)
    print(f"  객체 폴더: {OBJECTS_DIR}")
    print(f"  모드: 버리는 이미지만 삭제")
    
    # PNG 파일 목록
    if OBJECTS_DIR.exists():
        object_files = sorted(list(OBJECTS_DIR.glob("*.png")))
    
    print(f"  총 {len(object_files)}개 객체 발견")
    print("=" * 60)
    
    if not object_files:
        print("⚠️ 검수할 객체가 없습니다!")


@app.get("/", response_class=HTMLResponse)
async def index():
    """메인 페이지"""
    if not object_files:
        return HTMLResponse("""
        <html>
        <head><title>객체 검수</title></head>
        <body style="font-family: sans-serif; text-align: center; margin-top: 20vh;">
            <h1>⚠️ 검수할 객체가 없습니다</h1>
            <p>객체 폴더에 PNG 파일이 없습니다.</p>
        </body>
        </html>
        """)
    return RedirectResponse(url="/review/0")


@app.get("/review/{idx}", response_class=HTMLResponse)
async def review_page(idx: int):
    """검수 페이지"""
    global current_index
    current_index = idx
    
    if idx >= len(object_files):
        # 모든 검수 완료
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>검수 완료</title>
            <style>
                body {{ font-family: sans-serif; text-align: center; margin-top: 10vh; background: #1a1a1a; color: #fff; }}
                .stats {{ font-size: 1.5em; margin: 20px 0; }}
                .complete-btn {{ padding: 20px 40px; font-size: 1.5em; background: #28a745; color: white; 
                                border: none; border-radius: 10px; cursor: pointer; margin-top: 30px; }}
                .complete-btn:hover {{ background: #218838; }}
            </style>
        </head>
        <body>
            <h1>✅ 모든 객체 검수 완료!</h1>
            <div class="stats">
                <p>✅ 승인: {approved_count}개</p>
                <p>❌ 버림: {discarded_count}개</p>
            </div>
            <button class="complete-btn" onclick="completeReview()">🚀 검수 완료 & 데이터셋 생성</button>
            
            <script>
                async function completeReview() {{
                    try {{
                        await fetch('/api/complete', {{ method: 'POST' }});
                    }} catch(e) {{}}
                    document.body.innerHTML = '<h1 style="margin-top: 20vh;">✅ 서버 종료됨. 데이터셋 생성 중...</h1>';
                }}
            </script>
        </body>
        </html>
        """)
    
    obj_file = object_files[idx]
    total = len(object_files)
    progress = ((idx + 1) / total) * 100
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>객체 검수 ({idx + 1}/{total})</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ 
                font-family: sans-serif; 
                background: #1a1a1a; 
                color: #fff; 
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }}
            .header {{
                padding: 15px 20px;
                background: #2a2a2a;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .progress-bar {{
                width: 300px;
                height: 10px;
                background: #444;
                border-radius: 5px;
                overflow: hidden;
            }}
            .progress-fill {{
                height: 100%;
                background: #4CAF50;
                width: {progress}%;
                transition: width 0.3s;
            }}
            .stats {{
                display: flex;
                gap: 20px;
            }}
            .stat {{ font-size: 0.9em; }}
            .stat.approved {{ color: #4CAF50; }}
            .stat.discarded {{ color: #f44336; }}
            
            .main-content {{
                flex: 1;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }}
            .object-container {{
                background: repeating-conic-gradient(#3a3a3a 0% 25%, #2a2a2a 0% 50%) 50% / 20px 20px;
                border-radius: 10px;
                padding: 20px;
                max-width: 80vw;
                max-height: 70vh;
            }}
            .object-container img {{
                max-width: 100%;
                max-height: 60vh;
                display: block;
            }}
            
            .controls {{
                padding: 20px;
                background: #2a2a2a;
                display: flex;
                justify-content: center;
                gap: 30px;
            }}
            .btn {{
                padding: 15px 40px;
                font-size: 1.2em;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: transform 0.1s, background 0.2s;
            }}
            .btn:hover {{ transform: scale(1.05); }}
            .btn:active {{ transform: scale(0.95); }}
            .btn-approve {{ background: #4CAF50; color: white; }}
            .btn-approve:hover {{ background: #45a049; }}
            .btn-discard {{ background: #f44336; color: white; }}
            .btn-discard:hover {{ background: #da190b; }}
            .btn-complete {{ background: #FF9800; color: white; }}
            .btn-complete:hover {{ background: #e68a00; }}
            
            .filename {{
                text-align: center;
                padding: 10px;
                color: #888;
                font-size: 0.9em;
            }}
            
            .keyboard-hint {{
                text-align: center;
                padding: 10px;
                color: #666;
                font-size: 0.8em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <strong>객체 검수</strong> - {idx + 1} / {total}
            </div>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
            <div class="stats">
                <span class="stat approved">✅ {approved_count}</span>
                <span class="stat discarded">❌ {discarded_count}</span>
            </div>
        </div>
        
        <div class="filename">{obj_file.name}</div>
        
        <div class="main-content">
            <div class="object-container">
                <img src="/image/{idx}" alt="객체 이미지">
            </div>
        </div>
        
        <div class="keyboard-hint">
            키보드: → 다음 (유지) | D 버리기 (삭제) | Enter 완료
        </div>
        
        <div class="controls">
            <button class="btn btn-discard" onclick="discard()">❌ 버리기 (D)</button>
            <button class="btn btn-approve" onclick="next()">➡️ 다음 (유지)</button>
            <button class="btn btn-complete" onclick="completeNow()">🚀 여기까지만 완료</button>
        </div>
        
        <script>
            function next() {{
                // 그냥 다음으로 넘어감 (파일 유지)
                fetch('/api/approve/{idx}', {{ method: 'POST' }})
                    .then(() => window.location.href = '/review/{idx + 1}');
            }}
            
            function discard() {{
                fetch('/api/discard/{idx}', {{ method: 'POST' }})
                    .then(() => window.location.href = '/review/{idx + 1}');
            }}
            
            async function completeNow() {{
                if (confirm('검수를 종료하고 데이터셋 생성을 시작하시겠습니까?')) {{
                    try {{
                        await fetch('/api/complete', {{ method: 'POST' }});
                    }} catch(e) {{}}
                    document.body.innerHTML = '<h1 style="text-align:center; margin-top:20vh;">✅ 서버 종료됨. 데이터셋 생성 중...</h1>';
                }}
            }}
            
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'ArrowRight' || e.key === ' ') next();
                else if (e.key === 'd' || e.key === 'D' || e.key === 'Backspace' || e.key === 'Delete') discard();
                else if (e.key === 'Enter') completeNow();
            }});
        </script>
    </body>
    </html>
    """)


@app.get("/image/{idx}")
async def get_image(idx: int):
    """이미지 파일 제공"""
    if 0 <= idx < len(object_files):
        return FileResponse(object_files[idx])
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.post("/api/approve/{idx}")
async def approve_object(idx: int):
    """객체 승인 - 그냥 두기 (삭제 안 함)"""
    global approved_count
    
    if 0 <= idx < len(object_files):
        approved_count += 1
        return JSONResponse({"status": "approved", "file": object_files[idx].name})
    
    return JSONResponse({"error": "Invalid index"}, status_code=400)


@app.post("/api/discard/{idx}")
async def discard_object(idx: int):
    """객체 버리기 - 파일 삭제"""
    global discarded_count
    
    if 0 <= idx < len(object_files):
        file_to_delete = object_files[idx]
        try:
            file_to_delete.unlink()  # 파일 삭제
            discarded_count += 1
            return JSONResponse({"status": "discarded", "file": file_to_delete.name, "deleted": True})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    
    return JSONResponse({"error": "Invalid index"}, status_code=400)


@app.post("/api/complete")
async def complete_and_shutdown():
    """검수 완료 및 서버 종료"""
    remaining = len(object_files) - discarded_count
    print("\n" + "=" * 60)
    print("🏁 검수 완료!")
    print(f"  ✅ 승인 (유지): {approved_count}개")
    print(f"  ❌ 버림 (삭제): {discarded_count}개")
    print(f"  📂 남은 객체: {OBJECTS_DIR}")
    print("=" * 60)
    
    # 서버 종료
    os.kill(os.getpid(), signal.SIGTERM)
    return JSONResponse({"status": "success", "approved": approved_count, "discarded": discarded_count})


# ==========================================
# 메인 실행
# ==========================================

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 포트 받기
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8003
    
    print(f"\n🔍 객체 검수 서버 시작 - http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
