"""
Utility functions for file operations and caching
"""

import os
import hashlib
import pickle
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple
import cv2
import json
import math


def compute_dataset_hash(directory: Path) -> str:
    """dataset 디렉토리의 해시 계산 (파일 목록 + 수정 시간)"""
    hash_md5 = hashlib.md5()
    
    for root, dirs, files in sorted(os.walk(directory)):
        for file in sorted(files):
            filepath = Path(root) / file
            # 파일 경로와 수정 시간을 해시에 포함
            hash_md5.update(str(filepath).encode())
            hash_md5.update(str(filepath.stat().st_mtime).encode())
    
    return hash_md5.hexdigest()


def find_all_images(directory: Path, use_cache: bool = True, cache_dir: Path = None) -> List[Path]:
    """
    디렉토리에서 재귀적으로 모든 이미지 파일 찾기 (캐싱)
    
    Args:
        directory: 스캔할 디렉토리
        use_cache: 캐시 사용 여부
        cache_dir: 캐시 저장 디렉토리
    
    Returns:
        이미지 파일 경로 리스트
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent / ".dataset_cache"
        cache_dir.mkdir(exist_ok=True)
    
    image_list_cache = cache_dir / "image_list.pkl"
    dataset_hash_file = cache_dir / "dataset_hash.txt"
    
    if use_cache:
        # 현재 dataset 해시 계산
        current_hash = compute_dataset_hash(directory)
        
        # 이전 해시 로드
        prev_hash = None
        if dataset_hash_file.exists():
            with open(dataset_hash_file, 'r') as f:
                prev_hash = f.read().strip()
        
        # 해시가 같고 캐시 파일이 있으면 캐시 로드
        if current_hash == prev_hash and image_list_cache.exists():
            print(f"[Cache] Loading image list from cache...")
            with open(image_list_cache, 'rb') as f:
                images = pickle.load(f)
            print(f"[Cache] Loaded {len(images)} images from cache")
            return images
        
        print(f"[Cache] Dataset changed or no cache found. Scanning directory...")
    
    # 캐시가 없거나 dataset이 변경된 경우 스캔
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                images.append(Path(root) / file)
    
    images = sorted(images)
    
    if use_cache:
        # 캐시 저장
        with open(image_list_cache, 'wb') as f:
            pickle.dump(images, f)
        
        # 해시 저장
        with open(dataset_hash_file, 'w') as f:
            f.write(current_hash)
        
        print(f"[Cache] Saved {len(images)} images to cache")
    
    return images


def get_image_cache_key(image_path: Path) -> str:
    """이미지 파일의 캐시 키 생성 (경로 + 수정시간 + 크기)"""
    stat = image_path.stat()
    key_str = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cache(cache_file: Path) -> dict:
    """캐시 파일 로드"""
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    return {}


def save_cache(cache_file: Path, data: dict):
    """캐시 파일 저장"""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

def get_all_objects(original_input_dir: Path, masked_frames_dir: Path) -> List[Tuple[Image.Image, Dict]]:
    """
    Loads all segmented object images from MASKED_FRAMES_DIR and original input
    images from INPUT_DIR, returning them with metadata.

    Returns:
        A list of tuples, where each tuple contains an object image (PIL RGBA)
        and its metadata.
    """
    objects = []
    
    # Load from MASKED_FRAMES_DIR (segmented video frames)
    print(f"\n[Step 3c] Loading segmented frames from {masked_frames_dir}...")
    masked_files = sorted([p for p in masked_frames_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    for idx, file_path in enumerate(masked_files):
        try:
            img = Image.open(file_path).convert("RGBA")
            metadata = {
                "frame_idx": idx,
                "category": "object", # Assuming all are objects
                "filename": file_path.name
            }
            objects.append((img, metadata))
            print(f"  Loaded masked frame: {file_path.name}")
        except Exception as e:
            print(f"  [WARN] Could not load masked frame {file_path.name}: {e}")

    # Load from INPUT_DIR (original input images, assumed to be pre-masked or objects)
    print(f"\n[Step 3d] Loading original input images from {original_input_dir}...")
    original_files = sorted([p for p in original_input_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    start_idx = len(objects) # Continue indexing from where masked_files left off

    for i, file_path in enumerate(original_files):
        try:
            img = Image.open(file_path).convert("RGBA")
            metadata = {
                "frame_idx": start_idx + i,
                "category": "object", # Assuming all are objects
                "filename": file_path.name
            }
            objects.append((img, metadata))
            print(f"  Loaded original input image: {file_path.name}")
        except Exception as e:
            print(f"  [WARN] Could not load original input image {file_path.name}: {e}")
            
    if not objects:
        print("  Warning: No objects were loaded from any directory.")
        
    return objects

def load_camera_poses(path: Path) -> Dict[str, dict]:
    """
    backend/camera_poses.json 로드
    
    Args:
        path: camera_poses.json 파일 경로
    
    Returns:
        카메라 포즈 딕셔너리 {filename: {"position": [...], "look_at": [...]}}
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_center(poses: Dict[str, dict]) -> List[float]:
    """
    모든 look_at의 평균을 '객체 중심'으로 사용.
    (대부분 [0,0,0]일 가능성이 높지만, 일반화해서 계산)
    
    Args:
        poses: 카메라 포즈 딕셔너리
    
    Returns:
        [x, y, z] 객체 중심 좌표
    """
    xs, ys, zs = [], [], []
    for p in poses.values():
        lx, ly, lz = p["look_at"]
        xs.append(lx)
        ys.append(ly)
        zs.append(lz)

    n = len(xs) if xs else 1
    return [sum(xs)/n, sum(ys)/n, sum(zs)/n]

def compute_azimuth_elevation(
    position: List[float],
    center: List[float]
) -> tuple[float, float, float]:
    """
    카메라 위치와 객체 중심으로부터 azimuth, elevation, distance 계산
    
    Args:
        position: 카메라 위치 [x, y, z]
        center: 객체 중심 [x, y, z]
    
    Returns:
        (azimuth, elevation, distance) 튜플
        - azimuth: 수평각 (-180 ~ 180도, z축 기준)
        - elevation: 수직각 (-90 ~ 90도, y축 기준)
        - distance: 카메라와 객체 사이 거리
    """
    cx, cy, cz = center
    px, py, pz = position
    
    # 객체 중심 기준 위치 벡터
    vx, vy, vz = px - cx, py - cy, pz - cz
    r = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-8
    
    # 수평각(azimuth), 수직각(elevation)
    azimuth = math.degrees(math.atan2(vx, vz))      # x-z 평면 기준
    elevation = math.degrees(math.asin(vy / r))     # y 기준
    
    return azimuth, elevation, r

def frame_from_video(video_path: Path, output_dir: Path) -> None:
    """
    Veo3로 생성된 360도 영상에서 유일한 프레임만 분리 및 저장
    
    Args:
        video_path: Veo3에서 생성된 360도 영상 파일 경로
        output_dir: 분리된 프레임 저장 디렉토리 경로
    """
    print(f"\n[Frame Extraction] Extracting unique frames from video: {video_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    saved_idx = 0
    seen_hashes = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 해시로 변환
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        if frame_hash in seen_hashes:
            frame_idx += 1
            continue  # 이미 저장된 프레임이면 건너뜀

        seen_hashes.add(frame_hash)
        frame_filename = output_dir / f"frame_{saved_idx:04d}.png"
        cv2.imwrite(str(frame_filename), frame)
        print(f"  Saved unique frame: {frame_filename.name}")
        frame_idx += 1
        saved_idx += 1

    cap.release()
    print(f"✓ Extracted {saved_idx} unique frames to {output_dir}")

def _fixed_grid_selection(filtered: List[dict], ideal_positions: List[Tuple[float, float]]) -> List[dict]:
    """
    Fixed Grid: 이상적 위치(0°, 120°, -120°)에 가장 가까운 뷰 선택
    """
    chosen = []
    remaining = filtered.copy()
    
    for ideal_az, ideal_el in ideal_positions:
        if not remaining:
            break
        
        # 유클리드 거리로 가장 가까운 뷰 찾기
        best = min(remaining, key=lambda v: 
            math.sqrt((v["azimuth"] - ideal_az)**2 + (v["elevation"] - ideal_el)**2)
        )
        chosen.append(best)
        remaining.remove(best)
    
    return sorted(chosen, key=lambda v: v["azimuth"])


def _robust_greedy_selection(filtered: List[dict]) -> List[dict]:
    """
    Robust Greedy: 3D 구면 거리 기반 최대 분산 선택
    """
    def vec_distance(v1, v2):
        """3D 구면 거리 (각도)"""
        dot = sum(a * b for a, b in zip(v1["vec"], v2["vec"]))
        return math.degrees(math.acos(max(-1, min(1, dot))))
    
    # 1) 정면 선택: azimuth 0 + elevation 0에 가장 가까운 것
    first = min(filtered, key=lambda v: abs(v["azimuth"]) * 1.0 + abs(v["elevation"]) * 3.0)
    chosen = [first]
    remaining = [v for v in filtered if v is not first]
    
    if len(remaining) < 2:
        return chosen + remaining
    
    # 2) 두 번째: 첫 번째와 3D 거리 최대화
    second = max(remaining, key=lambda v: vec_distance(first, v) - abs(v["elevation"]) * 2)
    chosen.append(second)
    remaining = [v for v in remaining if v is not second]
    
    if len(remaining) < 1:
        return chosen + remaining
    
    # 3) 세 번째: 기존 2개와의 최소 거리를 최대화 (Max-Min)
    third = max(remaining, key=lambda v: 
        min(vec_distance(first, v), vec_distance(second, v)) - abs(v["elevation"]) * 2
    )
    chosen.append(third)
    
    return sorted(chosen, key=lambda v: v["azimuth"])