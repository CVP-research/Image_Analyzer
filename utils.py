"""
Utility functions for file operations and caching
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import List


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
