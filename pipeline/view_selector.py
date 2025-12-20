import json
import math
import random
from pathlib import Path
from typing import List, Dict
from typing import Tuple

# 전역 설정
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "train"
INPUT_DIR = Path("/home/rocknroll1397/Image_Analyzer/new/data/77_7859_15670/images")
MASKED_FRAMES_DIR = Path("/home/rocknroll1397/Image_Analyzer/new/output/masked_frames")
POSE_DIR = INPUT_DIR / "pose"
MASKED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
POSE_DIR.mkdir(parents=True, exist_ok=True)
POSES_FILE = POSE_DIR / "camera_poses.json"
# ============================================================
# Step 1: 카메라 포즈를 이용해서 "좋은" 3장 선택
# ============================================================
def select_best_views(input_images: List[Path], num_views: int = 3) -> List[Path]:
    """
    Adaptive Hybrid 알고리즘으로 Veo3 최적화 뷰 선택
    
    - 균일 분포: Fixed Grid (0°, 120°, -120° 근처 선택)
    - 불규칙 분포: Robust Greedy (3D 거리 최대화)
    
    자동으로 입력 분포를 분석하여 최적 전략 사용
    만약 포즈 파일이 없거나 문제 있으면, 이전처럼 랜덤 3장으로 fallback.
    """
    def pick_best_3(
        poses: Dict[str, dict],
        elevation_limit: float = 30.0,
        num_views: int = 3
    ):
        """
        Adaptive Hybrid: Veo3 few-shot 최적화 알고리즘
        
        전략:
        - 균일 분포 입력: Fixed Grid (이상적 위치 0°, 120°, -120°)
        - 불규칙 입력: Robust Greedy (3D 거리 최대화)
        
        자동으로 입력 분포를 분석하여 최적 전략 선택
        """
        center = compute_center(poses)
        view_infos = []
        
        for filename, p in poses.items():
            azimuth, elevation, distance = compute_azimuth_elevation(p["position"], center)
            
            # 3D 단위 벡터로 변환 (구면 거리 계산용)
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            x = math.cos(el_rad) * math.sin(az_rad)
            y = math.sin(el_rad)
            z = math.cos(el_rad) * math.cos(az_rad)
            
            view_infos.append({
                "filename": filename,
                "azimuth": azimuth,
                "elevation": elevation,
                "distance": distance,
                "vec": (x, y, z)
            })
        
        # 1) Elevation 필터링
        filtered = [v for v in view_infos if abs(v["elevation"]) <= elevation_limit]
        if len(filtered) < num_views:
            print(f"  [WARN] Only {len(filtered)} views within ±{elevation_limit}°, relaxing filter")
            filtered = sorted(view_infos, key=lambda v: abs(v["elevation"]))[:max(num_views, len(view_infos))]
        
        # 2) 입력 분포 분석: 이상적 위치에 가까운 뷰가 있는지 확인
        ideal_positions = [(0, 0), (120, 0), (-120, 0)]
        close_to_ideal = []
        
        for ideal_az, ideal_el in ideal_positions:
            for v in filtered:
                # 원형 거리 계산
                az_diff = min(abs(v["azimuth"] - ideal_az), 360 - abs(v["azimuth"] - ideal_az))
                el_diff = abs(v["elevation"] - ideal_el)
                
                if az_diff < 25 and el_diff < 15:  # 이상적 위치 근처
                    close_to_ideal.append((ideal_az, v))
                    break
        
        # 3) 전략 선택
        if len(close_to_ideal) >= 2:
            print(f"  [INFO] Uniform distribution detected, using Fixed Grid strategy")
            return _fixed_grid_selection(filtered, ideal_positions)
        else:
            print(f"  [INFO] Irregular distribution detected, using Robust Greedy strategy")
            return _robust_greedy_selection(filtered)
    
    print(f"\n[Step 1] Selecting {num_views} best views from {len(input_images)} images using camera poses...")

    # 1) 포즈 JSON 로드
    if not POSES_FILE.exists():
        print(f"  [WARN] {POSES_FILE} not found. Falling back to random selection.")
        if len(input_images) <= num_views:
            return input_images
        return random.sample(input_images, num_views)

    poses = load_camera_poses(POSES_FILE)
    if not poses:
        print("  [WARN] No poses found in camera_poses.json. Falling back to random selection.")
        if len(input_images) <= num_views:
            return input_images
        return random.sample(input_images, num_views)

    # 2) 포즈 기반으로 '좋은 3장' 선택 (Veo3 최적화)
    chosen = pick_best_3(poses, elevation_limit=30.0, num_views=num_views)
    chosen_filenames = [v["filename"] for v in chosen]
    print("  Chosen filenames (from poses):", chosen_filenames)
    
    # 선택된 뷰의 상세 정보 출력
    for v in chosen:
        print(f"    - {v['filename']}: azimuth={v['azimuth']:.1f}°, elevation={v['elevation']:.1f}°")

    # 3) 파일 이름 -> 실제 Path로 매핑
    name_to_path = {p.name: p for p in input_images}
    selected_paths: List[Path] = []
    for name in chosen_filenames:
        if name in name_to_path:
            selected_paths.append(name_to_path[name])
        else:
            print(f"  [WARN] Pose exists for {name} but file not found in INPUT_DIR.")

    # 4) 혹시 3장 다 못 찾았으면, 나머지는 랜덤으로 채우기
    if len(selected_paths) < num_views:
        print(f"  [INFO] Only {len(selected_paths)} matched. Filling the rest randomly.")
        remaining = [p for p in input_images if p not in selected_paths]
        extra = random.sample(remaining, min(num_views - len(selected_paths), len(remaining)))
        selected_paths.extend(extra)

    print(f"  Final selected views: {[p.name for p in selected_paths]}")
    return selected_paths

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

input_images = (
    list(INPUT_DIR.glob("*.png")) +
    list(INPUT_DIR.glob("*.jpg"))
)
if len(input_images) == 0:
    print(f"Error: No images found in {INPUT_DIR}")
    exit()
print(f"Found {len(input_images)} input images")

# Step 1: 뷰 선택
selected_views = select_best_views(input_images, num_views=3)