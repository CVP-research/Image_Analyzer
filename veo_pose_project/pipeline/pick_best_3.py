# pick_best_3.py
"""
camera_poses.json 기반으로
- elevation 필터링
- 정면(azimuth≈0°) 1장 무조건 포함
- 나머지 2장은 수평각이 최대한 벌어지게 선택
"""

import json
import math
from pathlib import Path
from typing import Dict, List

# === 설정 ===
BASE_DIR = Path(__file__).resolve().parent
POSES_FILE = BASE_DIR.parent / "backend" / "camera_poses.json"


def load_camera_poses(path: Path) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_center(poses: Dict[str, dict]) -> List[float]:
    xs, ys, zs = [], [], []
    for p in poses.values():
        lx, ly, lz = p["look_at"]
        xs.append(lx); ys.append(ly); zs.append(lz)

    n = len(xs) if xs else 1
    return [sum(xs)/n, sum(ys)/n, sum(zs)/n]


def circular_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def pick_best_3(
    poses: Dict[str, dict],
    elevation_limit: float = 60.0,
    num_views: int = 3
):
    center = compute_center(poses)
    cx, cy, cz = center

    # === 카메라 포즈에서 정보 추출 ===
    view_infos = []
    for filename, p in poses.items():
        px, py, pz = p["position"]
        vx, vy, vz = px - cx, py - cy, pz - cz
        r = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-8

        azimuth = math.degrees(math.atan2(vx, vz))
        elevation = math.degrees(math.asin(vy / r))

        view_infos.append({
            "filename": filename,
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": r,
        })

    # === 1) elevation 필터 ===
    filtered = [
        v for v in view_infos
        if abs(v["elevation"]) <= elevation_limit
    ]
    if len(filtered) < num_views:
        filtered = view_infos

    # === 2) 정면 찾기: azimuth 절대값 최소 ===
    front_view = min(filtered, key=lambda v: abs(v["azimuth"]))

    # front 제거한 나머지 후보들
    remaining = [v for v in filtered if v != front_view]

    if len(remaining) <= 2:
        chosen = [front_view] + remaining
        chosen = chosen[:3]
        chosen.sort(key=lambda v: v["azimuth"])
        return chosen

    # === 3) 나머지에서 첫 번째 기준 선택 ===
    # front와 가장 멀리 떨어진 azimuth 선택
    second = max(
        remaining,
        key=lambda v: circular_distance(v["azimuth"], front_view["azimuth"])
    )

    # === 4) 세 번째 선택: front/second 양쪽과 가장 멀리 떨어진 사진 ===
    remaining2 = [v for v in remaining if v != second]

    def score(v):
        return min(
            circular_distance(v["azimuth"], front_view["azimuth"]),
            circular_distance(v["azimuth"], second["azimuth"])
        )

    third = max(remaining2, key=score)

    chosen = [front_view, second, third]

    # 보기 좋게 정렬
    chosen.sort(key=lambda v: v["azimuth"])
    return chosen


def main():
    if not POSES_FILE.exists():
        print(f"[ERROR] {POSES_FILE} 파일을 찾을 수 없습니다.")
        return

    poses = load_camera_poses(POSES_FILE)
    if not poses:
        print("[ERROR] camera_poses.json 안에 포즈 데이터가 없습니다.")
        return

    chosen = pick_best_3(poses)

    print("\n=== 선택된 3장 (정면 포함) ===")
    for v in chosen:
        print(
            f"- {v['filename']:<15} | "
            f"azimuth={v['azimuth']:>7.2f}°, "
            f"elevation={v['elevation']:>7.2f}°"
        )

    filenames = [v["filename"] for v in chosen]
    print("\n=== 파일명 리스트 (Veo용) ===")
    print(filenames)


if __name__ == "__main__":
    main()
