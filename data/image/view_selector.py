import json
import math
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class ViewSelector:
    """
    카메라 포즈 기반 Veo3 최적화 뷰 선택기

    - Adaptive Hybrid 알고리즘 적용
      - 균일 분포: Fixed Grid
      - 불규칙 분포: Robust Greedy
    - fallback: 포즈 없거나 문제 시 랜덤 선택
    """

    def __init__(self, input_images: List[Path], poses_file: Optional[Path] = None):
        self.input_images = input_images
        self.poses_file = poses_file

    def select_best_views(self, num_views: int = 3) -> List[Path]:
        print(f"\n[ViewSelector] {len(self.input_images)}개 이미지에서 {num_views}장 선택 중...")

        poses = self._load_poses()
        if poses:
            chosen = self._pick_best_3(poses, elevation_limit=30.0, num_views=num_views)
            chosen_filenames = [v["filename"] for v in chosen]
            print("  선택된 파일 (포즈 기반):", chosen_filenames)
        else:
            print("  [WARN] 포즈 데이터 없음. 랜덤 선택으로 대체합니다.")
            return self._random_selection(num_views)

        # 파일 이름 -> Path 매핑
        name_to_path = {p.name: p for p in self.input_images}
        selected_paths: List[Path] = []
        for name in chosen_filenames:
            if name in name_to_path:
                selected_paths.append(name_to_path[name])
            else:
                print(f"  [WARN] 포즈 파일에 {name} 존재하지만 이미지 파일 없음.")

        # 부족하면 랜덤으로 채우기
        if len(selected_paths) < num_views:
            print(f"  [INFO] {len(selected_paths)}장만 매칭됨. 나머지는 랜덤으로 채웁니다.")
            remaining = [p for p in self.input_images if p not in selected_paths]
            extra = random.sample(remaining, min(num_views - len(selected_paths), len(remaining)))
            selected_paths.extend(extra)

        print(f"  최종 선택: {[p.name for p in selected_paths]}")
        return selected_paths

    # -------------------- 내부 유틸리티 --------------------
    def _load_poses(self) -> Optional[Dict[str, dict]]:
        """포즈 파일 로드"""
        if self.poses_file and self.poses_file.exists():
            try:
                with open(self.poses_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"  [ERROR] 포즈 파일 로드 실패: {e}")
                return None
        return None

    def _compute_center(self, poses: Dict[str, dict]) -> List[float]:
        """모든 look_at의 평균을 '객체 중심'으로 사용"""
        xs, ys, zs = [], [], []
        for p in poses.values():
            lx, ly, lz = p["look_at"]
            xs.append(lx)
            ys.append(ly)
            zs.append(lz)
        n = len(xs) if xs else 1
        return [sum(xs)/n, sum(ys)/n, sum(zs)/n]

    def _compute_azimuth_elevation(self, position: List[float], center: List[float]) -> Tuple[float, float, float]:
        """카메라 위치와 객체 중심으로부터 azimuth, elevation, distance 계산"""
        cx, cy, cz = center
        px, py, pz = position
        vx, vy, vz = px - cx, py - cy, pz - cz
        r = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-8
        azimuth = math.degrees(math.atan2(vx, vz))
        elevation = math.degrees(math.asin(vy / r))
        return azimuth, elevation, r

    def _pick_best_3(self, poses: Dict[str, dict], elevation_limit: float = 30.0, num_views: int = 3):
        """Adaptive Hybrid 알고리즘으로 3장 선택"""
        center = self._compute_center(poses)
        view_infos = []

        for filename, p in poses.items():
            azimuth, elevation, distance = self._compute_azimuth_elevation(p["position"], center)
            # 3D 단위 벡터 (구면 거리용)
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
            print(f"  [WARN] ±{elevation_limit}° 내 뷰가 {len(filtered)}개뿐. 필터 완화 중...")
            filtered = sorted(view_infos, key=lambda v: abs(v["elevation"]))[:max(num_views, len(view_infos))]

        # 2) 입력 분포 분석
        ideal_positions = [(0, 0), (120, 0), (-120, 0)]
        close_to_ideal = []
        for ideal_az, ideal_el in ideal_positions:
            for v in filtered:
                az_diff = min(abs(v["azimuth"] - ideal_az), 360 - abs(v["azimuth"] - ideal_az))
                el_diff = abs(v["elevation"] - ideal_el)
                if az_diff < 25 and el_diff < 15:
                    close_to_ideal.append((ideal_az, v))
                    break

        # 3) 전략 선택
        if len(close_to_ideal) >= 2:
            print(f"  [INFO] 균일 분포 감지됨, Fixed Grid 전략 사용")
            return self._fixed_grid_selection(filtered, ideal_positions)
        else:
            print(f"  [INFO] 불규칙 분포 감지됨, Robust Greedy 전략 사용")
            return self._robust_greedy_selection(filtered)

    def _fixed_grid_selection(self, filtered: List[dict], ideal_positions: List[Tuple[float, float]]) -> List[dict]:
        """Fixed Grid: 이상적 위치(0°, 120°, -120°)에 가장 가까운 뷰 선택"""
        chosen = []
        remaining = filtered.copy()
        
        for ideal_az, ideal_el in ideal_positions:
            if not remaining:
                break
            best = min(remaining, key=lambda v: 
                math.sqrt((v["azimuth"] - ideal_az)**2 + (v["elevation"] - ideal_el)**2)
            )
            chosen.append(best)
            remaining.remove(best)
        
        return sorted(chosen, key=lambda v: v["azimuth"])

    def _robust_greedy_selection(self, filtered: List[dict]) -> List[dict]:
        """Robust Greedy: 3D 구면 거리 기반 최대 분산 선택"""
        def vec_distance(v1, v2):
            dot = sum(a * b for a, b in zip(v1["vec"], v2["vec"]))
            return math.degrees(math.acos(max(-1, min(1, dot))))
        
        first = min(filtered, key=lambda v: abs(v["azimuth"]) * 1.0 + abs(v["elevation"]) * 3.0)
        chosen = [first]
        remaining = [v for v in filtered if v is not first]
        
        if len(remaining) < 2:
            return chosen + remaining
        
        second = max(remaining, key=lambda v: vec_distance(first, v) - abs(v["elevation"]) * 2)
        chosen.append(second)
        remaining = [v for v in remaining if v is not second]
        
        if len(remaining) < 1:
            return chosen + remaining
        
        third = max(remaining, key=lambda v: 
            min(vec_distance(first, v), vec_distance(second, v)) - abs(v["elevation"]) * 2
        )
        chosen.append(third)
        
        return sorted(chosen, key=lambda v: v["azimuth"])

    def _random_selection(self, num_views: int) -> List[Path]:
        if len(self.input_images) <= num_views:
            return self.input_images
        return random.sample(self.input_images, num_views)
