"""
Compositing functions for semantic image composition
"""

import numpy as np
from PIL import Image
from typing import Tuple, List, Dict


def compute_segment_averaged_depth(
    depth_map: np.ndarray,
    segments: List[Dict]
) -> np.ndarray:
    """
    각 segment별로 평균 depth를 계산하여 새로운 depth map 생성
    
    각 segment 영역 내의 모든 픽셀을 해당 segment의 평균 depth 값으로 대체합니다.
    이렇게 하면 각 segment가 하나의 균일한 depth 평면을 가지게 됩니다.
    
    Args:
        depth_map: 원본 depth map (H x W), 값이 클수록 가까움
        segments: segment 정보 리스트, 각 항목은 다음을 포함:
                  - 'segmentation': bool mask (H x W)
                  - 기타 메타데이터 (label, score 등)
    
    Returns:
        segment별 평균으로 재계산된 depth map (H x W)
    """
    # 새로운 depth map 초기화 (원본 복사)
    averaged_depth_map = depth_map.copy()
    
    # 각 segment에 대해 평균 depth 계산 및 적용
    for seg_info in segments:
        mask = seg_info['segmentation']
        
        # Segment 영역이 비어있으면 건너뛰기
        if mask.sum() == 0:
            continue
        
        # 해당 segment 영역의 평균 depth 계산
        segment_avg_depth = depth_map[mask].mean()
        
        # Segment 영역의 모든 픽셀을 평균 depth로 설정
        averaged_depth_map[mask] = segment_avg_depth
    
    return averaged_depth_map


def composite_simple(
    bg_image: Image.Image,
    overlay_image: Image.Image,
    overlay_position: Tuple[int, int]
) -> Image.Image:
    """
    단순 알파 블렌딩 합성 (depth 무시)
    
    Args:
        bg_image: 배경 이미지
        overlay_image: 누끼 이미지 (RGBA)
        overlay_position: 오버레이 중심 위치 (x, y)
    """
    # Ensure overlay image is RGBA
    overlay_image = overlay_image.convert("RGBA")
    
    bg_array = np.array(bg_image.convert("RGB"))
    overlay_array = np.array(overlay_image)
    
    h_bg, w_bg = bg_array.shape[:2]
    h_ov, w_ov = overlay_array.shape[:2]
    
    # 중심 기준으로 top-left 좌표 계산
    center_x, center_y = overlay_position
    x_offset = center_x - w_ov // 2
    y_offset = center_y - h_ov // 2
    
    # 알파 채널 추출 및 premultiplied alpha 적용
    alpha = overlay_array[:, :, 3] / 255.0
    overlay_rgba = overlay_array.astype(np.float32) / 255.0
    overlay_rgb = overlay_rgba[:, :, :3]
    overlay_rgb_premult = overlay_rgb * alpha[:, :, np.newaxis]
    
    # 배경도 float로 변환 (0-1 범위)
    bg_float = bg_array.astype(np.float32) / 255.0
    
    # 합성
    result = bg_float.copy()
    
    for i in range(h_ov):
        for j in range(w_ov):
            alpha_val = alpha[i, j]
            
            # 완전 투명한 픽셀은 건너뛰기
            if alpha_val < 0.001:
                continue
            
            bg_y = y_offset + i
            bg_x = x_offset + j
            
            if 0 <= bg_x < w_bg and 0 <= bg_y < h_bg:
                # Premultiplied alpha 블렌딩
                result[bg_y, bg_x] = (
                    overlay_rgb_premult[i, j] +
                    bg_float[bg_y, bg_x] * (1.0 - alpha_val)
                )
    
    # float(0-1)에서 uint8(0-255)로 변환
    result_uint8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result_uint8)


def composite_with_depth(
    bg_image: Image.Image,
    bg_depth_map: np.ndarray,
    overlay_image: Image.Image,
    overlay_depth: float,
    overlay_position: Tuple[int, int]
) -> Image.Image:
    """
    Depth 기반 합성 (occlusion-aware)
    
    Args:
        bg_image: 배경 이미지
        bg_depth_map: 배경 depth map (정규화 전 원본, 값이 클수록 가까움)
        overlay_image: 누끼 이미지 (RGBA)
        overlay_depth: 오버레이 depth 값 (0-1 정규화됨, 큰 값 = 가까움)
        overlay_position: 오버레이 중심 위치 (x, y)
    
    Returns:
        합성된 이미지
    """
    # Ensure overlay image is RGBA
    overlay_image = overlay_image.convert("RGBA")
    
    bg_array = np.array(bg_image.convert("RGB"))
    overlay_array = np.array(overlay_image)
    
    h_bg, w_bg = bg_array.shape[:2]
    h_ov, w_ov = overlay_array.shape[:2]
    
    # 배경 depth map 정규화 (0-1 범위)
    depth_min, depth_max = bg_depth_map.min(), bg_depth_map.max()
    bg_depth_norm = (bg_depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    
    # 중심 기준으로 top-left 좌표 계산
    center_x, center_y = overlay_position
    x_offset = center_x - w_ov // 2
    y_offset = center_y - h_ov // 2
    
    # 알파 채널 추출 및 premultiplied alpha 적용
    alpha = overlay_array[:, :, 3] / 255.0
    # RGBA를 float로 변환
    overlay_rgba = overlay_array.astype(np.float32) / 255.0
    overlay_rgb = overlay_rgba[:, :, :3]
    
    # Premultiplied alpha: RGB 값을 알파로 미리 곱함
    # 이렇게 하면 투명 부분의 검정 RGB(0,0,0)가 알파 0과 곱해져서 0이 됨
    overlay_rgb_premult = overlay_rgb * alpha[:, :, np.newaxis]
    
    # 배경도 float로 변환 (0-1 범위)
    bg_float = bg_array.astype(np.float32) / 255.0
    
    # 합성
    result = bg_float.copy()
    occlusion_count = 0
    total_pixels = 0
    
    for i in range(h_ov):
        for j in range(w_ov):
            # 완전 투명한 픽셀은 건너뛰기
            alpha_val = alpha[i, j]
            if alpha_val < 0.001:  # 거의 투명
                continue
            
            bg_y = y_offset + i
            bg_x = x_offset + j
            
            if 0 <= bg_x < w_bg and 0 <= bg_y < h_bg:
                total_pixels += 1
                
                # Depth 기반 occlusion
                bg_depth_value = bg_depth_norm[bg_y, bg_x]
                
                # bg_depth > overlay_depth이면 배경이 더 가까움 -> 오버레이 숨김
                if bg_depth_value > overlay_depth:
                    occlusion_count += 1
                    continue
                
                # Premultiplied alpha 블렌딩
                # overlay_rgb_premult는 이미 alpha가 곱해진 상태
                # 배경에는 (1 - alpha)를 곱함
                result[bg_y, bg_x] = (
                    overlay_rgb_premult[i, j] +
                    bg_float[bg_y, bg_x] * (1.0 - alpha_val)
                )
    
    if total_pixels > 0:
        occlusion_ratio = occlusion_count / total_pixels
        print(f"    Depth occlusion: {occlusion_count}/{total_pixels} pixels ({occlusion_ratio*100:.1f}%)")
        
        # 가려짐이 너무 심하면 경고
        if occlusion_ratio > 0.7:
            print(f"    ⚠ Warning: High occlusion ratio ({occlusion_ratio*100:.1f}%) - overlay may be too hidden")
    
    # float(0-1)에서 uint8(0-255)로 변환하여 이미지 반환
    result_uint8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result_uint8), occlusion_ratio if total_pixels > 0 else 0.0


def compute_segment_depth(
    depth_map: np.ndarray,
    segment_mask: np.ndarray,
    depth_offset: float = 0.05
) -> float:
    """
    Segment 영역의 평균 depth 계산 및 오프셋 적용
    
    Args:
        depth_map: 배경 depth map (값이 클수록 가까움)
        segment_mask: segment 마스크 (bool array)
        depth_offset: 앞으로 당길 오프셋 (비율)
    
    Returns:
        조정된 overlay depth 값 (정규화된 0-1 범위)
    """
    if segment_mask.sum() == 0:
        # segment가 없으면 중간 depth 반환
        return 0.5
    
    # Segment 영역의 평균 depth 계산
    segment_depth_raw = depth_map[segment_mask].mean()
    
    # Depth map 정규화 (0-1 범위로)
    depth_min, depth_max = depth_map.min(), depth_map.max()
    segment_depth_norm = (segment_depth_raw - depth_min) / (depth_max - depth_min + 1e-8)
    
    # 오버레이를 segment보다 약간 앞에 배치 (더 큰 값 = 더 가까움)
    overlay_depth = min(1.0, segment_depth_norm + depth_offset)
    
    print(f"    Segment depth: {segment_depth_norm:.3f} → Overlay depth: {overlay_depth:.3f} (offset: +{depth_offset})")
    
    return overlay_depth


def composite_on_segment(
    bg_image: Image.Image,
    segment_mask: np.ndarray,
    overlay_image: Image.Image,
    base_scale: float = 1.0,
    use_depth: bool = False,
    bg_depth_map: np.ndarray = None,
    depth_offset: float = 0.05
) -> Tuple[Image.Image, float]:
    """
    Segment 영역에 오버레이 합성 - segment 크기에 맞춰 자동 스케일링
    
    Args:
        bg_image: 배경 이미지
        segment_mask: segment 마스크 (bool array)
        overlay_image: 누끼 이미지 (RGBA)
        base_scale: 기본 스케일 (segment 대비)
        use_depth: depth 기반 occlusion 사용 여부
        bg_depth_map: 배경 이미지의 depth map (use_depth=True일 때 필요)
        depth_offset: overlay를 앞으로 당길 오프셋 (기본 0.05)
    
    Returns:
        (합성된 이미지, occlusion 비율)
    """
    # Segment 영역 분석
    ys, xs = np.where(segment_mask)
    if len(xs) == 0:
        # segment 없으면 중앙에 배치
        center_x, center_y = bg_image.size[0] // 2, bg_image.size[1] // 2
        segment_width = bg_image.size[0] // 4
        segment_height = bg_image.size[1] // 4
    else:
        # segment의 바운딩 박스 계산
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        segment_width = max_x - min_x
        segment_height = max_y - min_y
        
        # segment 중심에 배치
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
    
    # 오버레이 스케일 자동 계산
    # segment 크기의 60-80% 정도로 맞춤
    overlay_w, overlay_h = overlay_image.size
    
    # 가로/세로 비율 유지하면서 segment에 맞춤
    scale_x = (segment_width * 0.7 * base_scale) / overlay_w
    scale_y = (segment_height * 0.7 * base_scale) / overlay_h
    auto_scale = min(scale_x, scale_y)
    
    # 너무 작거나 크지 않도록 제한
    auto_scale = np.clip(auto_scale, 0.1, 3.0)
    
    if auto_scale != 1.0:
        new_w = int(overlay_w * auto_scale)
        new_h = int(overlay_h * auto_scale)
        overlay_image = overlay_image.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )
    
    print(f"    Segment size: {segment_width}x{segment_height}, Auto scale: {auto_scale:.2f}, Center: ({center_x}, {center_y})")
    
    # 합성 (adaptive depth offset)
    occlusion_ratio = 0.0
    if use_depth and bg_depth_map is not None:
        # 첫 시도: 기본 offset
        current_offset = depth_offset
        max_attempts = 3
        
        for attempt in range(max_attempts):
            # Segment의 depth 계산 (오프셋 적용)
            overlay_depth = compute_segment_depth(bg_depth_map, segment_mask, current_offset)
            
            # Depth 기반 합성
            result, occlusion_ratio = composite_with_depth(
                bg_image,
                bg_depth_map,
                overlay_image,
                overlay_depth,
                (center_x, center_y)
            )
            
            # 가려짐이 70% 이상이면 offset 증가해서 재시도
            if occlusion_ratio > 0.7 and attempt < max_attempts - 1:
                current_offset += 0.1
                print(f"    → Occlusion too high, retrying with offset={current_offset:.2f}")
            else:
                break
    else:
        # 단순 합성 (depth 무시)
        result = composite_simple(
            bg_image,
            overlay_image,
            (center_x, center_y)
        )
    
    return result, occlusion_ratio
