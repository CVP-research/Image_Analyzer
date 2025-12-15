import cv2
import numpy as np
from pathlib import Path
from ultralytics import SAM

# --- 설정 (객체 톤은 반드시 님의 Few-Shot 객체의 주된 색상으로 변경해주세요) ---
INPUT_VIDEO_PATH = Path("/home/rocknroll1397/Image_Analyzer/new/data/77_7859_15670/test3.mp4")
OUTPUT_VIDEO_PATH = Path("/home/rocknroll1397/Image_Analyzer/new/data/77_7859_15670/test3_distorted4.mp4")


# 필요한 모델 가중치 파일의 경로를 지정합니다.
SAM_MODEL_WEIGHTS = '/home/rocknroll1397/Image_Analyzer/sam2_l.pt' 

# --- 핵심 왜곡 함수: SAM이 감지한 영역의 색상을 평균화 (질감 제거) ---
# --- 핵심 왜곡 함수: 질감 제거 + 강력한 모양 단순화 (뭉개기) ---
# --- 핵심 왜곡 함수: 질감 제거 + 강력한 모양 단순화 (뭉개기) ---
def apply_segment_distortion_strong(frame: np.ndarray, masks: list) -> np.ndarray:
    """
    SAM 마스크를 받아 1) 매우 큰 커널로 모양을 확장하고 (크기 키우기)
    2) 동일하게 큰 커널로 닫힘(Closing) 연산을 적용하여 형태를 극단적으로 왜곡/단순화합니다.
    3) 해당 영역의 색상을 평균화하여 질감을 제거합니다.
    """
    processed_frame = frame.copy()
    
    # 🚨 [극단적인 왜곡 설정]: KERNEL_SIZE를 높여 확장 및 단순화 효과를 통합
    H, W, _ = frame.shape
    
    # **매우 공격적인 커널 크기:** 프레임 크기의 1/20 ~ 1/15 수준 (더 큰 왜곡 유도)
    KERNEL_SIZE_DISTORTION = min(H, W) // 20
    KERNEL_SIZE_DISTORTION = max(KERNEL_SIZE_DISTORTION, 25) # 최소 25 픽셀 이상 보장
    
    # 확장과 왜곡에 사용할 공격적인 커널
    aggressive_kernel = np.ones((KERNEL_SIZE_DISTORTION, KERNEL_SIZE_DISTORTION), np.uint8)
    ITERATIONS = 2 # 반복 횟수
    
    for mask_np in masks:
        num_pixels = np.sum(mask_np)
        if num_pixels == 0:
            continue
            
        # 1. 마스크 영역의 평균 색상 계산 (질감 제거 준비)
        object_pixels = frame[mask_np == 1]
        mean_color_bgr = np.mean(object_pixels, axis=0)
        mean_color_bgr = np.uint8(mean_color_bgr)
        
        
        # 🚨 2. [확장 및 극단적인 모양 단순화/왜곡]
        mask_8bit = (mask_np * 255).astype(np.uint8) 
        
        # 2-A. Dilation (확장): 객체 영역을 원본보다 확실하게 키웁니다.
        # 큰 커널을 사용했기 때문에 이 단계에서 이미 어느 정도 단순화가 진행됩니다.
        expanded_mask = cv2.dilate(mask_8bit, aggressive_kernel, iterations=1) 
        
        # 2-B. Closing (단순화/왜곡): 확장된 마스크에 동일한 큰 커널로 닫힘 연산을 적용
        # 닫힘 연산: 침식(Erosion) 후 팽창(Dilation)
        # 이 과정에서 내부 구멍이 메워지고, 모서리가 뭉툭해지며, 형태가 극단적으로 단순화됩니다.
        
        # 1차 침식 (Erosion): 형태를 극단적으로 단순화
        eroded_expanded = cv2.erode(expanded_mask, aggressive_kernel, iterations=1)
        
        # 2차 팽창 (Dilation): 크기를 복구하고 뭉툭한 형태를 완성 (최종 왜곡된 마스크)
        smoothed_mask = cv2.dilate(eroded_expanded, aggressive_kernel, iterations=1)
        
        
        # 3. [프레임에 적용]: 단순화되고 확장된 모양에 평균 색상 적용
        
        smoothed_mask_np = (smoothed_mask // 255).astype(np.uint8)
        
        # 뭉개진 마스크 영역을 평균 색상으로 채우기
        processed_frame[smoothed_mask_np == 1] = mean_color_bgr
            
    return processed_frame

# --- 최종 비디오 처리 파이프라인 (함수명만 변경) ---
def process_video_with_sam_distortion():
    
    # ... (SAM 모델 로드 및 비디오 처리 로직은 이전 코드와 동일) ...

    try:
        sam_model = SAM(SAM_MODEL_WEIGHTS)
        print(f"✅ Ultralytics SAM 모델 ({SAM_MODEL_WEIGHTS}) 로드 완료.")
    except Exception as e:
        print(f"❌ 오류: SAM 모델 로드 실패. 에러: {e}")
        return
    
    cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (frame_width, frame_height))

    print(f"--- SAM 기반 강력 단순화/질감 제거 비디오 생성 시작 ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        H, W, _ = frame.shape
        center_prompt = [(W // 2, H // 2)]
        results = sam_model(frame, points=center_prompt)
        
        masks_list = []
        if results and results[0].masks is not None:
            masks_tensor = results[0].masks.data.cpu().numpy()
            for mask in masks_tensor:
                masks_list.append(mask.astype(np.uint8))
        
        # 4. 강력한 모양 단순화 및 질감 제거 왜곡 적용
        distorted_frame = apply_segment_distortion_strong(frame, masks_list)
        out.write(distorted_frame)
        
    cap.release()
    out.release()
    print(f"✅ SAM 기반 강력 단순화/질감 제거 테스트 비디오 저장 완료: {OUTPUT_VIDEO_PATH.name}")
process_video_with_sam_distortion()