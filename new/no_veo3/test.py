import cv2
import numpy as np
import os

# --- 1. 설정 변수 (사용자 수정 필요) ---
FILE_NAME = "/home/rocknroll1397/Image_Analyzer/new/data/77_7859_15670/masks/frame000001.png"    # 이미지와 마스크 파일의 공통 이름



def diagnose_mask_file(mask_path):
    print(f"\n--- 마스크 파일 진단 시작: {mask_path} ---")
    
    if not os.path.exists(mask_path):
        print(f"❌ 오류: 지정된 경로에 마스크 파일을 찾을 수 없습니다.")
        return

    try:
        # 마스크를 단일 채널 (GRAYSCALE)로 로드합니다. (IMREAD_GRAYSCALE = 0)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print("❌ 오류: OpenCV가 파일을 이미지로 인식하지 못했습니다. 파일이 손상되었거나 형식이 잘못되었을 수 있습니다.")
            return

        # --- 진단 결과 출력 ---
        print(f"✅ 데이터 타입 (Dtype): {mask.dtype}")
        print(f"✅ 이미지 크기 (Shape): {mask.shape}")
        
        # 픽셀 값 진단
        min_val = np.min(mask)
        max_val = np.max(mask)
        
        print(f"✅ 최소 픽셀 값 (Min): {min_val}")
        print(f"✅ 최대 픽셀 값 (Max): {max_val}")
        # 🚨🚨🚨 핵심 추가 기능: 픽셀 값별 개수 세기 🚨🚨🚨
        unique, counts = np.unique(mask, return_counts=True)
        pixel_counts = dict(zip(unique, counts))
        
        print("\n--- 픽셀 값별 개수 (Pixel Count) ---")
        
        total_pixels = mask.size
        object_pixels = 0
        
        for value, count in pixel_counts.items():
            percentage = (count / total_pixels) * 100
            print(f"  > 픽셀 값 {value:3d}: {count} 개 ({percentage:.2f}%)")
            if value > 0:
                object_pixels += count
                
        background_pixels = pixel_counts.get(0, 0)
        
        print(f"\n✅ 총 픽셀 수: {total_pixels} 개")
        print(f"✅ 배경 (값 0) 픽셀 수: {background_pixels} 개")
        print(f"✅ 객체 (값 > 0) 픽셀 수: {object_pixels} 개")
        
        if max_val <= 1:
            print("\n🚨🚨 진단 결과: 마스크의 최대 픽셀 값이 1 이하입니다. 🚨🚨")
            print("이것이 육안으로 이미지가 검게 보이는 이유입니다. 객체 영역의 값은 1입니다.")
            print("객체 추출 시, 이 마스크에 255를 곱하여 사용해야 합니다.")
        elif max_val <= 255:
            print("\n✅ 진단 결과: 마스크 값은 0~255 범위에 있습니다. 정상입니다.")
            
    except Exception as e:
        print(f"처리 중 예상치 못한 오류 발생: {e}")


# --- 3. 함수 실행 ---
if __name__ == "__main__":
    # 실행 전, 위 FILE_NAME과 BASE_DIR을 실제 경로에 맞게 수정해주세요.
    diagnose_mask_file(FILE_NAME)