"""
Real-time Webcam Segmentation using Fine-tuned YOLOv8
Press 'q' to quit, 's' to save screenshot
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

def draw_segmentation_overlay(frame, results, alpha=0.5):
    """
    세그멘테이션 마스크를 프레임에 오버레이
    
    Args:
        frame: 원본 프레임
        results: YOLO 결과
        alpha: 투명도 (0=투명, 1=불투명)
    """
    overlay = frame.copy()
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
        boxes = results[0].boxes.data.cpu().numpy()  # (N, 6) [x1,y1,x2,y2,conf,cls]
        
        # 각 객체에 대해
        for idx, (mask, box) in enumerate(zip(masks, boxes)):
            # Mask resize to frame size
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_bool = mask_resized > 0.5
            
            # 랜덤 색상 (객체마다 다른 색)
            color = np.random.randint(0, 255, 3).tolist()
            
            # 마스크 영역에 색상 적용
            overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
            
            # Bounding box 그리기
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # 레이블 표시
            label = f"{results[0].names[int(cls)]}: {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - text_h - 10), 
                         (int(x1) + text_w, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 오버레이 합성
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return result

def main():
    # 모델 경로
    model_path = Path("runs/segment/monkey_finetune/weights/best.pt")
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please run train_yolo.py first to train the model.")
        return
    
    print("="*60)
    print("Loading fine-tuned YOLOv8s-seg model...")
    print("="*60)
    
    # 모델 로드
    model = YOLO(str(model_path))
    
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")
    print("\n" + "="*60)
    print("Starting webcam inference...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'c' - Toggle confidence display")
    print("="*60 + "\n")
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return
    
    # 웹캠 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # FPS 계산용
    fps_time = time.time()
    fps = 0
    frame_count = 0
    screenshot_count = 0
    show_conf = True
    
    print("✅ Webcam opened successfully!")
    print("Starting real-time segmentation...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Cannot read frame")
            break
        
        # YOLO 추론
        results = model(frame, conf=0.25, iou=0.45, verbose=False)
        
        # 세그멘테이션 오버레이
        annotated_frame = draw_segmentation_overlay(frame.copy(), results, alpha=0.4)
        
        # FPS 계산
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - fps_time)
            fps_time = current_time
        
        # FPS 표시
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 검출 개수 표시
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f"Objects: {num_detections}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Confidence threshold 표시
        if show_conf:
            cv2.putText(annotated_frame, "Conf: 0.25 | IoU: 0.45", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 화면 표시
        cv2.imshow('YOLOv8 Segmentation - Webcam', annotated_frame)
        
        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('c'):
            show_conf = not show_conf
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Webcam inference finished!")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps:.1f}")
    print(f"Screenshots saved: {screenshot_count}")
    print("="*60)

if __name__ == "__main__":
    main()
