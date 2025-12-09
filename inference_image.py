"""
Image Inference using Fine-tuned YOLOv8
Test with static images
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys

def draw_segmentation_overlay(frame, results, alpha=0.5):
    """세그멘테이션 마스크를 프레임에 오버레이"""
    overlay = frame.copy()
    
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()
        
        for idx, (mask, box) in enumerate(zip(masks, boxes)):
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_bool = mask_resized > 0.5
            
            color = (0, 255, 0)  # Green
            overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
            
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label = f"{results[0].names[int(cls)]}: {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (int(x1), int(y1) - text_h - 10), 
                         (int(x1) + text_w, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_image.py <image_path>")
        print("\nExample:")
        print("  python inference_image.py test.jpg")
        print("  python inference_image.py output/dataset/test_result/images/test_bg00_obj040_bed.png")
        return
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    model_path = Path("runs/segment/monkey_finetune/weights/best.pt")
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    print("="*60)
    print("Loading fine-tuned YOLOv8s-seg model...")
    print("="*60)
    
    model = YOLO(str(model_path))
    
    print(f"Model loaded: {model_path}")
    print(f"Classes: {model.names}")
    print(f"\nProcessing: {image_path}")
    
    # 이미지 로드
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"❌ Error: Cannot read image: {image_path}")
        return
    
    # YOLO 추론
    results = model(image, conf=0.25, iou=0.45, verbose=False)
    
    # 세그멘테이션 오버레이
    annotated_image = draw_segmentation_overlay(image.copy(), results, alpha=0.4)
    
    # 검출 정보 출력
    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"\n✅ Detection completed!")
    print(f"Objects detected: {num_detections}")
    
    if num_detections > 0:
        boxes = results[0].boxes.data.cpu().numpy()
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2, conf, cls = box
            print(f"  Object {idx+1}: {model.names[int(cls)]} ({conf:.3f}) at [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    
    # 결과 저장
    output_path = image_path.parent / f"{image_path.stem}_result{image_path.suffix}"
    cv2.imwrite(str(output_path), annotated_image)
    print(f"\n📸 Result saved: {output_path}")
    
    # 화면 표시
    print("\nPress any key to close the window...")
    cv2.imshow('YOLOv8 Segmentation Result', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
