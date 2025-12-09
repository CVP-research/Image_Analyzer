"""
YOLOv8 Segmentation Fine-tuning Script
Dataset: Custom synthetic dataset (1600 images)
Model: YOLOv8s-seg (Small) - 빠른 학습 (1-1.5시간)
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # 데이터셋 경로
    data_yaml = Path("output/dataset/result/data.yaml")
    
    if not data_yaml.exists():
        print(f"❌ Error: {data_yaml} not found!")
        print("Please run process_from_backgrounds.py first to generate dataset.")
        return
    
    # YOLOv8s-seg 모델 선택 (빠른 학습, 1600장 데이터에 충분)
    # - YOLOv8n-seg: 너무 가벼움
    # - YOLOv8s-seg: ✅ 빠른 학습(1-1.5시간), 좋은 정확도 (추천)
    # - YOLOv8m-seg: 더 높은 정확도, 느림 (2.5-3시간)
    # - YOLOv8l-seg: 높은 정확도, 매우 느림 (5-6시간)
    # - YOLOv8x-seg: GPU 메모리 부족 위험
    
    print("\n" + "="*60)
    print("Loading YOLOv8s-seg pretrained model...")
    print("="*60 + "\n")
    
    model = YOLO('yolov8s-seg.pt')  # Small 모델 (pretrained on COCO)
    
    # Fine-tuning 설정
    print("\n" + "="*60)
    print("Starting Fine-tuning...")
    print("="*60 + "\n")
    
    results = model.train(
        data=str(data_yaml),
        epochs=100,              # Small 모델은 빠르므로 100 epochs 가능
        imgsz=640,               # 640이 Small에 최적
        batch=24,                # 16 → 24 (GPU 메모리 여유 있음)
        patience=20,             # Early stopping patience
        device=device,
        workers=8,               # 데이터 로딩 병렬화
        
        # Optimizer
        optimizer='AdamW',       # Fine-tuning에 적합
        lr0=0.002,              # 0.001 → 0.002 (더 빠른 수렴)
        lrf=0.01,               # Final learning rate (0.01 * lr0)
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation (품질 향상을 위해 강화)
        hsv_h=0.03,             # Hue augmentation (0.015 → 0.03)
        hsv_s=0.8,              # Saturation (0.7 → 0.8)
        hsv_v=0.5,              # Value (0.4 → 0.5)
        degrees=15.0,           # Rotation (0 → 15°) - 다양한 각도
        translate=0.15,         # Translation (0.1 → 0.15)
        scale=0.7,              # Scaling (0.5 → 0.7)
        shear=0.0,              # Shear (불필요)
        perspective=0.0001,     # Perspective (0 → 0.0001) - 약간 추가
        flipud=0.0,             # Vertical flip (물건은 상하 반전 없음)
        fliplr=0.5,             # Horizontal flip 50%
        mosaic=0.0,             # Mosaic (끄기)
        mixup=0.0,              # MixUp (끄기)
        copy_paste=0.1,         # Copy-paste (0 → 0.1) - 약간 추가
        
        # Training
        cos_lr=True,            # Cosine LR scheduler
        close_mosaic=10,        # Disable mosaic for last N epochs
        amp=True,               # Automatic Mixed Precision (속도 향상)
        cache=True,             # 이미지 캐싱 (첫 epoch 후 매우 빠름)
        
        # Validation
        val=True,
        plots=True,             # Training plots 생성
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        
        # Output
        project='runs/segment',
        name='monkey_finetune',
        exist_ok=True,
        
        # Verbose
        verbose=True,
        seed=42
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"\nBest model saved at: runs/segment/monkey_finetune/weights/best.pt")
    print(f"Last model saved at: runs/segment/monkey_finetune/weights/last.pt")
    print(f"Results and plots: runs/segment/monkey_finetune/")
    
    # Validation on test set
    print("\n" + "="*60)
    print("Running final validation...")
    print("="*60 + "\n")
    
    metrics = model.val()
    
    print("\nFinal Metrics:")
    print(f"  mAP50: {metrics.seg.map50:.3f}")
    print(f"  mAP50-95: {metrics.seg.map:.3f}")
    print(f"  Precision: {metrics.seg.mp:.3f}")
    print(f"  Recall: {metrics.seg.mr:.3f}")
    
    print("\n" + "="*60)
    print("To run inference, use:")
    print("  python inference_webcam.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
