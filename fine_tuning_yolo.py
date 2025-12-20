"""
YOLOv8 Segmentation Fine-tuning
Dataset: Custom synthetic dataset from pipeline
Model: YOLOv8m-seg (Medium) - COCO pretrained → Fine-tuning

사용법:
    python fine_tuning_yolo.py {프로젝트명}
    python fine_tuning_yolo.py train
    python fine_tuning_yolo.py my_object --epochs 200
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import time
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))
from util.paths import JobPaths


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 세그멘테이션 모델 파인튜닝",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
    python fine_tuning_yolo.py train           # train 프로젝트 데이터로 학습
    python fine_tuning_yolo.py my_object       # my_object 프로젝트 데이터로 학습
    python fine_tuning_yolo.py train --epochs 200 --batch 8
        """
    )
    parser.add_argument("project", type=str, help="프로젝트명")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수 (기본: 100)")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기 (기본: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="이미지 크기 (기본: 640)")
    parser.add_argument("--model", type=str, default="yolov8m-seg.pt", help="베이스 모델 (기본: yolov8m-seg.pt)")
    args = parser.parse_args()

    # 경로 설정
    base_dir = Path(__file__).parent
    paths = JobPaths(base_dir=base_dir, project_name=args.project)
    
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print(f"🚀 YOLO Fine-tuning: {args.project}")
    print("=" * 60)
    print(f"  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    
    # 데이터셋 경로
    data_yaml = paths.yolo_dataset_dir / "data.yaml"
    
    if not data_yaml.exists():
        print(f"\n❌ Error: {data_yaml} not found!")
        print("   먼저 main.py를 실행하여 데이터셋을 생성하세요.")
        print(f"   python main.py {args.project}")
        return

    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Base Model: {args.model}")
    print("=" * 60)

    model = None
    max_retries = 10
    retry_count = 0
    
    # 학습 결과 저장 경로
    train_name = args.project

    while retry_count < max_retries:
        try:
            last_checkpoint_path = Path(f"runs/segment/{train_name}/weights/last.pt")

            # Start or resume training
            if last_checkpoint_path.exists():
                print("\n" + "=" * 60)
                print(f"✅ Checkpoint found! Resuming training from: {last_checkpoint_path}")
                print("=" * 60 + "\n")
                model = YOLO(last_checkpoint_path)
                model.train(resume=True)
            else:
                print("\n" + "=" * 60)
                print(f"🚀 Starting Fine-tuning from pretrained {args.model}...")
                print("=" * 60 + "\n")
                
                model = YOLO(args.model)
                
                model.train(
                    data=str(data_yaml),
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=device,
                    project='runs/segment',
                    name=train_name,
                    exist_ok=True,
                    # Fine-tuning 최적화 설정
                    lr0=0.01,           # 초기 학습률
                    lrf=0.01,           # 최종 학습률 비율
                    warmup_epochs=3,    # Warmup 에폭
                    weight_decay=0.0005,
                    augment=True,       # 데이터 증강
                    mosaic=1.0,         # Mosaic augmentation
                    mixup=0.1,          # Mixup augmentation
                    copy_paste=0.1,     # Copy-paste augmentation (seg용)
                )
            
            print("\n" + "=" * 60)
            print("✅ Training successfully completed!")
            print("=" * 60)
            break  # Exit loop on success

        except Exception as e:
            retry_count += 1
            print(f"\n🔥🔥🔥 An error occurred: {e} 🔥🔥🔥")
            print(f"Attempting to resume... (Attempt {retry_count}/{max_retries})")
            
            # Release memory
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared.")

            if retry_count >= max_retries:
                print("❌ Maximum retries reached. Exiting.")
                return
            
            print("Waiting 30 seconds before retrying...")
            time.sleep(30)

    if model is None:
        print("❌ Training could not be initialized or failed completely.")
        return
        
    # --- Final Validation ---
    print("\n" + "=" * 60)
    print("Training process finished! Running final validation...")
    print("=" * 60)
    
    try:
        metrics = model.val()
        
        print("\nFinal Metrics:")
        print(f"  mAP50: {metrics.seg.map50:.3f}")
        print(f"  mAP50-95: {metrics.seg.map:.3f}")
        print(f"  Precision: {metrics.seg.mp:.3f}")
        print(f"  Recall: {metrics.seg.mr:.3f}")

    except Exception as e:
        print(f"🔥🔥🔥 An error occurred during final validation: {e} 🔥🔥🔥")

    print(f"\n📁 결과물 위치:")
    print(f"  Best model: runs/segment/{train_name}/weights/best.pt")
    print(f"  Last model: runs/segment/{train_name}/weights/last.pt")
    print(f"  Results:    runs/segment/{train_name}/")


if __name__ == "__main__":
    main()
