"""
YOLOv8 Segmentation Training from Scratch
Dataset: Custom synthetic dataset (1600 positive + 200 negative)
Model: YOLOv8s-seg (Small) - Scratch 학습 (COCO 지식 배제)
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import time

TRAIN_NAME = "train_veo3_v2"

def main():
    # GPU 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # 데이터셋 경로
    data_yaml = Path(f"new/dataset/{TRAIN_NAME}/data.yaml")
    
    if not data_yaml.exists():
        print(f"❌ Error: {data_yaml} not found!")
        print("Please run process_from_backgrounds.py first to generate dataset.")
        return

    model = None
    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        try:
            last_checkpoint_path = Path(f"runs/segment/{TRAIN_NAME}/weights/last.pt")

            # Start or resume training
            if last_checkpoint_path.exists():
                print("\n" + "="*60)
                print(f"✅ Checkpoint found! Resuming training from: {last_checkpoint_path}")
                print("="*60 + "\n")
                model = YOLO(last_checkpoint_path)
                model.train(resume=True)
            else:
                print("\n" + "="*60)
                print("🚀 No checkpoint found. Starting a new training session...")
                print("="*60 + "\n")
                
                model = YOLO('yolov8m-seg.pt')
                
                print("\n" + "="*60)
                print("Starting Fine-tuning...")
                print("="*60 + "\n")
                
                model.train(
                    data=str(data_yaml),
                    epochs=100,
                    imgsz=640,
                    batch=15,
                    device=device,
                    project='runs/segment',
                    name=TRAIN_NAME,
                    exist_ok=True
                )
            
            print("\n" + "="*60)
            print("✅ Training successfully completed!")
            print("="*60)
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
                return # Give up
            
            print("Waiting 30 seconds before retrying...")
            time.sleep(30)

    if model is None:
        print("❌ Training could not be initialized or failed completely.")
        return
        
    # --- Final Validation ---
    print("\n" + "="*60)
    print("Training process finished! Running final validation...")
    print("="*60)
    
    try:
        metrics = model.val()
        
        print("\nFinal Metrics:")
        print(f"  mAP50: {metrics.seg.map50:.3f}")
        print(f"  mAP50-95: {metrics.seg.map:.3f}")
        print(f"  Precision: {metrics.seg.mp:.3f}")
        print(f"  Recall: {metrics.seg.mr:.3f}")

    except Exception as e:
        print(f"🔥🔥🔥 An error occurred during final validation: {e} 🔥🔥🔥")

    print(f"\nBest model saved at: runs/segment/{TRAIN_NAME}/weights/best.pt")
    print(f"Last model saved at: runs/segment/{TRAIN_NAME}/weights/last.pt")
    print(f"Results and plots: runs/segment/{TRAIN_NAME}/")


if __name__ == "__main__":
    main()
