"""
Hard Negative Mining for YOLO Segmentation

학습된 모델로 배경 이미지들을 추론하여
False Positive (잘못 검출한 것들)을 자동으로 수집하여
Negative 샘플로 추가

사용법:
    python generate_hard_negatives.py --model runs/segment/monkey_finetune/weights/best.pt
"""

from ultralytics import YOLO
from pathlib import Path
import random
from PIL import Image
import shutil

def generate_hard_negatives(
    model_path: str,
    dataset_dir: Path = Path("dataset/train"),
    output_dir: Path = Path("output/dataset/result"),
    max_negatives: int = 500,
    confidence_threshold: float = 0.3  # 낮은 confidence도 수집 (애매한 것들)
):
    """
    학습된 모델로 배경 이미지를 추론하여 False Positive 수집
    
    Args:
        model_path: 학습된 YOLO 모델 경로
        dataset_dir: Places365 데이터셋 경로
        output_dir: YOLO 데이터셋 출력 경로
        max_negatives: 최대 negative 샘플 수
        confidence_threshold: 검출 confidence 임계값
    """
    print(f"\n[Hard Negative Mining]")
    print(f"Model: {model_path}")
    print(f"Target: {max_negatives} hard negatives")
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 배경 이미지 수집
    all_backgrounds = []
    for category_dir in dataset_dir.iterdir():
        if category_dir.is_dir():
            images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
            all_backgrounds.extend(images)
    
    print(f"Found {len(all_backgrounds)} background images")
    
    # 랜덤하게 섞기
    random.shuffle(all_backgrounds)
    
    # Hard negative 수집
    hard_negatives = []
    images_train = output_dir / "images" / "train"
    labels_train = output_dir / "labels" / "train"
    
    for idx, bg_path in enumerate(all_backgrounds):
        if len(hard_negatives) >= max_negatives:
            break
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed: {idx+1}/{len(all_backgrounds)}, Found: {len(hard_negatives)}")
        
        try:
            # 추론
            results = model.predict(str(bg_path), conf=confidence_threshold, verbose=False)
            
            # False Positive가 있는지 확인
            if results and len(results[0].boxes) > 0:
                # 검출됨 → Hard Negative!
                hard_negatives.append(bg_path)
                
                # YOLO 데이터셋에 추가 (빈 레이블)
                img = Image.open(bg_path).convert("RGB")
                
                # 1024x1024로 리사이즈 (기존 데이터와 동일)
                img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                
                output_filename = f"hard_negative_{len(hard_negatives):04d}.png"
                image_path = images_train / output_filename
                label_path = labels_train / output_filename.replace('.png', '.txt')
                
                # 이미지 저장
                img.save(image_path)
                
                # 빈 레이블 저장 (negative sample)
                label_path.write_text("")
                
                print(f"    ✓ Hard negative: {bg_path.name} (conf={results[0].boxes.conf.max():.3f})")
        
        except Exception as e:
            print(f"    ✗ Error: {bg_path.name}: {e}")
            continue
    
    print(f"\n✓ Collected {len(hard_negatives)} hard negatives")
    print(f"  Saved to: {images_train}")
    
    return hard_negatives


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hard Negative Mining for YOLO")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument("--max", type=int, default=500, help="Maximum number of hard negatives")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    
    args = parser.parse_args()
    
    generate_hard_negatives(
        model_path=args.model,
        max_negatives=args.max,
        confidence_threshold=args.conf
    )


if __name__ == "__main__":
    main()
