#!/usr/bin/env python3
"""
Hard Negative Sampling - 합성에 사용된 배경 추가

합성 시 저장된 배경 원본들을:
- 60% 확률로 선택
- 객체 없는 원본 상태로 negative 추가
- Label: 빈 파일 (객체 없음)
- Train/Val split: 배경이 속한 split과 동일하게 유지
"""

import random
import shutil
from pathlib import Path
from tqdm import tqdm

# 경로 설정
BASE_DIR = Path("/home/rocknroll1397/Image_Analyzer")
OUTPUT_DIR = BASE_DIR / "output" / "dataset" / "result"
BACKGROUNDS_DIR = OUTPUT_DIR / "backgrounds"  # 합성 시 저장된 배경들

IMAGES_TRAIN = OUTPUT_DIR / "images" / "train"
IMAGES_VAL = OUTPUT_DIR / "images" / "val"
LABELS_TRAIN = OUTPUT_DIR / "labels" / "train"
LABELS_VAL = OUTPUT_DIR / "labels" / "val"


def collect_saved_backgrounds() -> dict:
    """
    합성 시 저장된 배경 이미지 수집
    
    Returns:
        {split: [bg_paths]} - train/val별 배경 경로 리스트
    """
    print("[Step 1] Collecting saved backgrounds...")
    
    backgrounds = {'train': [], 'val': []}
    
    for split in ['train', 'val']:
        split_dir = BACKGROUNDS_DIR / split
        if split_dir.exists():
            bg_files = sorted(split_dir.glob("bg*.png"))
            backgrounds[split] = bg_files
            print(f"  {split}: {len(bg_files)} backgrounds")
    
    total = sum(len(bgs) for bgs in backgrounds.values())
    print(f"  ✓ Total: {total} backgrounds")
    
    return backgrounds


def add_hard_negatives(selection_rate: float = 0.6):
    """
    합성에 사용된 배경을 negative로 추가
    
    Args:
        selection_rate: 배경 선택 확률 (기본 60%)
    """
    print("=" * 60)
    print(f"Adding Hard Negatives from Used Backgrounds ({selection_rate:.0%})")
    print("=" * 60)
    
    # 1. 저장된 배경 수집
    backgrounds = collect_saved_backgrounds()
    
    if not backgrounds['train'] and not backgrounds['val']:
        print("\n⚠ Error: No saved backgrounds found!")
        print(f"  Please check: {BACKGROUNDS_DIR}")
        return
    
    # 2. 60% 확률로 선택
    print(f"\n[Step 2] Selecting backgrounds with {selection_rate:.0%} probability...")
    selected = {'train': [], 'val': []}
    
    for split, bg_paths in backgrounds.items():
        for bg_path in bg_paths:
            if random.random() < selection_rate:
                selected[split].append(bg_path)
    
    print(f"  ✓ Selected {len(selected['train']) + len(selected['val'])} backgrounds")
    print(f"    Train: {len(selected['train'])}")
    print(f"    Val: {len(selected['val'])}")
    
    # 3. 기존 negative 번호 찾기
    print(f"\n[Step 3] Finding next available negative index...")
    existing_negatives = list(IMAGES_TRAIN.glob("negative_*.png")) + list(IMAGES_VAL.glob("negative_*.png"))
    
    if existing_negatives:
        max_idx = max([
            int(p.stem.split('_')[1]) 
            for p in existing_negatives 
            if len(p.stem.split('_')) > 1 and p.stem.split('_')[1].isdigit()
        ])
        start_idx = max_idx + 1
    else:
        start_idx = 0
    
    print(f"  Starting from negative_{start_idx:04d}")
    
    # 4. Negative 추가
    print(f"\n[Step 4] Adding negatives...")
    neg_idx = start_idx
    
    for split in ['train', 'val']:
        for bg_path in tqdm(selected[split], desc=f"Adding {split} negatives", unit="bg"):
            filename = f"negative_{neg_idx:04d}.png"
            
            # Split에 따라 저장 위치 결정
            if split == 'train':
                img_dest = IMAGES_TRAIN / filename
                label_dest = LABELS_TRAIN / filename.replace('.png', '.txt')
            else:
                img_dest = IMAGES_VAL / filename
                label_dest = LABELS_VAL / filename.replace('.png', '.txt')
            
            # 이미지 복사
            shutil.copy(bg_path, img_dest)
            
            # 빈 label 생성
            label_dest.write_text("")
            
            neg_idx += 1
    
    # 5. 최종 통계
    print(f"\n[Step 5] Final statistics...")
    train_images = len(list(IMAGES_TRAIN.glob("*.png")))
    val_images = len(list(IMAGES_VAL.glob("*.png")))
    
    empty_train = sum(1 for p in LABELS_TRAIN.glob("*.txt") if p.stat().st_size == 0)
    empty_val = sum(1 for p in LABELS_VAL.glob("*.txt") if p.stat().st_size == 0)
    
    print(f"  Total images: {train_images + val_images} ({train_images} train, {val_images} val)")
    print(f"  Negatives: {empty_train + empty_val} ({empty_train} train, {empty_val} val)")
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully added {len(selected['train']) + len(selected['val'])} hard negatives!")
    print("=" * 60)


def main():
    """
    메인 엔트리 포인트
    """
    print("\nConfiguration:")
    print(f"  Backgrounds source: {BACKGROUNDS_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Selection rate: 60%")
    print()
    
    response = input("Add hard negatives from saved backgrounds? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    add_hard_negatives(selection_rate=0.6)


if __name__ == "__main__":
    main()
