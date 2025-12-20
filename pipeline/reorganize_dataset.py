"""
배경별로 그룹화하여 train/val 재분배

같은 배경 이미지를 사용한 샘플들을 묶어서 train 또는 val에만 배치
데이터 유출 방지
"""

import shutil
from pathlib import Path
from collections import defaultdict
import random


def parse_filename(filename: str) -> dict:
    """
    파일명 파싱
    - 네거티브: negative_XXXX.png 또는 negative_bgXXXX_YYYY.png
    - 포지티브: bg숫자_객체명_숫자.png
    """
    stem = filename.replace('.png', '')
    
    if stem.startswith('negative_'):
        # negative_1234.png 또는 negative_bg0123_4567.png
        parts = stem.split('_')
        if len(parts) >= 3 and parts[1].startswith('bg'):
            # negative_bg0123_4567
            bg_id = parts[1]  # bg0123
            return {'type': 'negative', 'bg_id': bg_id, 'filename': filename}
        else:
            # negative_1234 (새로 생성한 네거티브)
            return {'type': 'negative_new', 'bg_id': None, 'filename': filename}
    else:
        # bg0123_object_0001.png
        parts = stem.split('_')
        if parts[0].startswith('bg'):
            bg_id = parts[0]
            return {'type': 'positive', 'bg_id': bg_id, 'filename': filename}
        else:
            return {'type': 'unknown', 'bg_id': None, 'filename': filename}


def group_by_background(dataset_path: Path):
    """
    배경별로 그룹화
    """
    train_images = list((dataset_path / "images" / "train").glob("*.png"))
    val_images = list((dataset_path / "images" / "val").glob("*.png"))
    
    all_images = []
    for img in train_images:
        info = parse_filename(img.name)
        info['split'] = 'train'
        info['path'] = img
        all_images.append(info)
    
    for img in val_images:
        info = parse_filename(img.name)
        info['split'] = 'val'
        info['path'] = img
        all_images.append(info)
    
    # 배경별로 그룹화
    bg_groups = defaultdict(list)
    no_bg_samples = []  # bg_id가 없는 샘플들 (새로 생성한 네거티브)
    
    for img_info in all_images:
        if img_info['bg_id'] is None:
            no_bg_samples.append(img_info)
        else:
            bg_groups[img_info['bg_id']].append(img_info)
    
    return bg_groups, no_bg_samples


def redistribute(dataset_path: Path, train_ratio: float = 0.8):
    """
    배경별로 그룹화하여 train/val 재분배
    """
    print("=" * 60)
    print("배경별 그룹화 및 train/val 재분배")
    print("=" * 60)
    
    # 1. 그룹화
    print("\n1. 배경별로 그룹화 중...")
    bg_groups, no_bg_samples = group_by_background(dataset_path)
    
    print(f"   배경 그룹 수: {len(bg_groups)}")
    print(f"   배경 없는 샘플 수: {len(no_bg_samples)}")
    
    # 배경별 샘플 수 통계
    bg_counts = [(bg_id, len(samples)) for bg_id, samples in bg_groups.items()]
    bg_counts.sort(key=lambda x: x[1], reverse=True)
    print(f"\n   배경당 평균 샘플 수: {sum(c for _, c in bg_counts) / len(bg_counts):.1f}")
    print(f"   최대 샘플 수: {bg_counts[0][1]} (배경: {bg_counts[0][0]})")
    print(f"   최소 샘플 수: {bg_counts[-1][1]} (배경: {bg_counts[-1][0]})")
    
    # 2. 배경 그룹을 train/val로 분할
    print("\n2. 배경 그룹을 train/val로 분할 중...")
    bg_list = list(bg_groups.keys())
    random.shuffle(bg_list)
    
    train_bg_count = int(len(bg_list) * train_ratio)
    train_bgs = set(bg_list[:train_bg_count])
    val_bgs = set(bg_list[train_bg_count:])
    
    print(f"   Train 배경 수: {len(train_bgs)}")
    print(f"   Val 배경 수: {len(val_bgs)}")
    
    # 3. 백업 및 임시 디렉토리 생성
    print("\n3. 백업 및 재배치 준비 중...")
    backup_dir = dataset_path / "backup_before_reorg"
    temp_dir = dataset_path / "temp_reorg"
    
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    backup_dir.mkdir(parents=True)
    temp_dir.mkdir(parents=True)
    (temp_dir / "images" / "train").mkdir(parents=True)
    (temp_dir / "images" / "val").mkdir(parents=True)
    (temp_dir / "labels" / "train").mkdir(parents=True)
    (temp_dir / "labels" / "val").mkdir(parents=True)
    
    # 기존 데이터 백업
    shutil.copytree(dataset_path / "images", backup_dir / "images")
    shutil.copytree(dataset_path / "labels", backup_dir / "labels")
    print(f"   백업 완료: {backup_dir}")
    
    # 4. 재배치
    print("\n4. 샘플 재배치 중...")
    train_count = 0
    val_count = 0
    
    # 배경 그룹별 재배치
    for bg_id, samples in bg_groups.items():
        target_split = 'train' if bg_id in train_bgs else 'val'
        
        for sample in samples:
            src_img = sample['path']
            src_label = dataset_path / "labels" / sample['split'] / src_img.name.replace('.png', '.txt')
            
            dst_img = temp_dir / "images" / target_split / src_img.name
            dst_label = temp_dir / "labels" / target_split / src_img.name.replace('.png', '.txt')
            
            shutil.copy2(src_img, dst_img)
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                dst_label.touch()  # 빈 라벨 생성
            
            if target_split == 'train':
                train_count += 1
            else:
                val_count += 1
    
    # 배경 없는 샘플들도 비율에 맞게 재배치
    random.shuffle(no_bg_samples)
    train_no_bg_count = int(len(no_bg_samples) * train_ratio)
    
    for i, sample in enumerate(no_bg_samples):
        target_split = 'train' if i < train_no_bg_count else 'val'
        
        src_img = sample['path']
        src_label = dataset_path / "labels" / sample['split'] / src_img.name.replace('.png', '.txt')
        
        dst_img = temp_dir / "images" / target_split / src_img.name
        dst_label = temp_dir / "labels" / target_split / src_img.name.replace('.png', '.txt')
        
        shutil.copy2(src_img, dst_img)
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.touch()
        
        if target_split == 'train':
            train_count += 1
        else:
            val_count += 1
    
    print(f"   Train: {train_count}개")
    print(f"   Val: {val_count}개")
    
    # 5. 기존 데이터 삭제 및 새 데이터로 교체
    print("\n5. 기존 데이터 교체 중...")
    shutil.rmtree(dataset_path / "images")
    shutil.rmtree(dataset_path / "labels")
    
    shutil.move(str(temp_dir / "images"), str(dataset_path / "images"))
    shutil.move(str(temp_dir / "labels"), str(dataset_path / "labels"))
    
    temp_dir.rmdir()
    
    # 6. 결과 확인
    print("\n6. 결과 확인...")
    train_images = list((dataset_path / "images" / "train").glob("*.png"))
    val_images = list((dataset_path / "images" / "val").glob("*.png"))
    
    train_neg = sum(1 for img in train_images if img.name.startswith('negative_'))
    train_pos = len(train_images) - train_neg
    val_neg = sum(1 for img in val_images if img.name.startswith('negative_'))
    val_pos = len(val_images) - val_neg
    
    print(f"\n   Train: {len(train_images)}개 (포지티브: {train_pos}, 네거티브: {train_neg})")
    print(f"   Val: {len(val_images)}개 (포지티브: {val_pos}, 네거티브: {val_neg})")
    print(f"   비율: Train {len(train_images)/(len(train_images)+len(val_images))*100:.1f}% / Val {len(val_images)/(len(train_images)+len(val_images))*100:.1f}%")
    
    # 데이터 유출 체크
    print("\n7. 데이터 유출 체크...")
    train_bg_groups, _ = group_by_background(dataset_path)
    train_bgs_actual = set()
    val_bgs_actual = set()
    
    for img in train_images:
        info = parse_filename(img.name)
        if info['bg_id']:
            train_bgs_actual.add(info['bg_id'])
    
    for img in val_images:
        info = parse_filename(img.name)
        if info['bg_id']:
            val_bgs_actual.add(info['bg_id'])
    
    overlap = train_bgs_actual & val_bgs_actual
    if overlap:
        print(f"   ⚠️  경고: {len(overlap)}개 배경이 train과 val에 모두 존재!")
        print(f"   중복 배경: {list(overlap)[:10]}")
    else:
        print(f"   ✅ 데이터 유출 없음! Train과 Val에 공통 배경 없음")
    
    print("\n" + "=" * 60)
    print("재배치 완료!")
    print(f"백업 위치: {backup_dir}")
    print("=" * 60)


def main():
    dataset_path = Path("output/dataset/result_sum")
    redistribute(dataset_path, train_ratio=0.8)


if __name__ == "__main__":
    main()
