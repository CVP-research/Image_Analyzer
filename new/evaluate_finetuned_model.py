import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm

# ==========================================
# 🚨 사용자 설정 영역 🚨
# 스크립트 실행 전, 아래 경로들을 올바르게 수정해주세요.
# ==========================================

# 학습 폴더명
# TRAIN_NAME = "train_no_veo3_2"
TRAIN_NAME = "train_veo3"
# TRAIN_NAME = "train_veo3_v1"
# TRAIN_NAME = "train_veo3_v2"
# 1. 파인튜닝된 YOLO segmentation 모델의 .pt 파일 경로
FINE_TUNED_MODEL_PATH = Path(f"/home/rocknroll1397/Image_Analyzer/runs/segment/{TRAIN_NAME}/weights/best.pt")

VAL_DATA_YAML = f"/home/rocknroll1397/Image_Analyzer/new/dataset/{TRAIN_NAME}/data.yaml"

# 2. 평가할 원본 이미지들이 있는 폴더 경로
INPUT_IMAGES_DIR = Path("data/77_7859_15670/images_real")

# 3. 실제 정답 마스크(.png)들이 있는 폴더 경로
#    - segment_server_real.py에서 저장한 마스크들이 있는 폴더를 지정합니다.
#    - 파일명 형식: {원본_이미지명}_{인덱스}_mask.png (예: image_01_00_mask.png)
GROUND_TRUTH_MASKS_DIR = Path("output/masked_frames_real")

# 4. 결과물을 저장할 폴더 경로 (시각화 이미지, AP 결과 텍스트 파일)
OUTPUT_DIR = Path(f"output/evaluation_results_{TRAIN_NAME}")

# 5. AP 계산을 위한 IoU 임계값
IOU_THRESHOLD = 0.75

# ==========================================
# 핵심 유틸리티 함수
# ==========================================

def calculate_iou(mask1, mask2):
    """두 개의 이진 마스크 간의 IoU(Intersection over Union)를 계산합니다."""
    # 마스크가 0 또는 1의 값을 갖도록 이진화
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou

def find_best_match(pred_mask, gt_masks):
    """하나의 예측 마스크에 대해 가장 높은 IoU를 갖는 실제 정답 마스크를 찾습니다."""
    best_iou = -1
    best_gt_idx = -1
    for i, gt_mask in enumerate(gt_masks):
        iou = calculate_iou(pred_mask, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = i
    return best_iou, best_gt_idx

def load_ground_truth_masks(image_name_stem, h, w):
    """특정 이미지에 대한 모든 실제 정답 마스크들을 로드합니다."""
    gt_masks = []
    # 파일 시스템에서 해당 이미지 이름으로 시작하는 모든 마스크 파일을 찾음
    mask_files = sorted(GROUND_TRUTH_MASKS_DIR.glob(f"{image_name_stem}_*_mask.png"))
    
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # 원본 이미지 크기에 맞게 마스크를 리사이즈하거나 패딩해야 할 수 있지만,
            # 여기서는 마스크가 원본 이미지 크기와 동일하다고 가정합니다.
            # 만약 크기가 다르다면, 아래 주석 처리된 리사이즈 코드를 활성화해야 합니다.
            # if mask.shape[:2] != (h, w):
            #     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            gt_masks.append(mask)
    return gt_masks

def visualize_results(image, pred_masks, gt_masks, output_path):
    """예측과 실제 정답 마스크를 원본 이미지 위에 시각화합니다."""
    vis_image = image.copy()
    
    # 실제 정답 마스크 외곽선 그리기 (초록색)
    for gt_mask in gt_masks:
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
        
    # 예측 마스크 외곽선 그리기 (빨간색)
    for pred_mask in pred_masks:
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 0, 255), 2)
        
    cv2.imwrite(str(output_path), vis_image)

# ==========================================
# 메인 평가 로직
# ==========================================

def main():
    """메인 평가 프로세스를 실행합니다."""
    print("🚀 모델 평가를 시작합니다...")
    
    # 1. 설정 확인 및 폴더 생성
    if not FINE_TUNED_MODEL_PATH.exists():
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {FINE_TUNED_MODEL_PATH}")
        return
    if not INPUT_IMAGES_DIR.exists():
        print(f"❌ 오류: 이미지 폴더를 찾을 수 없습니다: {INPUT_IMAGES_DIR}")
        return
    if not GROUND_TRUTH_MASKS_DIR.exists():
        print(f"❌ 오류: 실제 정답 마스크 폴더를 찾을 수 없습니다: {GROUND_TRUTH_MASKS_DIR}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    vis_dir = OUTPUT_DIR / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # 2. 모델 로드
    print(f"🔍 모델 로딩 중: {FINE_TUNED_MODEL_PATH}")
    model = YOLO(FINE_TUNED_MODEL_PATH)
    
    # 3. 이미지 순회 및 평가
    image_paths = sorted(list(INPUT_IMAGES_DIR.glob("*.jpg")) + list(INPUT_IMAGES_DIR.glob("*.png")))
    
    all_image_results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    print(f"🖼️ 총 {len(image_paths)}개의 이미지에 대해 평가를 진행합니다.")
    # for image_path in tqdm(image_paths, desc="Evaluating Images"):
    #     image = cv2.imread(str(image_path))
    #     h, w, _ = image.shape
        
    #     # 예측 수행
    #     results = model.predict(image, conf=0.25) # conf는 필요에 따라 조절
        
    #     pred_masks = []
    #     if results[0].masks is not None:
    #         # 마스크를 (h, w) 크기의 이진 np.array로 변환
    #         for mask_tensor in results[0].masks.data:
    #             mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    #             # 모델 출력 마스크는 원본 이미지 크기와 다를 수 있으므로 리사이즈
    #             if mask_np.shape[:2] != (h, w):
    #                 mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    #             pred_masks.append(mask_np)

    #     # 실제 정답 로드
    #     gt_masks = load_ground_truth_masks(image_path.stem, h, w)
        
    #     # TP, FP, FN 계산
    #     tp = 0
    #     fp = 0
        
    #     if not gt_masks:
    #         fp = len(pred_masks) # 정답이 없는데 예측만 있으면 모두 FP
    #         fn = 0
    #     elif not pred_masks:
    #         fn = len(gt_masks) # 예측이 없는데 정답만 있으면 모두 FN
    #     else:
    #         gt_matched = [False] * len(gt_masks)
    #         for pred_mask in pred_masks:
    #             best_iou, best_gt_idx = find_best_match(pred_mask, gt_masks)
                
    #             if best_iou > IOU_THRESHOLD and not gt_matched[best_gt_idx]:
    #                 tp += 1
    #                 gt_matched[best_gt_idx] = True
    #             else:
    #                 fp += 1
    #         fn = len(gt_masks) - sum(gt_matched)

    #     # 결과 저장
    #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    #     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
    #     all_image_results.append({
    #         "image": image_path.name,
    #         "tp": tp,
    #         "fp": fp,
    #         "fn": fn,
    #         "precision": precision,
    #         "recall": recall,
    #         "f1_score": f1_score
    #     })
        
    #     total_tp += tp
    #     total_fp += fp
    #     total_fn += fn

    #     # 시각화 결과 저장
    #     visualize_results(image, pred_masks, gt_masks, vis_dir / f"{image_path.stem}_result.jpg")

    val_results = model.val(
        data=VAL_DATA_YAML,
        imgsz=1024,
        conf=0.001,
        iou=0.75,
        device=0,
        split="test",
        save_json=True,
        verbose=True
    )

    # # 4. 최종 결과 계산 및 저장
    # print("\n📊 최종 결과를 계산하고 저장합니다...")
    
    # # Micro-average Precision, Recall, F1
    # micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    # micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    # micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # # Macro-average Precision, Recall, F1
    # macro_precision = np.mean([res['precision'] for res in all_image_results])
    # macro_recall = np.mean([res['recall'] for res in all_image_results])
    # macro_f1 = np.mean([res['f1_score'] for res in all_image_results])
    
    # # AP (Average Precision)는 일반적으로 PR 곡선의 면적으로 계산됩니다.
    # # 여기서는 단순화하여 F1-score와 Precision/Recall을 주요 지표로 사용합니다.
    # # COCO 스타일의 mAP 계산은 더 복잡한 로직이 필요합니다.
    # # 여기서는 Precision/Recall을 기반으로 한 평가 지표를 제공합니다.

    # summary = {
    #     "Configuration": {
    #         "model_path": str(FINE_TUNED_MODEL_PATH),
    #         "iou_threshold": IOU_THRESHOLD,
    #     },
    #     "Overall_Metrics": {
    #         "Micro_Average_Precision": micro_precision,
    #         "Micro_Average_Recall": micro_recall,
    #         "Micro_Average_F1-Score": micro_f1,
    #         "Macro_Average_Precision": macro_precision,
    #         "Macro_Average_Recall": macro_recall,
    #         "Macro_Average_F1-Score": macro_f1,
    #         "total_true_positives": total_tp,
    #         "total_false_positives": total_fp,
    #         "total_false_negatives": total_fn,
    #     },
    #     "Per_Image_Results": all_image_results
    # }
    
    # result_text_path = OUTPUT_DIR / "evaluation_summary.txt"
    # with open(result_text_path, 'w', encoding='utf-8') as f:
    #     json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 평가 완료!")
    # print(f"  - 시각화 결과 저장 위치: {vis_dir}")
    # print(f"  - 상세 결과 파일: {result_text_path}")
    # print("\n--- 최종 요약 ---")
    # print(f"  - Precision (Macro): {macro_precision:.4f}")
    # print(f"  - Recall (Macro): {macro_recall:.4f}")
    # print(f"  - F1-Score (Macro): {macro_f1:.4f}")
    # print("------------------")
    # print("\n📊 YOLO Validation Results")
    print(f"Precision: {val_results.seg.p}")
    print(f"Recall   : {val_results.seg.r}")
    print(f"F1 Score : {val_results.seg.f1}")
    print(f"mAP50-95: {val_results.seg.map}")
    print(f"mAP50    : {val_results.seg.map50}")
    print(f"mAP75    : {val_results.seg.map75}")

if __name__ == "__main__":
    main()
