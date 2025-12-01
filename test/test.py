import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import SAM

# Add parent directory to sys.path for imports from the main project
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from composite import composite_on_segment, compute_segment_averaged_depth
from depth import compute_depth
from semantic_matcher import SemanticMatcher
from utils import get_all_objects


# Project directories (relative to project root, not test folder)
BASE_DIR = PROJECT_ROOT
DATASET_DIR = BASE_DIR / "dataset" / "train"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output" / "dataset"
MASKED_FRAMES_DIR = OUTPUT_DIR / "masked_frames"

# Default video path (in test folder)
DEFAULT_VIDEO_PATH = TEST_DIR / "monkey.mp4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASKED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

SAM_MODEL = None


def get_sam_model():
    """Load and cache the SAM model once."""
    global SAM_MODEL
    if SAM_MODEL is None:
        print("Loading SAM model...")
        SAM_MODEL = SAM("sam2_l.pt")
        print("SAM model loaded.")
    return SAM_MODEL


def frame_from_video(video_path: Path, output_dir: Path) -> None:
    """Extract unique frames from a video using frame content hashing."""
    print(f"\n[Frame Extraction] Extracting unique frames from video: {video_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    saved_idx = 0
    seen_hashes = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        if frame_hash in seen_hashes:
            frame_idx += 1
            continue

        seen_hashes.add(frame_hash)
        frame_filename = output_dir / f"frame_{saved_idx:04d}.png"
        cv2.imwrite(str(frame_filename), frame)
        print(f"  Saved unique frame: {frame_filename.name}")
        frame_idx += 1
        saved_idx += 1

    cap.release()
    print(f"Extracted {saved_idx} unique frames to {output_dir}")


def segment_objects(frame_dir: Path) -> None:
    """Segment the largest object in each frame and save RGBA masks."""
    print(f"\n[Step 3] Segmenting objects from frames in {frame_dir} -> {MASKED_FRAMES_DIR}...")

    model = get_sam_model()
    frames = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    if not frames:
        print(f"  Warning: No frames found in {frame_dir}")
        return

    for idx, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  [WARN] Cannot read {frame_path.name}, skipping.")
            continue

        pred_results = model.predict(img, task="segment")

        if not pred_results or not hasattr(pred_results[0], "masks") or pred_results[0].masks is None:
            print(f"  [WARN] No mask detected in {frame_path.name}, skipping.")
            continue

        largest_mask = None
        largest_area = 0
        for mask in pred_results[0].masks.data:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            area = mask_np.sum()
            if area > largest_area:
                largest_area = area
                largest_mask = mask_np

        if largest_mask is None:
            print(f"  [WARN] No mask data found in {frame_path.name}, skipping.")
            continue

        largest_mask = 1 - largest_mask
        kernel = np.ones((5, 5), np.uint8)
        largest_mask = cv2.erode(largest_mask, kernel, iterations=1)

        obj_only = np.zeros_like(img)
        obj_only[largest_mask == 1] = img[largest_mask == 1]
        alpha = (largest_mask * 255).astype(np.uint8)
        rgba_cv = cv2.cvtColor(obj_only, cv2.COLOR_BGR2BGRA)
        rgba_cv[:, :, 3] = alpha

        rgba_pil = Image.fromarray(cv2.cvtColor(rgba_cv, cv2.COLOR_BGRA2RGBA))

        out_path = MASKED_FRAMES_DIR / f"{frame_path.stem}_masked.png"
        rgba_pil.save(out_path)
        print(f"  Segmented and saved: {out_path.name}")

    print(f"Completed segmentation of {len(frames)} frames.")


def find_suitable_backgrounds(
    object_category: str,
    semantic_locations: List[str],
    broad_categories: List[str] = None,
    max_backgrounds: int = 5,
    similarity_threshold: float = 0.8,
    max_workers: int = 5,
) -> List[Dict]:
    """Find semantically suitable backgrounds using SemanticMatcher."""
    print(f"\n[Step 4] Finding semantically suitable backgrounds...")
    if not DATASET_DIR.exists():
        print(f"  Dataset directory not found: {DATASET_DIR}")
        print("  Please place background images under this path (e.g., dataset/train/...) and retry.")
        return []
    matcher = SemanticMatcher(similarity_threshold=similarity_threshold)

    suitable_backgrounds = matcher.find_suitable_backgrounds(
        semantic_locations=semantic_locations,
        dataset_dir=DATASET_DIR,
        max_backgrounds=max_backgrounds,
        max_workers=max_workers,
        broad_categories=broad_categories,
    )

    matcher.embedding_manager.save_label_cache()
    print(f"Found {len(suitable_backgrounds)} suitable backgrounds")
    return suitable_backgrounds


def blend_object(obj_image: Image.Image, bg_image: Image.Image) -> Image.Image:
    """
    Perform Poisson (seamless) blending of an object image onto a background,
    and return the blended result (full composite). The object is centered on
    the background. If the object has an alpha channel, that is used as mask;
    otherwise a non-zero RGB mask is inferred.
    """
    # Convert PIL to OpenCV format
    obj_rgba = obj_image.convert("RGBA")
    bg_rgba = bg_image.convert("RGBA")

    obj_cv = cv2.cvtColor(np.array(obj_rgba), cv2.COLOR_RGBA2BGRA)
    bg_cv = cv2.cvtColor(np.array(bg_rgba), cv2.COLOR_RGBA2BGRA)

    # Resize object only if larger than background
    bg_h, bg_w = bg_cv.shape[:2]
    obj_h, obj_w = obj_cv.shape[:2]
    if obj_w > bg_w or obj_h > bg_h:
        scale = min(bg_w / obj_w, bg_h / obj_h) * 0.9
        obj_cv = cv2.resize(obj_cv, (int(obj_w * scale), int(obj_h * scale)), interpolation=cv2.INTER_AREA)
        obj_h, obj_w = obj_cv.shape[:2]

    # Mask: alpha if present, else non-zero RGB
    if obj_cv.shape[2] == 4:
        mask = obj_cv[:, :, 3]
    else:
        mask = cv2.cvtColor(obj_cv, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Position: mask centroid if available, else center of background
    M = cv2.moments(mask_bin)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (cX, cY)
    else:
        center = (bg_w // 2, bg_h // 2)

    # Seamless cloning
    blended = cv2.seamlessClone(
        obj_cv[:, :, :3],           # src BGR
        bg_cv[:, :, :3],            # dst BGR
        mask_bin,                   # mask uint8
        center,                     # center point
        cv2.MIXED_CLONE
    )

    # Convert back to PIL (RGB)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))


def composite_naturally(
    objects: List[Tuple[Image.Image, Dict]],
    backgrounds: List[Dict],
    use_depth: bool = False,
    use_lighting: bool = False,
    overlay_scale: float = 1.0,
    depth_offset: float = 0.05,
) -> List[Path]:
    """Composite objects onto backgrounds with optional depth handling."""
    print(f"\n[Step 5] Compositing objects naturally...")
    print(f"  Use depth: {use_depth}")
    print(f"  Use lighting: {use_lighting}")
    if use_lighting:
        print("  [TODO] Lighting adjustment not implemented yet")

    output_paths = []

    for bg_idx, bg_info in enumerate(backgrounds):
        bg_depth_map = None
        averaged_depth_map = None
        if use_depth:
            print(f"\n  Computing depth for background {bg_idx}...")
            _, bg_depth_map = compute_depth(bg_info["bg_image"])
            print(f"    Depth range: [{bg_depth_map.min():.2f}, {bg_depth_map.max():.2f}]")

            from segment import run_segmentation

            annotations, _ = run_segmentation(bg_info["bg_image"])
            segments = [{"segmentation": mask} for mask, label in annotations]
            averaged_depth_map = compute_segment_averaged_depth(bg_depth_map, segments)

        for obj_idx, (obj_image, obj_meta) in enumerate(objects):
            try:
                composite_img = composite_on_segment(
                    bg_info["bg_image"],
                    bg_info["segment_mask"],
                    obj_image,
                    base_scale=overlay_scale,
                    use_depth=use_depth,
                    bg_depth_map=averaged_depth_map if use_depth else None,
                    depth_offset=depth_offset,
                )

                # Derive object cutout from composite vs background for Poisson blending
                bg_rgb = bg_info["bg_image"].convert("RGB")
                comp_rgb = composite_img.convert("RGB")

                bg_cv = cv2.cvtColor(np.array(bg_rgb), cv2.COLOR_RGB2BGR)
                comp_cv = cv2.cvtColor(np.array(comp_rgb), cv2.COLOR_RGB2BGR)

                diff_gray = cv2.cvtColor(cv2.absdiff(comp_cv, bg_cv), cv2.COLOR_BGR2GRAY)
                _, obj_mask = cv2.threshold(diff_gray, 1, 255, cv2.THRESH_BINARY)

                if obj_mask.max() == 0:
                    blended_img = composite_img  # fallback if mask is empty
                else:
                    obj_only = cv2.bitwise_and(comp_cv, comp_cv, mask=obj_mask)
                    obj_only_rgba = cv2.cvtColor(obj_only, cv2.COLOR_BGR2BGRA)
                    obj_only_rgba[:, :, 3] = obj_mask
                    obj_cutout = Image.fromarray(cv2.cvtColor(obj_only_rgba, cv2.COLOR_BGRA2RGBA))
                    blended_img = blend_object(obj_cutout, bg_info["bg_image"])

                depth_suffix = "_depth" if use_depth else ""
                output_filename = (
                    f"composite_bg{bg_idx}_obj{obj_idx}_"
                    f"{bg_info['segment_label'].replace(' ', '_')}{depth_suffix}.png"
                )
                output_path = OUTPUT_DIR / output_filename
                blended_img.save(output_path)
                output_paths.append(output_path)

                print(f"  Saved: {output_filename}")

            except Exception as e:
                print(f"  Error: bg{bg_idx} + obj{obj_idx}: {e}")
                import traceback

                traceback.print_exc()
                continue

    print(f"Generated {len(output_paths)} composite images")
    return output_paths


def run_test_from_video(video_path: Optional[Path] = None):
    """
    Run the pipeline starting from an existing video.

    Args:
        video_path: Path to the video file. If None, uses the default monkey.mp4.
    """
    if video_path is None:
        video_path = DEFAULT_VIDEO_PATH
    else:
        video_path = Path(video_path).resolve()

    existing_masks = sorted(MASKED_FRAMES_DIR.glob("*.png"))
    if existing_masks:
        print(f"Found {len(existing_masks)} masked frame(s). Skipping video extraction and segmentation.")
    else:
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return

        # Use video filename (without extension) as the frames directory name
        video_name = video_path.stem
        frames_dir = OUTPUT_DIR / f"{video_name}_frames"
        frame_from_video(video_path, frames_dir)
        segment_objects(frames_dir)

    backgrounds = find_suitable_backgrounds(
        object_category="monkey doll",
        semantic_locations=["shelf", "bed", "couch", "table", "toy box"],
        broad_categories=["home", "indoor", "living room", "bedroom", "house interior"],
        max_backgrounds=5,
        similarity_threshold=0.8,
        max_workers=5,
    )

    if len(backgrounds) == 0:
        print("No suitable backgrounds found.")
        return

    objects = get_all_objects(
        original_input_dir=INPUT_DIR,
        masked_frames_dir=MASKED_FRAMES_DIR,
    )

    if len(objects) == 0:
        print("No objects loaded. Check masked frames or input images.")
        return

    outputs = composite_naturally(
        objects=objects,
        backgrounds=backgrounds,
        use_depth=True,
        use_lighting=False,
        overlay_scale=0.8,
        depth_offset=0.05,
    )

    if outputs:
        print("\nGenerated images:")
        for p in outputs:
            print(f"  - {p}")
    else:
        print("\nNo composite images were generated.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run image composition test pipeline from a video file."
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help=f"Path to the video file. Defaults to {DEFAULT_VIDEO_PATH}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = Path(args.video) if args.video else None
    run_test_from_video(video_path=video_path)
