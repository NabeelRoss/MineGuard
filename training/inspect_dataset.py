"""
Dataset Inspection and Validation Module for MineGuard AI.

Validates the dataset integrity, verifies annotation syntax and bounding box coordinates,
computes class distributions across splits, and generates sample visualizations.
Outputs report to outputs/dataset_reports/dataset_report.txt and
sample montage to outputs/dataset_reports/annotated_samples.jpg.
"""

import sys
from pathlib import Path
import yaml
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config


def load_dataset_config(yaml_path: Path) -> dict:
    """Loads and validates a YOLO data.yaml configuration file."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required_keys = ["names"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing required key '{k}' in {yaml_path}")

    # Normalize names to dictionary format {id: name}
    names = data["names"]
    if isinstance(names, list):
        data["names_dict"] = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        data["names_dict"] = {int(k): str(v) for k, v in names.items()}
    else:
        raise TypeError(f"Unsupported format for 'names' in {yaml_path}: {type(names)}")

    return data


def validate_yolo_bbox(
    cls_id_str: str,
    x_c_str: str,
    y_c_str: str,
    w_str: str,
    h_str: str,
    num_classes: int,
) -> tuple[bool, str, int, list[float]]:
    """
    Validates a single YOLO format bounding box row.
    Format: <class_id> <x_center> <y_center> <width> <height>
    """
    try:
        cls_id = int(cls_id_str)
        x_c = float(x_c_str)
        y_c = float(y_c_str)
        w = float(w_str)
        h = float(h_str)
    except ValueError as e:
        return False, f"Non-numeric values in line: {e}", -1, []

    if cls_id < 0 or cls_id >= num_classes:
        return False, f"Class ID {cls_id} out of bounds [0, {num_classes - 1}]", cls_id, []

    # Bounding box coordinate sanity checks (allowing minor float edge tolerance)
    if not (0.0 <= x_c <= 1.0) or not (0.0 <= y_c <= 1.0):
        return False, f"Center coordinates out of bounds: x={x_c:.4f}, y={y_c:.4f}", cls_id, []

    if w <= 0.0 or h <= 0.0 or w > 1.05 or h > 1.05:
        return False, f"Invalid dimensions: w={w:.4f}, h={h:.4f}", cls_id, []

    return True, "Valid", cls_id, [x_c, y_c, w, h]


def inspect_split(split_name: str, split_dir: Path, num_classes: int, class_names: dict) -> dict:
    """Inspects an individual dataset split (images and corresponding annotations)."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]) if images_dir.exists() else []
    label_files = sorted([f for f in labels_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]) if labels_dir.exists() else []

    stats = {
        "split": split_name,
        "split_dir": str(split_dir),
        "total_images": len(image_files),
        "total_labels": len(label_files),
        "missing_labels": [],       # Images without label file (background images)
        "empty_labels": [],         # Label files with 0 objects (background)
        "corrupted_images": [],     # Unreadable image files
        "invalid_bboxes": [],       # Files with coordinate/class errors
        "valid_objects_count": 0,
        "class_counts": {cls_id: 0 for cls_id in class_names.keys()},
        "image_shapes": {},
        "valid_samples": [],        # Cache for visual sampling
    }

    labels_map = {f.stem: f for f in label_files}

    for img_path in image_files:
        # 1. Image integrity check
        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            stats["corrupted_images"].append(img_path.name)
            continue

        h, w = img.shape[:2]
        shape_key = f"{w}x{h}"
        stats["image_shapes"][shape_key] = stats["image_shapes"].get(shape_key, 0) + 1

        # 2. Check correspondence
        lbl_path = labels_map.get(img_path.stem)
        if lbl_path is None:
            stats["missing_labels"].append(img_path.name)
            continue

        # 3. Read annotation file
        try:
            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = [line.strip() for line in lf.readlines() if line.strip()]
        except Exception as e:
            stats["invalid_bboxes"].append((lbl_path.name, f"Read error: {e}"))
            continue

        if len(lines) == 0:
            stats["empty_labels"].append(lbl_path.name)
            continue

        parsed_boxes = []
        for line_idx, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                stats["invalid_bboxes"].append(
                    (lbl_path.name, f"Line {line_idx}: expected 5 tokens, got {len(parts)}")
                )
                continue

            valid, msg, cls_id, bbox = validate_yolo_bbox(*parts, num_classes=num_classes)
            if not valid:
                stats["invalid_bboxes"].append((lbl_path.name, f"Line {line_idx}: {msg}"))
            else:
                stats["valid_objects_count"] += 1
                stats["class_counts"][cls_id] = stats["class_counts"].get(cls_id, 0) + 1
                parsed_boxes.append((cls_id, bbox))

        if len(parsed_boxes) > 0 and len(stats["valid_samples"]) < 12:
            stats["valid_samples"].append({
                "image_path": img_path,
                "boxes": parsed_boxes,
            })

    return stats


def generate_annotated_sample_montage(samples: list, class_names: dict, output_path: Path):
    """Draws bounding boxes on selected sample images and stitches them into a grid image."""
    if not samples:
        return

    # Palette for classes: 0: Person (Cyan), 1: Hardhat (Yellow), 2: Safety Vest (Orange)
    colors = {
        0: (255, 200, 0),    # Cyan/Sky blue in BGR
        1: (0, 230, 255),    # Yellow in BGR
        2: (50, 120, 255),   # Orange in BGR
    }

    target_w, target_h = 480, 480
    rendered_images = []

    for item in samples[:6]:
        img = cv2.imread(str(item["image_path"]))
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        canvas = cv2.resize(img, (target_w, target_h))
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        for cls_id, (xc, yc, bw, bh) in item["boxes"]:
            # Convert normalized YOLO bbox to absolute coords on canvas
            abs_xc = xc * orig_w * scale_x
            abs_yc = yc * orig_h * scale_y
            abs_w = bw * orig_w * scale_x
            abs_h = bh * orig_h * scale_y

            x1 = int(max(0, abs_xc - abs_w / 2))
            y1 = int(max(0, abs_yc - abs_h / 2))
            x2 = int(min(target_w - 1, abs_xc + abs_w / 2))
            y2 = int(min(target_h - 1, abs_yc + abs_h / 2))

            color = colors.get(cls_id, (0, 255, 0))
            name = class_names.get(cls_id, f"Class {cls_id}")

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label_text = f"{name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (lw, lh), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            cv2.rectangle(canvas, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), color, -1)
            cv2.putText(canvas, label_text, (x1 + 3, y1 - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Split label on top-left of image
        cv2.putText(
            canvas,
            item["image_path"].name,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered_images.append(canvas)

    if not rendered_images:
        return

    # Form a 2-column or 3-column grid
    num_imgs = len(rendered_images)
    cols = 3 if num_imgs >= 3 else num_imgs
    rows = (num_imgs + cols - 1) // cols

    # Pad with blank images if necessary
    while len(rendered_images) < rows * cols:
        rendered_images.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    row_blocks = []
    for r in range(rows):
        row_imgs = rendered_images[r * cols : (r + 1) * cols]
        row_blocks.append(np.hstack(row_imgs))

    montage = np.vstack(row_blocks)
    cv2.imwrite(str(output_path), montage)


def run_inspection():
    """Main execution function for dataset validation and reporting."""
    print("=" * 70)
    print("           MINEGUARD AI - DATASET INSPECTION & VALIDATION")
    print("=" * 70)

    raw_yaml_path = config.RAW_DATA_YAML
    print(f"Reading dataset configuration from: {raw_yaml_path}")
    data_cfg = load_dataset_config(raw_yaml_path)

    class_names = data_cfg["names_dict"]
    num_classes = len(class_names)
    print(f"Verified {num_classes} classes:")
    for cid, cname in class_names.items():
        print(f"  [ID {cid}] -> {cname}")

    splits = {
        "train": config.DATASET_ROOT / "train",
        "valid": config.DATASET_ROOT / "valid",
        "test": config.DATASET_ROOT / "test",
    }

    all_stats = []
    total_all_images = 0
    total_all_objects = 0
    total_all_corrupted = 0
    all_samples = []

    for split_name, split_path in splits.items():
        print(f"\nScanning split: '{split_name}' at {split_path}...")
        stats = inspect_split(split_name, split_path, num_classes, class_names)
        all_stats.append(stats)
        total_all_images += stats["total_images"]
        total_all_objects += stats["valid_objects_count"]
        total_all_corrupted += len(stats["corrupted_images"])
        all_samples.extend(stats["valid_samples"])

        print(f"  Images: {stats['total_images']}")
        print(f"  Labels: {stats['total_labels']}")
        print(f"  Objects: {stats['valid_objects_count']}")
        print(f"  Missing annotations (bg images): {len(stats['missing_labels'])}")
        print(f"  Empty annotations (bg images): {len(stats['empty_labels'])}")
        print(f"  Corrupted images: {len(stats['corrupted_images'])}")
        print(f"  Invalid boxes: {len(stats['invalid_bboxes'])}")

    # Overall class distribution
    combined_class_counts = {cid: 0 for cid in class_names.keys()}
    for s in all_stats:
        for cid, count in s["class_counts"].items():
            combined_class_counts[cid] += count

    # Generate Report Text
    report_lines = [
        "=" * 70,
        "              MINEGUARD AI - DATASET VALIDATION REPORT",
        "=" * 70,
        f"Dataset Root: {config.DATASET_ROOT}",
        f"Classes Defined ({num_classes}): {class_names}",
        "",
        "----------------------------------------------------------------------",
        "SPLIT BREAKDOWN",
        "----------------------------------------------------------------------",
        f"{'Split':<10} {'Images':<8} {'Labels':<8} {'Objects':<10} {'Missing':<10} {'Empty':<8} {'Corrupt':<8} {'Invalid':<8}",
        "-" * 70,
    ]

    for s in all_stats:
        report_lines.append(
            f"{s['split']:<10} {s['total_images']:<8} {s['total_labels']:<8} {s['valid_objects_count']:<10} "
            f"{len(s['missing_labels']):<10} {len(s['empty_labels']):<8} {len(s['corrupted_images']):<8} "
            f"{len(s['invalid_bboxes']):<8}"
        )

    report_lines.extend([
        "-" * 70,
        f"{'TOTAL':<10} {total_all_images:<8} {sum(s['total_labels'] for s in all_stats):<8} "
        f"{total_all_objects:<10} {sum(len(s['missing_labels']) for s in all_stats):<10} "
        f"{sum(len(s['empty_labels']) for s in all_stats):<8} {total_all_corrupted:<8} "
        f"{sum(len(s['invalid_bboxes']) for s in all_stats):<8}",
        "",
        "----------------------------------------------------------------------",
        "CLASS DISTRIBUTION",
        "----------------------------------------------------------------------",
        f"{'Class ID':<10} {'Class Name':<20} {'Instances':<12} {'Percentage':<10}",
        "-" * 70,
    ])

    for cid, cname in class_names.items():
        cnt = combined_class_counts.get(cid, 0)
        pct = (cnt / total_all_objects * 100.0) if total_all_objects > 0 else 0.0
        report_lines.append(f"{cid:<10} {cname:<20} {cnt:<12} {pct:.2f}%")

    report_lines.extend([
        "-" * 70,
        f"{'TOTAL':<10} {'ALL CLASSES':<20} {total_all_objects:<12} 100.00%",
        "",
        "----------------------------------------------------------------------",
        "IMAGE RESOLUTIONS DETECTED",
        "----------------------------------------------------------------------",
    ])

    for s in all_stats:
        report_lines.append(f"Split [{s['split']}]:")
        for res, count in sorted(s["image_shapes"].items(), key=lambda x: x[1], reverse=True)[:5]:
            report_lines.append(f"  {res}: {count} images")

    report_lines.extend([
        "",
        "----------------------------------------------------------------------",
        "INTEGRITY STATUS & REMARKS",
        "----------------------------------------------------------------------",
        f"Corrupted Images Detected: {total_all_corrupted}",
        f"Invalid Bounding Boxes Detected: {sum(len(s['invalid_bboxes']) for s in all_stats)}",
        "Background Images: Allowed by design (unannotated or empty label files).",
        "Integrity Verdict: " + ("PASS - READY FOR TRAINING" if total_all_corrupted == 0 else "FAIL - REQUIRES CORRUPTED IMAGE REMOVAL"),
        "=" * 70,
    ])

    report_content = "\n".join(report_lines)

    report_file = config.DATASET_REPORTS_DIR / "dataset_report.txt"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"\nSaved dataset report to: {report_file}")

    # Generate Visual Montage
    sample_file = config.DATASET_REPORTS_DIR / "annotated_samples.jpg"
    generate_annotated_sample_montage(all_samples, class_names, sample_file)
    print(f"Saved annotated visual montage to: {sample_file}")

    print("\n" + report_content)


if __name__ == "__main__":
    run_inspection()
