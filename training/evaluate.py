"""
Model Evaluation Module for MineGuard AI.

Evaluates the trained YOLO11n PPE detector on the held-out TEST split.
Calculates Precision, Recall, F1-score, mAP50, mAP50-95, per-class performance metrics,
confusion matrix, and saves all evaluation plots, per_class_metrics.csv, and evaluation_report.txt under
outputs/training/evaluation/.
"""

import sys
import shutil
import argparse
import csv
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO11n model on the MineGuard test dataset split."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.TRAINED_MODEL_PATH),
        help=f"Path to model weights to evaluate (default: {config.TRAINED_MODEL_PATH})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(config.DATASET_PATH),
        help=f"Path to dataset yaml configuration (default: {config.DATASET_PATH})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.IMAGE_SIZE,
        help=f"Evaluation image size (default: {config.IMAGE_SIZE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Evaluation batch size (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help=f"Device to run evaluation on (default: {config.DEVICE})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=config.IOU_THRESHOLD,
        help=f"NMS IoU threshold (default: {config.IOU_THRESHOLD})",
    )
    return parser.parse_args()


def calculate_f1(precision: float, recall: float) -> float:
    """Calculates harmonic mean F1-score from precision and recall."""
    if (precision + recall) <= 1e-8:
        return 0.0
    return float(2.0 * (precision * recall) / (precision + recall))


def evaluate_model(args):
    """Executes validation on test split and outputs a comprehensive evaluation report."""
    model_path = Path(args.model)
    data_path = Path(args.data)
    eval_output_dir = config.EVALUATION_OUTPUTS_DIR
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("               MINEGUARD AI - MODEL TEST EVALUATION")
    print("=" * 70)
    print(f"Target Model Weights    : {model_path}")
    print(f"Dataset Configuration   : {data_path}")
    print(f"Evaluation Split        : TEST")
    print(f"Device                  : {args.device}")
    print(f"Confidence Threshold    : {args.conf}")
    print(f"IoU Threshold           : {args.iou}")
    print(f"Report Output Directory : {eval_output_dir}")
    print("=" * 70)

    if not model_path.exists():
        if model_path == config.TRAINED_MODEL_PATH and config.PRETRAINED_MODEL_PATH.exists():
            print(f"\n[NOTICE] Trained model weights '{model_path}' not found yet.")
            print(f"Falling back to pretrained baseline weights at: {config.PRETRAINED_MODEL_PATH}")
            model_path = config.PRETRAINED_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. "
                "Please check the path or train the model first using 'python training/train.py'."
            )

    # Load YOLO model
    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))

    # Run evaluation on test split
    print("\nRunning inference and metrics computation on TEST split...")
    results = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(config.TRAINING_OUTPUTS_DIR),
        name="evaluation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # Extract metrics safely
    box = results.box
    names = results.names

    overall_precision = float(box.mp) if hasattr(box, "mp") else 0.0
    overall_recall = float(box.mr) if hasattr(box, "mr") else 0.0
    overall_f1 = calculate_f1(overall_precision, overall_recall)
    overall_map50 = float(box.map50) if hasattr(box, "map50") else 0.0
    overall_map50_95 = float(box.map) if hasattr(box, "map") else 0.0

    # Per-class metrics
    per_class_p = box.p if hasattr(box, "p") and box.p is not None else []
    per_class_r = box.r if hasattr(box, "r") and box.r is not None else []
    per_class_map50 = box.ap50 if hasattr(box, "ap50") and box.ap50 is not None else []
    per_class_map50_95 = box.ap if hasattr(box, "ap") and box.ap is not None else []

    # Speed metrics
    speed = results.speed if hasattr(results, "speed") else {}
    preprocess_ms = speed.get("preprocess", 0.0)
    inference_ms = speed.get("inference", 0.0)
    postprocess_ms = speed.get("postprocess", 0.0)
    total_latency_ms = preprocess_ms + inference_ms + postprocess_ms
    eval_fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

    # Save per_class_metrics.csv
    csv_file = eval_output_dir / "per_class_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1_score", "map50", "map50_95"])
        for idx, (cls_id, cls_name) in enumerate(names.items()):
            p_val = float(per_class_p[idx]) if idx < len(per_class_p) else 0.0
            r_val = float(per_class_r[idx]) if idx < len(per_class_r) else 0.0
            f1_val = calculate_f1(p_val, r_val)
            map50_val = float(per_class_map50[idx]) if idx < len(per_class_map50) else 0.0
            map50_95_val = float(per_class_map50_95[idx]) if idx < len(per_class_map50_95) else 0.0
            writer.writerow([cls_id, cls_name, f"{p_val:.4f}", f"{r_val:.4f}", f"{f1_val:.4f}", f"{map50_val:.4f}", f"{map50_95_val:.4f}"])
        writer.writerow(["ALL", "Average / Mean", f"{overall_precision:.4f}", f"{overall_recall:.4f}", f"{overall_f1:.4f}", f"{overall_map50:.4f}", f"{overall_map50_95:.4f}"])

    print(f"Saved per-class CSV metrics to: {csv_file}")

    # Ensure results.png exists in evaluation directory
    training_results_png = config.TRAINING_OUTPUTS_DIR / "yolo11n_mineguard" / "results.png"
    eval_results_png = eval_output_dir / "results.png"
    if training_results_png.exists() and not eval_results_png.exists():
        shutil.copy2(training_results_png, eval_results_png)

    # Format text report
    lines = [
        "=" * 70,
        "              MINEGUARD AI - MODEL TEST EVALUATION REPORT",
        "=" * 70,
        f"Evaluated Model       : {model_path}",
        f"Dataset Split         : TEST",
        f"Dataset Config        : {data_path}",
        f"Device Used           : {args.device}",
        f"Image Dimension       : {args.imgsz}",
        f"Confidence Threshold  : {args.conf}",
        f"IoU Threshold         : {args.iou}",
        "",
        "----------------------------------------------------------------------",
        "INFERENCE SPEED & LATENCY (MEASURED)",
        "----------------------------------------------------------------------",
        f"Preprocess Latency    : {preprocess_ms:.2f} ms",
        f"Inference Latency     : {inference_ms:.2f} ms",
        f"Postprocess Latency   : {postprocess_ms:.2f} ms",
        f"Total End-to-End Latency: {total_latency_ms:.2f} ms",
        f"Evaluation Throughput : {eval_fps:.1f} FPS",
        "",
        "----------------------------------------------------------------------",
        "OVERALL PERFORMANCE METRICS",
        "----------------------------------------------------------------------",
        f"Precision (mean)      : {overall_precision:.4f} ({overall_precision * 100:.2f}%)",
        f"Recall (mean)         : {overall_recall:.4f} ({overall_recall * 100:.2f}%)",
        f"F1-Score (mean)       : {overall_f1:.4f} ({overall_f1 * 100:.2f}%)",
        f"mAP@0.5               : {overall_map50:.4f} ({overall_map50 * 100:.2f}%)",
        f"mAP@0.5:0.95          : {overall_map50_95:.4f} ({overall_map50_95 * 100:.2f}%)",
        "",
        "----------------------------------------------------------------------",
        "PER-CLASS PERFORMANCE BREAKDOWN",
        "----------------------------------------------------------------------",
        f"{'Class ID':<10} {'Class Name':<16} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'mAP50':<11} {'mAP50-95':<11}",
        "-" * 70,
    ]

    for idx, (cls_id, cls_name) in enumerate(names.items()):
        p_val = float(per_class_p[idx]) if idx < len(per_class_p) else 0.0
        r_val = float(per_class_r[idx]) if idx < len(per_class_r) else 0.0
        f1_val = calculate_f1(p_val, r_val)
        map50_val = float(per_class_map50[idx]) if idx < len(per_class_map50) else 0.0
        map50_95_val = float(per_class_map50_95[idx]) if idx < len(per_class_map50_95) else 0.0

        lines.append(
            f"{cls_id:<10} {cls_name:<16} {p_val:<11.4f} {r_val:<11.4f} {f1_val:<11.4f} {map50_val:<11.4f} {map50_95_val:<11.4f}"
        )

    lines.extend([
        "-" * 70,
        f"{'ALL':<10} {'Average / Mean':<16} {overall_precision:<11.4f} {overall_recall:<11.4f} {overall_f1:<11.4f} {overall_map50:<11.4f} {overall_map50_95:<11.4f}",
        "",
        "----------------------------------------------------------------------",
        "CONFUSION MATRIX & ARTIFACTS",
        "----------------------------------------------------------------------",
    ])

    if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
        matrix = results.confusion_matrix.matrix
        lines.append("Confusion Matrix Array (Rows=True, Cols=Predicted):")
        lines.append(str(matrix))
        lines.append("")

    lines.extend([
        f"Saved Report          : {eval_output_dir / 'evaluation_report.txt'}",
        f"Saved CSV Breakdown   : {csv_file}",
        f"Saved Plots & Charts  : {eval_output_dir}",
        "Artifacts Generated   : confusion_matrix.png, BoxPR_curve.png, BoxF1_curve.png, results.png, per_class_metrics.csv",
        "",
        "----------------------------------------------------------------------",
        "DOMAIN LIMITATION NOTICE",
        "----------------------------------------------------------------------",
        "Note: The evaluation dataset is primarily construction-related imagery.",
        "Underground coal mines introduce severe low-light, dust, and non-uniform",
        "shadowing conditions. Field deployment requires checkpoint validation.",
        "=" * 70,
    ])

    report_text = "\n".join(lines)
    report_file = eval_output_dir / "evaluation_report.txt"
    report_file.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\nReport successfully saved to: {report_file}")
    return results


if __name__ == "__main__":
    cli_args = parse_args()
    evaluate_model(cli_args)
