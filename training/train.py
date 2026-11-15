"""
YOLO11n Training Pipeline for MineGuard AI.

Implements transfer learning on the Atharion underground PPE dataset.
Supports CLI overrides, early stopping, automatic CUDA/CPU selection,
graceful CUDA Out-Of-Memory (OOM) error handling, and model deployment
to models/trained/best.pt.
"""

import sys
import shutil
import argparse
from pathlib import Path
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as config
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n PPE detector on MineGuard dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.PRETRAINED_MODEL_PATH),
        help=f"Path to pretrained weights or model yaml (default: {config.PRETRAINED_MODEL_PATH})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(config.DATASET_PATH),
        help=f"Path to dataset yaml configuration (default: {config.DATASET_PATH})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Number of training epochs (default: {config.EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Training batch size (default: {config.BATCH_SIZE})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.IMAGE_SIZE,
        help=f"Image dimension (default: {config.IMAGE_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help=f"Computation device: 'cuda:0', '0', 'cpu' (default: {config.DEVICE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.MAX_WORKERS,
        help=f"Dataloader worker processes (default: {config.MAX_WORKERS})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=config.EARLY_STOPPING_PATIENCE,
        help=f"Early stopping patience in epochs (default: {config.EARLY_STOPPING_PATIENCE})",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(config.TRAINING_OUTPUTS_DIR),
        help=f"Directory to store training runs (default: {config.TRAINING_OUTPUTS_DIR})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11n_mineguard",
        help="Run name within the project directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the most recent checkpoint",
    )
    return parser.parse_args()


def train_model(args):
    """Executes YOLO11n fine-tuning with full diagnostics and error handling."""
    model_path = Path(args.model)
    data_path = Path(args.data)
    project_dir = Path(args.project)

    print("=" * 70)
    print("               MINEGUARD AI - YOLO11N TRAINING")
    print("=" * 70)
    print(f"Pretrained Model Source : {model_path}")
    print(f"Dataset Configuration   : {data_path}")
    print(f"Hardware Device Target  : {args.device}")
    print(f"Epochs Configured       : {args.epochs}")
    print(f"Batch Size              : {args.batch_size}")
    print(f"Image Resolution (imgsz): {args.imgsz}")
    print(f"Dataloader Workers      : {args.workers}")
    print(f"Early Stopping Patience : {args.patience}")
    print(f"Outputs Destination     : {project_dir / args.name}")
    print("=" * 70)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights file not found at: {model_path}. "
            f"Please verify {config.PRETRAINED_MODEL_PATH} exists."
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found at: {data_path}. "
            "Please ensure training/data.yaml is generated."
        )

    # Load YOLO architecture with pretrained weights
    print(f"\n[1/3] Loading YOLO architecture from: {model_path}")
    model = YOLO(str(model_path))

    # Clear CUDA cache prior to training
    if "cuda" in str(args.device) and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[2/3] Initiating training run...")
    try:
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=str(project_dir),
            name=args.name,
            exist_ok=True,
            save=True,
            save_period=-1,
            plots=True,
            verbose=True,
            resume=args.resume,
        )
    except torch.cuda.OutOfMemoryError as oom_err:
        print("\n" + "!" * 70)
        print("ERROR: CUDA OUT-OF-MEMORY (OOM) ENCOUNTERED!")
        print("!" * 70)
        print("The GPU ran out of video memory with the current batch size and image size.")
        print(f"Current Batch Size: {args.batch_size} | Image Size: {args.imgsz}")
        print("\nRECOMMENDED RESOLUTION:")
        print("Do NOT silently switch to CPU (which would drastically degrade training speed).")
        print("Rerun training with a reduced batch size:")
        print(f"    python training/train.py --batch-size {max(1, args.batch_size // 2)}")
        print("Or if necessary, reduce image resolution:")
        print(f"    python training/train.py --batch-size {max(1, args.batch_size // 2)} --imgsz 512")
        print("!" * 70)
        sys.exit(1)
    except RuntimeError as r_err:
        err_msg = str(r_err)
        if "out of memory" in err_msg.lower():
            print("\n" + "!" * 70)
            print("ERROR: CUDA OUT-OF-MEMORY (OOM) DETECTED!")
            print("!" * 70)
            print(f"Details: {err_msg}")
            print(f"Current Batch Size: {args.batch_size} | Image Size: {args.imgsz}")
            print("RECOMMENDED RESOLUTION: Rerun with smaller batch size:")
            print(f"    python training/train.py --batch-size {max(1, args.batch_size // 2)}")
            print("!" * 70)
            sys.exit(1)
        else:
            raise r_err

    # [3/3] Deployment of best model weights
    run_dir = project_dir / args.name
    best_weights_src = run_dir / "weights" / "best.pt"

    if best_weights_src.exists():
        config.TRAINED_DIR.mkdir(parents=True, exist_ok=True)
        dest_model_path = config.TRAINED_MODEL_PATH
        shutil.copy2(best_weights_src, dest_model_path)

        print("\n" + "=" * 70)
        print("                  TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Best Weights Generated     : {best_weights_src}")
        print(f"Deployed Model Target      : {dest_model_path}")
        print(f"Pretrained Weights Preserved: {model_path} (Untouched)")
        print(f"Training Artifacts & Plots : {run_dir}")
        print("=" * 70)
    else:
        print("\nWARNING: Training completed but best.pt was not located in run weights directory.")
        print(f"Looked for: {best_weights_src}")

    return results


if __name__ == "__main__":
    cli_args = parse_args()
    train_model(cli_args)
