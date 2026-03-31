"""YOLOv8n fine-tuning pipeline with W&B experiment tracking.

Wraps the Ultralytics training API with sensible defaults for this project.
All hyperparameters are exposed as CLI arguments so nothing is hard-coded.

W&B logging is automatic — Ultralytics calls wandb.log() internally when
WANDB_PROJECT is set or when you pass project/name to model.train().

Usage:
    python -m oakd_vision.detector.yolo_trainer \\
        --data data/dataset/dataset.yaml \\
        --epochs 200 \\
        --project oakd-vision-ml \\
        --run-name yolov8n-custom-v1

After training, best weights are at:
    runs/detect/<run-name>/weights/best.pt
    runs/detect/<run-name>/weights/last.pt
"""

import argparse
from pathlib import Path

import wandb
from ultralytics import YOLO


def train(
    data: str,
    epochs: int = 200,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 0.01,
    lrf: float = 0.01,
    patience: int = 50,
    project: str = "oakd-vision-ml",
    run_name: str = "yolov8n-custom",
    weights: str = "yolov8n.pt",
    device: str = "",
) -> Path:
    """Fine-tune YOLOv8n on the custom dataset.

    Args:
        data: path to dataset.yaml
        epochs: total training epochs
        imgsz: square input resolution (pixels)
        batch: batch size (-1 = auto)
        lr0: initial learning rate
        lrf: final learning rate as a fraction of lr0
        patience: early-stopping patience (epochs without improvement)
        project: W&B / Ultralytics project name
        run_name: experiment name (creates runs/detect/<run_name>/)
        weights: starting checkpoint — "yolov8n.pt" downloads COCO pretrained
        device: "" = auto, "0" = GPU 0, "cpu" = force CPU

    Returns:
        Path to the best.pt weights file.
    """
    model = YOLO(weights)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        patience=patience,
        project=project,
        name=run_name,
        device=device,
        # Augmentation — standard for fine-tuning on a small dataset
        hsv_h=0.015,    # hue jitter
        hsv_s=0.7,      # saturation jitter
        hsv_v=0.4,      # value jitter
        flipud=0.0,     # no vertical flip (robot camera is fixed)
        fliplr=0.5,
        mosaic=1.0,     # mosaic augmentation (4-image)
        mixup=0.1,
        degrees=5.0,    # small rotation (camera tilts vary a little)
        translate=0.1,
        scale=0.5,
        # Logging
        save=True,
        save_period=20,
        plots=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_pt}")
    return best_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on custom dataset")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--project", default="oakd-vision-ml")
    parser.add_argument("--run-name", default="yolov8n-custom")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        run_name=args.run_name,
        weights=args.weights,
        device=args.device,
    )
