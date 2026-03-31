"""Evaluate a trained YOLOv8 model on the test split.

Runs Ultralytics validation to compute mAP@50, mAP@50:95, per-class AP,
precision, recall, and generates confusion matrix + PR curves. Results are
saved to an evaluation_report.md in the same directory as the model.

Usage:
    python -m oakd_vision.detector.evaluate \\
        --model models/best.pt \\
        --data data/dataset/dataset.yaml \\
        --split test
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def evaluate(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.001,  # low conf to capture full PR curve
    iou: float = 0.6,
    device: str = "",
) -> dict:
    """Run validation and write evaluation_report.md.

    Args:
        model_path: path to best.pt
        data_yaml: path to dataset.yaml
        split: "train" | "val" | "test"
        imgsz: inference resolution
        conf: confidence threshold for evaluation
        iou: IoU threshold for NMS during evaluation
        device: "" = auto, "cpu", "0", etc.

    Returns:
        Dict with mAP50, mAP50_95, precision, recall, per_class_ap.
    """
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        plots=True,
        save_json=True,
    )

    results = {
        "mAP50":      round(float(metrics.box.map50), 4),
        "mAP50_95":   round(float(metrics.box.map), 4),
        "precision":  round(float(metrics.box.mp), 4),
        "recall":     round(float(metrics.box.mr), 4),
        "per_class_ap50": {
            name: round(float(ap), 4)
            for name, ap in zip(metrics.names.values(), metrics.box.ap50)
        },
    }

    _write_report(model_path, data_yaml, split, results, metrics)
    _print_summary(results)

    return results


def _write_report(model_path, data_yaml, split, results, metrics):
    report_path = Path(model_path).parent / "evaluation_report.md"
    lines = [
        "# YOLOv8 Evaluation Report\n",
        f"**Model:** `{model_path}`  \n",
        f"**Dataset:** `{data_yaml}`  \n",
        f"**Split:** {split}\n\n",
        "## Summary\n\n",
        f"| Metric | Value |\n|--------|-------|\n",
        f"| mAP@50     | {results['mAP50']:.4f} |\n",
        f"| mAP@50:95  | {results['mAP50_95']:.4f} |\n",
        f"| Precision  | {results['precision']:.4f} |\n",
        f"| Recall     | {results['recall']:.4f} |\n\n",
        "## Per-class AP@50\n\n",
        "| Class | AP@50 |\n|-------|-------|\n",
    ]
    for cls_name, ap in results["per_class_ap50"].items():
        lines.append(f"| {cls_name} | {ap:.4f} |\n")

    # Target check
    target_met = "PASS" if results["mAP50"] >= 0.70 else "FAIL"
    lines += [
        "\n## Target\n\n",
        f"mAP@50 ≥ 0.70: **{target_met}** (got {results['mAP50']:.4f})\n",
    ]

    report_path.write_text("".join(lines))
    print(f"\nReport written to {report_path}")


def _print_summary(results):
    print("\n" + "=" * 40)
    print(f"  mAP@50    : {results['mAP50']:.4f}  (target ≥ 0.70)")
    print(f"  mAP@50:95 : {results['mAP50_95']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on test split")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    evaluate(args.model, args.data, args.split, args.imgsz, device=args.device)
