"""P3 Traversability CNN — evaluation and ablation comparison.

Loads one or more trained checkpoints, runs inference on the val set,
and produces:
  - Per-class precision, recall, F1
  - Confusion matrix (saved as PNG)
  - Ablation comparison table across strategies (saved to report)
  - Markdown report saved to runs/fusion/evaluation_report.md

Usage:
    # Evaluate a single strategy
    python -m oakd_vision.fusion.evaluate_fusion --strategy concat

    # Run full ablation (all available checkpoints)
    python -m oakd_vision.fusion.evaluate_fusion --ablation
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from oakd_vision.fusion.fusion_model import TraversabilityNet
from oakd_vision.fusion.traversability_dataset import (
    make_train_val_datasets,
    LABEL_TO_INT,
    INT_TO_LABEL,
    NUM_CLASSES,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


LABEL_NAMES = [INT_TO_LABEL[i] for i in range(NUM_CLASSES)]
LABEL_COLOURS = ["#2ecc71", "#f1c40f", "#e74c3c", "#95a5a6"]  # free/caution/obstacle/unknown


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    """Run model on full loader. Returns (all_preds, all_labels) as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []
    for rgb, depth, labels in loader:
        rgb, depth = rgb.to(device), depth.to(device)
        logits = model(rgb, depth)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute per-class precision, recall, F1 and overall accuracy."""
    metrics = {}
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)
        support   = (labels == c).sum()
        metrics[INT_TO_LABEL[c]] = {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "support":   int(support),
        }
    accuracy = (preds == labels).mean()
    metrics["overall_accuracy"] = float(accuracy)
    return metrics


def confusion_matrix(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    return cm


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: Path):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    # Normalize rows to percentages
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}\n({count})", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


def plot_f1_comparison(results: dict, save_path: Path):
    """Bar chart comparing F1 per class across all evaluated strategies."""
    if not HAS_MPL or len(results) < 2:
        return
    strategies = list(results.keys())
    x = np.arange(NUM_CLASSES)
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, strategy in enumerate(strategies):
        f1_vals = [results[strategy]["metrics"][lbl]["f1"] for lbl in LABEL_NAMES]
        offset = (i - len(strategies) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_vals, width, label=strategy)
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Ablation: F1 per Class across Fusion Strategies")
    ax.legend()
    ax.axhline(0.7, color="grey", linestyle="--", linewidth=0.8, label="target")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved F1 comparison → {save_path}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(results: dict, report_path: Path):
    lines = ["# P3 Traversability CNN — Evaluation Report\n"]

    # --- Per-strategy sections ---
    for strategy, data in results.items():
        m = data["metrics"]
        lines.append(f"## Strategy: {strategy}\n")
        lines.append(f"**Checkpoint:** `{data['checkpoint']}`  ")
        lines.append(f"**Val frames:** {data['val_frames']}  ")
        lines.append(f"**Val patches:** {data['val_patches']}\n")
        lines.append(f"**Overall accuracy:** {m['overall_accuracy']:.1%}\n")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|-------|-----------|--------|----|---------|")
        for lbl in LABEL_NAMES:
            mc = m[lbl]
            lines.append(
                f"| {lbl} | {mc['precision']:.3f} | {mc['recall']:.3f} | "
                f"{mc['f1']:.3f} | {mc['support']} |"
            )
        lines.append("")

    # --- Ablation comparison table ---
    if len(results) > 1:
        lines.append("## Ablation Comparison\n")
        header = "| Strategy | Accuracy | F(free) | C(caution) | O(obstacle) | U(unknown) |"
        sep    = "|----------|----------|---------|------------|-------------|------------|"
        lines.append(header)
        lines.append(sep)
        for strategy, data in results.items():
            m = data["metrics"]
            lines.append(
                f"| {strategy} | {m['overall_accuracy']:.1%} | "
                f"{m['free']['f1']:.3f} | {m['caution']['f1']:.3f} | "
                f"{m['obstacle']['f1']:.3f} | {m['unknown']['f1']:.3f} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"  Saved report → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_strategy(strategy: str, cfg: dict, device: torch.device) -> dict | None:
    checkpoint = Path(cfg["training"]["checkpoint_dir"]) / strategy / "best.pt"
    if not checkpoint.exists():
        print(f"  No checkpoint found at {checkpoint} — skipping.")
        return None

    print(f"\nEvaluating: {strategy}  ({checkpoint})")

    _, val_ds = make_train_val_datasets(
        raw_dir      = cfg["data"]["raw_dir"],
        train_split  = cfg["data"]["train_split"],
        patch_size   = tuple(cfg["data"]["patch_size"]),
        grid_cols    = cfg["data"]["grid_cols"],
        grid_rows    = cfg["data"]["grid_rows"],
        depth_max_mm = cfg["data"]["depth_max_mm"],
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    model = TraversabilityNet(
        embedding_dim   = cfg["model"]["embedding_dim"],
        num_classes     = cfg["model"]["num_classes"],
        fusion_strategy = strategy,
        dropout         = cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    preds, labels = run_inference(model, val_loader, device)
    metrics = compute_metrics(preds, labels)
    cm = confusion_matrix(preds, labels)

    # Print summary
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.1%}")
    for lbl in LABEL_NAMES:
        mc = metrics[lbl]
        print(f"  {lbl:8s}: P={mc['precision']:.3f}  R={mc['recall']:.3f}  "
              f"F1={mc['f1']:.3f}  support={mc['support']}")

    # Save confusion matrix
    out_dir = Path(cfg["training"]["checkpoint_dir"]) / strategy
    plot_confusion_matrix(cm, f"Confusion Matrix — {strategy}", out_dir / "confusion_matrix.png")

    return {
        "checkpoint":  str(checkpoint),
        "val_frames":  len(val_ds._find_labeled_frames()) if hasattr(val_ds, '_find_labeled_frames') else "?",
        "val_patches": len(val_ds),
        "metrics":     metrics,
        "cm":          cm,
    }


def main(cfg: dict, strategies: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    for strategy in strategies:
        result = evaluate_strategy(strategy, cfg, device)
        if result:
            results[strategy] = result

    if not results:
        print("No checkpoints found. Train at least one strategy first.")
        return

    # Ablation F1 comparison plot
    out_dir = Path(cfg["training"]["checkpoint_dir"])
    plot_f1_comparison(results, out_dir / "ablation_f1.png")

    # Markdown report
    write_report(results, out_dir / "evaluation_report.md")

    print(f"\nDone. Results in {out_dir.resolve()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",   default="training/configs/fusion_config.yaml")
    parser.add_argument("--strategy", default=None,
                        choices=["concat", "attention", "gated"],
                        help="Evaluate a single strategy")
    parser.add_argument("--ablation", action="store_true",
                        help="Evaluate all available checkpoints for ablation table")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.ablation:
        strategies = ["concat", "attention", "gated"]
    elif args.strategy:
        strategies = [args.strategy]
    else:
        # Default: evaluate whatever checkpoints exist
        ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
        strategies = [d.name for d in ckpt_dir.iterdir()
                      if d.is_dir() and (d / "best.pt").exists()]
        if not strategies:
            strategies = ["concat"]

    main(cfg, strategies)
