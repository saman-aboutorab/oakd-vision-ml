"""P3 Traversability CNN — training loop.

Trains TraversabilityNet with a selected fusion strategy.
Logs to W&B, saves best checkpoint by val_loss.

Run:
    python -m oakd_vision.fusion.train_fusion
    python -m oakd_vision.fusion.train_fusion --config training/configs/fusion_config.yaml
    python -m oakd_vision.fusion.train_fusion --strategy attention
    python -m oakd_vision.fusion.train_fusion --strategy gated
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from oakd_vision.fusion.fusion_model import TraversabilityNet
from oakd_vision.fusion.traversability_dataset import (
    make_train_val_datasets,
    compute_class_weights,
    INT_TO_LABEL,
    NUM_CLASSES,
)

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for rgb, depth, labels in loader:
        rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

        logits = model(rgb, depth)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def val_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    # Per-class tracking for accuracy breakdown
    class_correct = torch.zeros(NUM_CLASSES)
    class_total   = torch.zeros(NUM_CLASSES)

    for rgb, depth, labels in loader:
        rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)
        logits = model(rgb, depth)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for c in range(NUM_CLASSES):
            mask = labels == c
            class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            class_total[c]   += mask.sum().item()

    per_class = {
        INT_TO_LABEL[c]: (class_correct[c] / class_total[c].clamp(min=1)).item()
        for c in range(NUM_CLASSES)
    }
    return total_loss / len(loader), correct / total, per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict, strategy_override: str | None = None):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    strategy = strategy_override or cfg["model"]["fusion_strategy"]
    print(f"Fusion strategy: {strategy}\n")

    # --- Dataset ---
    train_ds, val_ds = make_train_val_datasets(
        raw_dir     = cfg["data"]["raw_dir"],
        train_split = cfg["data"]["train_split"],
        patch_size  = tuple(cfg["data"]["patch_size"]),
        grid_cols   = cfg["data"]["grid_cols"],
        grid_rows   = cfg["data"]["grid_rows"],
        depth_max_mm= cfg["data"]["depth_max_mm"],
    )

    class_weights = compute_class_weights(train_ds).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # --- Model ---
    model = TraversabilityNet(
        embedding_dim    = cfg["model"]["embedding_dim"],
        num_classes      = cfg["model"]["num_classes"],
        fusion_strategy  = strategy,
        dropout          = cfg["model"]["dropout"],
    ).to(device)

    counts = model.param_count()
    print(f"Parameters: {counts['model_total']['trainable']:,} trainable / "
          f"{counts['model_total']['total']:,} total\n")

    # Weighted cross-entropy compensates for caution class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["training"]["lr_step_epochs"],
        gamma=0.1,
    )

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"]) / strategy
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- W&B ---
    run_name = f"{cfg['wandb']['run_name']}_{strategy}"
    use_wandb = False
    if WANDB:
        try:
            wandb.init(
                project=cfg["wandb"]["project"],
                name=run_name,
                config={**cfg, "fusion_strategy": strategy},
            )
            use_wandb = True
        except Exception as e:
            print(f"W&B disabled ({e}). Run 'wandb login' to enable.")

    best_val_loss = float("inf")
    epochs = cfg["training"]["epochs"]
    print(f"Training {epochs} epochs — checkpoints → {checkpoint_dir}\n")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            cfg["training"]["grad_clip"],
        )
        val_loss, val_acc, per_class = val_one_epoch(
            model, val_loader, criterion, device,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        per_class_str = "  ".join(
            f"{k[0].upper()}={v:.2f}" for k, v in per_class.items()
        )
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={train_loss:.4f}/{train_acc:.2%}  "
              f"val={val_loss:.4f}/{val_acc:.2%}  "
              f"[{per_class_str}]  lr={lr:.6f}")

        if use_wandb:
            log = {
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss,     "val_acc": val_acc,
                "lr": lr, "epoch": epoch,
            }
            log.update({f"val_acc_{k}": v for k, v in per_class.items()})
            wandb.log(log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  → best saved (val_loss={val_loss:.4f})")

    torch.save(model.state_dict(), checkpoint_dir / "last.pt")
    if use_wandb:
        wandb.finish()

    print(f"\nDone. Best val_loss={best_val_loss:.4f} → {checkpoint_dir}/best.pt")
    return best_val_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",   default="training/configs/fusion_config.yaml")
    parser.add_argument("--strategy", default=None,
                        choices=["concat", "attention", "gated"],
                        help="Override fusion_strategy from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, strategy_override=args.strategy)
