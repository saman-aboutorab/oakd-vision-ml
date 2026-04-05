"""P2 ReID training loop.

Manual training loop (no high-level API):
  - PKSampler guarantees P×K batches for batch-hard mining
  - BatchHardTripletLoss with hardest positive/negative per anchor
  - StepLR learning rate schedule
  - W&B logging of loss, active triplet fraction, LR
  - Checkpoints best model by validation loss

Run:
    python -m oakd_vision.tracker.train_reid
    python -m oakd_vision.tracker.train_reid --config training/configs/reid_config.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from oakd_vision.tracker.reid_model import ReIDNet
from oakd_vision.tracker.losses import BatchHardTripletLoss
from oakd_vision.tracker.triplet_dataset import ReIDDataset, PKSampler

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    total_active = 0.0
    n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        embeddings = model(imgs)
        loss, active_frac = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_active += active_frac
        n += 1

    return total_loss / n, total_active / n


def val_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            embeddings = model(imgs)
            loss, _ = criterion(embeddings, labels)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def main(cfg: dict):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset split: 80% identities train, 20% val ---
    reid_dir = Path(cfg["data"]["reid_dir"])
    all_ids = sorted([d.name for d in reid_dir.iterdir() if d.is_dir()])
    split = int(len(all_ids) * cfg["data"]["train_split"])
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]
    print(f"Identities — train: {len(train_ids)}, val: {len(val_ids)}")

    crop_size = tuple(cfg["data"]["crop_size"])
    P = cfg["sampler"]["P"]
    K = cfg["sampler"]["K"]

    train_dataset = ReIDDataset(reid_dir, crop_size=crop_size, augment=True,
                                identity_filter=train_ids)
    val_dataset   = ReIDDataset(reid_dir, crop_size=crop_size, augment=False,
                                identity_filter=val_ids)

    train_sampler = PKSampler(train_dataset, P=P, K=K)
    train_loader  = DataLoader(train_dataset, batch_sampler=train_sampler,
                               num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset, batch_size=32, shuffle=False,
                               num_workers=4, pin_memory=True)

    # --- Model ---
    model = ReIDNet(
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = BatchHardTripletLoss(margin=cfg["loss"]["margin"])
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg["training"]["lr"],
                           weight_decay=cfg["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["training"]["lr_step_epochs"],
        gamma=0.1,
    )

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- W&B ---
    if WANDB:
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"]["run_name"],
            config=cfg,
        )

    best_val_loss = float("inf")
    epochs = cfg["training"]["epochs"]

    print(f"\nTraining for {epochs} epochs...\n")
    for epoch in range(1, epochs + 1):
        train_loss, active_frac = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            cfg["training"]["grad_clip"]
        )
        val_loss = val_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f}  active={active_frac:.2%}  "
              f"val_loss={val_loss:.4f}  lr={lr:.6f}")

        if WANDB:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                       "active_triplets": active_frac, "lr": lr, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    # Always save last
    torch.save(model.state_dict(), checkpoint_dir / "last.pt")
    if WANDB:
        wandb.finish()
    print(f"\nDone. Best val_loss={best_val_loss:.4f}  →  {checkpoint_dir}/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/reid_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
