"""TraversabilityDataset — patch-level dataset for P3 fusion model.

Each labeled frame (640×480 RGB + 640×480 uint16 depth) is divided into a
GRID_COLS × GRID_ROWS grid. Every cell produces one sample:

    rgb_patch   : [3, H, W] float tensor (ImageNet normalized)
    depth_patch : [1, H, W] float tensor (normalized to [0, 1])
    label       : int  0=free  1=caution  2=obstacle  3=unknown

One labeled frame → 48 patch samples (8 cols × 6 rows).
125 labeled frames → 6000 patch samples before train/val split.

Depth encoding
--------------
Raw uint16 values are in millimetres. We:
  1. Clip to depth_max_mm (default 4000mm = 4m)
  2. Divide by depth_max_mm → [0, 1] float
  3. Zero-depth pixels (no stereo measurement) stay at 0.0

This puts depth on the same [0,1] scale as RGB and avoids the model being
confused by outlier values from stereo noise.

Data augmentation (train only)
-------------------------------
Applied identically to both RGB and depth patches so spatial alignment is
preserved. Horizontal flip only — vertical flip would confuse floor geometry.
Color jitter applied to RGB only (depth has no colour).
"""

import json
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

LABEL_TO_INT = {
    "free":     0,
    "caution":  1,
    "obstacle": 2,
    "unknown":  3,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
NUM_CLASSES = len(LABEL_TO_INT)

# ImageNet normalization for RGB branch (pretrained ResNet18 backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TraversabilityDataset(Dataset):
    """Patch-level dataset.

    Args:
        raw_dir:      Path to dataset/traversability/raw/
        patch_size:   (H, W) each patch is resized to this before feeding the model
        grid_cols:    number of horizontal grid divisions (default 8)
        grid_rows:    number of vertical grid divisions (default 6)
        depth_max_mm: depth values clipped to this range before normalizing
        augment:      apply random horizontal flip + RGB color jitter
        frame_paths:  if provided, only use these specific frames (for train/val split)
    """

    def __init__(
        self,
        raw_dir: str | Path,
        patch_size: tuple[int, int] = (64, 64),
        grid_cols: int = 8,
        grid_rows: int = 6,
        depth_max_mm: int = 4000,
        augment: bool = False,
        frame_paths: Optional[list[Path]] = None,
    ):
        self.raw_dir     = Path(raw_dir)
        self.patch_size  = patch_size   # (H, W)
        self.grid_cols   = grid_cols
        self.grid_rows   = grid_rows
        self.depth_max   = depth_max_mm
        self.augment     = augment

        # Build flat list of (rgb_path, depth_path, row, col, label_int)
        self.samples: list[tuple[Path, Path, int, int, int]] = []

        frames = frame_paths if frame_paths is not None else self._find_labeled_frames()
        for rgb_path in frames:
            stem = rgb_path.stem.replace("_rgb", "")
            depth_path  = self.raw_dir / f"{stem}_depth.npy"
            labels_path = self.raw_dir / f"{stem}_labels.json"

            if not depth_path.exists() or not labels_path.exists():
                continue

            labels_data = json.loads(labels_path.read_text())
            grid = labels_data["labels"]   # list[list[str]], [row][col]

            for row in range(grid_rows):
                for col in range(grid_cols):
                    label_str = grid[row][col]
                    label_int = LABEL_TO_INT.get(label_str, 3)  # unknown as fallback
                    self.samples.append((rgb_path, depth_path, row, col, label_int))

        print(f"TraversabilityDataset: {len(frames)} frames → {len(self.samples)} patches  "
              f"[augment={augment}]")

    def _find_labeled_frames(self) -> list[Path]:
        rgb_paths = sorted(self.raw_dir.glob("*_rgb.jpg"))
        labeled = [p for p in rgb_paths
                   if (self.raw_dir / (p.stem.replace("_rgb", "") + "_labels.json")).exists()]
        return labeled

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rgb_path, depth_path, row, col, label = self.samples[idx]

        # Load full frame
        bgr = cv2.imread(str(rgb_path))
        depth_mm = np.load(str(depth_path))   # uint16, mm

        img_h, img_w = bgr.shape[:2]
        ph, pw = self.patch_size

        # Crop the cell from full frame
        cell_h = img_h / self.grid_rows
        cell_w = img_w / self.grid_cols
        y1 = int(row * cell_h)
        y2 = int((row + 1) * cell_h)
        x1 = int(col * cell_w)
        x2 = int((col + 1) * cell_w)

        rgb_crop   = bgr[y1:y2, x1:x2]           # [cell_h, cell_w, 3] BGR
        depth_crop = depth_mm[y1:y2, x1:x2]      # [cell_h, cell_w] uint16

        # Augmentation: horizontal flip (applied to both)
        if self.augment and random.random() < 0.5:
            rgb_crop   = cv2.flip(rgb_crop, 1)
            depth_crop = np.fliplr(depth_crop).copy()

        # --- RGB tensor ---
        rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB)
        pil_rgb  = Image.fromarray(rgb_crop)

        rgb_tf = transforms.Compose([
            transforms.Resize(self.patch_size),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2) if self.augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        rgb_tensor = rgb_tf(pil_rgb)   # [3, H, W]

        # --- Depth tensor ---
        depth_resized = cv2.resize(depth_crop.astype(np.float32), (pw, ph),
                                   interpolation=cv2.INTER_NEAREST)
        # Normalize: clip to max, divide to [0,1], zero stays zero
        depth_norm = np.clip(depth_resized, 0, self.depth_max) / self.depth_max
        depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).float()  # [1, H, W]

        return rgb_tensor, depth_tensor, label


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def make_train_val_datasets(
    raw_dir: str | Path,
    train_split: float = 0.8,
    patch_size: tuple[int, int] = (64, 64),
    grid_cols: int = 8,
    grid_rows: int = 6,
    depth_max_mm: int = 4000,
    seed: int = 42,
) -> tuple[TraversabilityDataset, TraversabilityDataset]:
    """Split labeled frames into train/val datasets.

    Split is done at the **frame level** (not patch level) to prevent
    patches from the same frame appearing in both train and val.

    Returns:
        (train_dataset, val_dataset)
    """
    raw_dir = Path(raw_dir)
    rgb_paths = sorted(raw_dir.glob("*_rgb.jpg"))
    labeled = [p for p in rgb_paths
               if (raw_dir / (p.stem.replace("_rgb", "") + "_labels.json")).exists()]

    random.seed(seed)
    random.shuffle(labeled)

    n_train = int(len(labeled) * train_split)
    train_frames = labeled[:n_train]
    val_frames   = labeled[n_train:]

    print(f"Frame split: {len(train_frames)} train / {len(val_frames)} val  "
          f"(total {len(labeled)} labeled frames)")

    train_ds = TraversabilityDataset(
        raw_dir, patch_size, grid_cols, grid_rows, depth_max_mm,
        augment=True, frame_paths=train_frames,
    )
    val_ds = TraversabilityDataset(
        raw_dir, patch_size, grid_cols, grid_rows, depth_max_mm,
        augment=False, frame_paths=val_frames,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Class weights (for imbalanced caution class)
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: TraversabilityDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Returns [NUM_CLASSES] float tensor — pass to nn.CrossEntropyLoss(weight=...).
    """
    counts = torch.zeros(NUM_CLASSES)
    for *_, label in dataset.samples:
        counts[label] += 1
    # Inverse frequency, normalized
    weights = counts.sum() / (NUM_CLASSES * counts.clamp(min=1))
    print(f"Class weights: { {INT_TO_LABEL[i]: f'{weights[i]:.2f}' for i in range(NUM_CLASSES)} }")
    return weights
