"""Build a YOLOv8-ready train/val/test split from a flat labeled dataset.

Expects the source directory to contain:
    images/    *.jpg (or .png)
    labels/    *.txt  — YOLO format: one line per object: class cx cy w h
    classes.txt        — one class name per line (index = class_id)

After running, the output directory will contain:
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
    dataset.yaml   — ready for `yolo train data=<output>/dataset.yaml`

Only images that have a matching label file (even if empty = background) are
included. Images without any label file are skipped with a warning.

Usage:
    python -m oakd_vision.capture.dataset_builder \\
        --source data/labeled \\
        --output data/dataset \\
        --split 0.7 0.2 0.1 \\
        --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Optional

import yaml


class DatasetBuilder:
    """Splits a flat YOLO-format dataset into train/val/test subsets."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        split: tuple = (0.7, 0.2, 0.1),
        seed: int = 42,
    ):
        """
        Args:
            source_dir: directory containing images/, labels/, classes.txt
            output_dir: where to write the split dataset + dataset.yaml
            split: (train, val, test) fractions — must sum to 1.0
            seed: random seed for reproducible splits
        """
        assert abs(sum(split) - 1.0) < 1e-6, "Split fractions must sum to 1.0"

        self.source = Path(source_dir)
        self.output = Path(output_dir)
        self.split = split
        self.seed = seed

    def build(self) -> Path:
        """Run the split and write output. Returns the path to dataset.yaml."""
        class_names = self._read_classes()
        pairs = self._collect_pairs()

        if not pairs:
            raise ValueError(f"No labeled image pairs found in {self.source}")

        train_pairs, val_pairs, test_pairs = self._split(pairs)

        print(f"Dataset: {len(pairs)} samples → "
              f"train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

        for subset, subset_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
            self._copy_subset(subset, subset_pairs)

        yaml_path = self._write_yaml(class_names)
        print(f"dataset.yaml written to {yaml_path}")
        return yaml_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_classes(self) -> list:
        classes_file = self.source / "classes.txt"
        if not classes_file.exists():
            raise FileNotFoundError(f"classes.txt not found in {self.source}")
        names = [ln.strip() for ln in classes_file.read_text().splitlines() if ln.strip()]
        print(f"Classes ({len(names)}): {names}")
        return names

    def _collect_pairs(self) -> list:
        """Find all (image_path, label_path) pairs where both files exist."""
        images_dir = self.source / "images"
        labels_dir = self.source / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"images/ directory not found in {self.source}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"labels/ directory not found in {self.source}")

        pairs = []
        skipped = 0
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in self.IMAGE_EXTS:
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                skipped += 1

        if skipped:
            print(f"Warning: {skipped} images skipped (no matching .txt label file)")

        return pairs

    def _split(self, pairs: list) -> tuple:
        rng = random.Random(self.seed)
        shuffled = pairs.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self.split[0])
        n_val = int(n * self.split[1])

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]

        return train, val, test

    def _copy_subset(self, subset: str, pairs: list):
        img_out = self.output / "images" / subset
        lbl_out = self.output / "labels" / subset
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    def _write_yaml(self, class_names: list) -> Path:
        data = {
            "path": str(self.output.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_names),
            "names": class_names,
        }
        yaml_path = self.output / "dataset.yaml"
        yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build YOLOv8-ready train/val/test split")
    parser.add_argument("--source", required=True, help="Flat labeled dataset directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--split", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    builder = DatasetBuilder(args.source, args.output, tuple(args.split), args.seed)
    builder.build()
