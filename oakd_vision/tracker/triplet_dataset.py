"""ReID dataset and PKSampler for batch-hard triplet training.

PKSampler: each batch contains exactly P identities × K crops.
This guarantees every batch has valid positives (same identity) and
negatives (different identity) — required for batch-hard mining.

Directory structure expected:
    dataset/reid/
    ├── shoe_001/
    │   ├── shoe_001_0000.jpg
    │   ├── shoe_001_0001.jpg
    │   └── ...
    ├── shoe_002/
    └── ...
"""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image


# ImageNet normalization — required since backbone is pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(crop_size: tuple, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ReIDDataset(Dataset):
    """Loads all crops from dataset/reid/ and assigns integer identity labels.

    Args:
        reid_dir: path to root folder containing identity subfolders
        crop_size: (H, W) to resize each crop
        augment: apply random augmentation (True for train, False for val/eval)
        identity_filter: if provided, only include these identity folder names
    """

    def __init__(
        self,
        reid_dir: str,
        crop_size: tuple = (128, 128),
        augment: bool = True,
        identity_filter: list = None,
    ):
        self.transform = build_transforms(crop_size, augment)
        self.samples = []   # list of (image_path, label_int)
        self.label_to_name = {}

        reid_path = Path(reid_dir)
        identity_dirs = sorted([d for d in reid_path.iterdir() if d.is_dir()])

        if identity_filter is not None:
            identity_dirs = [d for d in identity_dirs if d.name in identity_filter]

        for label_int, id_dir in enumerate(identity_dirs):
            self.label_to_name[label_int] = id_dir.name
            for img_path in sorted(id_dir.glob("*.jpg")):
                self.samples.append((img_path, label_int))

        self.num_identities = len(identity_dirs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


class PKSampler(Sampler):
    """Samples P identities × K crops per batch.

    Guarantees each batch has at least K samples per identity,
    which is required for batch-hard triplet mining to work.

    Args:
        dataset: ReIDDataset
        P: number of identities per batch
        K: number of crops per identity per batch
    """

    def __init__(self, dataset: ReIDDataset, P: int = 8, K: int = 4):
        self.P = P
        self.K = K

        # Group sample indices by label
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(dataset.samples):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.labels = list(self.label_to_indices.keys())
        # Drop identities with fewer than K samples
        self.labels = [l for l in self.labels if len(self.label_to_indices[l]) >= K]

        self.batch_size = P * K
        self.num_batches = len(self.labels) // P

    def __iter__(self):
        labels = self.labels.copy()
        random.shuffle(labels)

        for i in range(self.num_batches):
            batch_labels = labels[i * self.P:(i + 1) * self.P]
            indices = []
            for label in batch_labels:
                pool = self.label_to_indices[label]
                chosen = random.choices(pool, k=self.K)  # sample with replacement if needed
                indices.extend(chosen)
            yield from indices

    def __len__(self):
        return self.num_batches * self.batch_size
