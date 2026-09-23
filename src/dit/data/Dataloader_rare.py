import os
from pathlib import Path

import numpy as np
import torch
from config import data_raw_dir
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

RARE_CLASSES = {"ndbe": 0, "neo": 1}
RARE_TRAIN_ROOT = os.path.join(data_raw_dir, "RARE25-train-data")


def _rare_transform(img_size):
    # same preprocessing as get_gastronet_dataloader, so inputs match the pretrained model
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class RareDataset(Dataset):
    """RARE25 images laid out as <root>/<center>/<class>/*.png, with class folders mapped by RARE_CLASSES
    (ndbe -> 0, neo -> 1)."""

    def __init__(self, root=RARE_TRAIN_ROOT, transform=None, centers=None):
        self.transform = transform
        self.samples = []  # (path, label, center)

        root = Path(root)
        center_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        if centers is not None:
            center_dirs = [d for d in center_dirs if d.name in centers]
        if not center_dirs:
            raise FileNotFoundError(f"No center folders found in {root} (centers={centers})")

        for center_dir in center_dirs:
            for class_name, label in RARE_CLASSES.items():
                self.samples += [(str(p), label, center_dir.name) for p in sorted((center_dir / class_name).glob("*.png"))]

    @property
    def labels(self):
        return [label for _, label, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_rare_dataloader(
    batch_size=32,
    img_size=256,
    num_workers=0,
    root=None,
    centers=None,
    balanced=False,
) -> DataLoader:
    """Training dataloader for the labelled RARE25 dataset (all of RARE25-train-data).

    Batches are `(images, labels)`: images [B, 3, img_size, img_size] in [-1, 1], labels in {0: ndbe, 1: neo}.
    `balanced=True` samples classes uniformly (with replacement) instead of shuffling.
    """
    dataset = RareDataset(root=root or RARE_TRAIN_ROOT, transform=_rare_transform(img_size), centers=centers)
    counts = np.bincount(dataset.labels, minlength=len(RARE_CLASSES))
    print(f"RARE25 train: {len(dataset)} images, " + ", ".join(f"{c}={counts[i]}" for c, i in RARE_CLASSES.items()))

    sampler = None
    if balanced:
        weights = 1.0 / counts[dataset.labels]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(dataset), replacement=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_rare_val_dataloader(batch_size=32, img_size=256, num_workers=0) -> DataLoader:
    """Validation dataloader for OOD evaluation (to be implemented once the validation data is available).

    Must not shuffle, and should yield `(images, labels)` like `get_rare_dataloader` (images in [-1, 1],
    labels 0 = ndbe / 1 = neo). If the dataset exposes `samples` as (path, label, center) tuples, the OOD
    scores CSV also records path and center.
    """
    raise NotImplementedError("get_rare_val_dataloader is not implemented yet (validation data pending)")
