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
RARE_VAL_ROOT = os.path.join(data_raw_dir, "RARE25-val-data")


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
    """RARE25 images laid out as <root>/<center>/<class>/*.png (train) or <root>/<class>/*.png (val, no
    center folders), with class folders mapped by RARE_CLASSES (ndbe -> 0, neo -> 1). Without center folders
    the center is the hospital prefix of the filename (e.g. "amc" in amc_1092_wle_image_ndbe_na_na_1.png)."""

    def __init__(self, root=RARE_TRAIN_ROOT, transform=None, centers=None):
        self.transform = transform
        self.samples = []  # (path, label, center)

        root = Path(root)
        if any((root / class_name).is_dir() for class_name in RARE_CLASSES):
            if centers is not None:
                raise ValueError(f"{root} has no center folders, cannot filter by centers={centers}")
            groups = [(root, None)]
        else:
            groups = [(d, d.name) for d in sorted(root.iterdir()) if d.is_dir()]
            if centers is not None:
                groups = [(d, name) for d, name in groups if name in centers]
            if not groups:
                raise FileNotFoundError(f"No center folders found in {root} (centers={centers})")

        for group_dir, center in groups:
            for class_name, label in RARE_CLASSES.items():
                self.samples += [
                    (str(p), label, center if center is not None else p.name.split("_")[0])
                    for p in sorted((group_dir / class_name).glob("*.png"))
                ]
        if not self.samples:
            raise FileNotFoundError(f"No images found in {root}")

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


def get_rare_val_dataloader(batch_size=32, img_size=256, num_workers=0, root=None) -> DataLoader:
    """Validation dataloader for OOD evaluation (RARE25-val-data/{ndbe,neo}/*.png).

    Same preprocessing and labels as `get_rare_dataloader`. Not shuffled, so the OOD scores CSV rows line up
    with `dataset.samples` (path, label, center).
    """
    dataset = RareDataset(root=root or RARE_VAL_ROOT, transform=_rare_transform(img_size))
    counts = np.bincount(dataset.labels, minlength=len(RARE_CLASSES))
    print(f"RARE25 val: {len(dataset)} images, " + ", ".join(f"{c}={counts[i]}" for c, i in RARE_CLASSES.items()))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
