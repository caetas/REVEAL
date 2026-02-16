from config import data_raw_dir
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import zipfile
from pathlib import Path
from typing import Callable, Tuple
import torch


"""DATALOADER FOR .ZIP FILES"""


class ZipDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading images from a zip file.

    Args:
        transform (Callable): A callable object (e.g., a torchvision transform) to apply to the loaded images.
        zip_path (Path): The path to the zip file containing the images.
        image_suffix (str): The file suffix (e.g., ".jpg") that valid image files should have.

    Attributes:
        transform (Callable): The provided image transformation function.
        zip_path (Path): The path to the zip file.
        images (list): A list of valid image file names within the zip file.
        image_folder_members (dict): A dictionary mapping image file names to corresponding ZipInfo objects.

    Methods:
        __len__(self): Returns the length of the dataset.
        __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]: Returns an image and a dummy label for a given index.

    Static Methods:
        _valid_member(m: zipfile.ZipInfo, image_suffix: str) -> bool: Checks if a member is a valid image file.
    """
    def __init__(
            self,
            transform: Callable,
            zip_path: Path,
            image_suffix: str,
    ):

        # Assign variables
        self.transform = transform
        self.zip_path = zip_path
        self.images = []

        # Load the zip file
        image_zip = zipfile.ZipFile(self.zip_path)

        # Get the members of the zip file
        self.image_folder_members = {
            str(Path(m.filename)): m
            for m in sorted(image_zip.infolist(), key=lambda x: x.filename)
        }

        # Get the image names from the zip file, check whether they are valid
        for image_name, m in self.image_folder_members.items():
            if not self._valid_member(
                    m, image_suffix
            ):
                continue
            self.images.append(image_name)

    @staticmethod
    def _valid_member(
            m: zipfile.ZipInfo,
            image_suffixes: list,
    ):
        """Returns True if the member is valid based on the list of suffixes"""
        return (
                any(m.filename.endswith(suffix) for suffix in image_suffixes)
                and not m.is_dir()
        )
    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the items for a dataloader object"""

        # Open the zip file
        with zipfile.ZipFile(self.zip_path) as image_zip:

            # Open the image data from the zip file based on index
            with image_zip.open(
                    self.image_folder_members[self.images[index]].filename
            ) as image_file:
                image = Image.open(image_file).convert("RGB")

        # Create dummy label to return DINO dataloader
        label = torch.tensor(0)

        # Apply torchvision transforms if defined
        if self.transform:
            image = self.transform(image)

        return image, label


"""FUNCTION FOR CONCATENATING .ZIP DATASETS"""

def concat_zip_datasets(
        parent_folder: str,
        transform: Callable,
        image_suffix: list = ['.png', 'jpg'],
        datasets: list = None,
):
    """
        Concatenates multiple ZipDatasets into a single ConcatDataset.

        Args:
            parent_folder (str): The path to the parent folder containing multiple zip files to be combined.
            transform (Callable): A callable object (e.g., a torchvision transform) to apply to the loaded images.
            image_suffix (str, optional): The file suffix (e.g., ".jpg") that valid image files should have.
                Defaults to '.png'.

        Returns:
            torch.utils.data.ConcatDataset: A ConcatDataset containing all the ZipDatasets.

        Note:
            To use this function, provide the path to the parent folder containing the zip files you want to combine.
            You can also specify a custom image_suffix and transformation function.
        """

    parent = Path(parent_folder)

    # find ALL .zip files recursively and sort for deterministic order
    zip_files = sorted(p for p in parent.rglob('*.zip') if p.is_file())

    # filter by dataset substrings if provided
    if datasets is not None:
        included_files = [
            p for p in zip_files
            if any(ds.lower() in p.name.lower() for ds in datasets)
        ]
    else:
        included_files = zip_files

    # Construct datasets for each zip file
    dataset = [
        ZipDataset(
            transform=transform,
            zip_path=zip_path,
            image_suffix=image_suffix
        )
        for zip_path in included_files
    ]

    # Concatenate the datasets
    dataset = torch.utils.data.ConcatDataset(dataset)

    return dataset


class GastroNetDataset(Dataset):
    def __init__(self, img_size=256, transform=None):
        # Initialize dataset, e.g., load data files
        self.data_dir = os.path.join(data_raw_dir, "GastroNet-spaarne")
        self.data_files = os.listdir(self.data_dir)
        self.transform = transform  # Define any transformations if needed

    def __len__(self):
        # Return the total number of samples
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load and return a sample from the dataset at the given index
        img_path = os.path.join(self.data_dir, self.data_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0


def get_gastronet_dataloader(
    batch_size=32,
    img_size=256,
    num_workers=0,
    fid = False,
):
    if fid:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Resize((299, 299)),  # Resize to 299x299 for InceptionV3
                transforms.ToTensor(),
                # No normalization - InceptionV3 expects [0, 1] range
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = GastroNetDataset(img_size=img_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader

def get_gastronet_full_dataloader(
    batch_size=32,
    img_size=256,
    num_workers=0,
    folder = None,
    fid = False,
):
    
    if fid:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Resize((299, 299)),  # Resize to 299x299 for InceptionV3
                transforms.ToTensor(),
                # No normalization - InceptionV3 expects [0, 1] range
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    dataset = concat_zip_datasets(
        parent_folder=os.path.join(data_raw_dir, "GastroNet-spaarne") if folder is None else folder,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader
