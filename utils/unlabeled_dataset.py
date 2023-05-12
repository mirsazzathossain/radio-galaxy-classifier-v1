# -*- coding: utf-8 -*-

"""This file contains the dataset class for unlabeled data."""

__author__ = "Mir Sazzat Hossain"


from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UnlabeledDataset(Dataset):
    """Unlabeled dataset for image classification."""

    def __init__(
        self,
        data_path: str,
        transform: transforms.Compose = None,
        image_size: int = 151
    ) -> None:
        """
        Initialize the dataset.

        :param data_path: Path to the data.
        :type data_path: str
        :param transform: Transformations to be applied to the data.
        :type transform: transforms.Compose
        :param image_size: Size of the image.
        :type image_size: int
        """
        self.data_path = data_path
        self.transform = transform
        self.image_size = image_size
        self.data = self.load_data()
        self.length = len(self.data)

    def load_data(self) -> list:
        """Load the data."""
        data = []
        for path in Path(f"{self.data_path}").glob("**/*"):
            img = Image.open(path)
            img = img.resize((self.image_size, self.image_size))
            img = self.transform if self.transform else self.augment(img)
            data.append(img)
        return data

    def augment(self, image: Image) -> torch.Tensor:
        """
        Augment the image.

        :param image: Image to be augmented.
        :type image: PIL.Image.Image

        :return: Augmented image.
        :rtype: torch.Tensor
        """
        augmentations = transforms.Compose(
            [
                transforms.Pad((0, 0, 20, 20), fill=0),
                transforms.CenterCrop(self.image_size),
                transforms.RandomRotation(
                    360,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    expand=False,
                ),
                transforms.ToTensor(),
            ]
        )

        return augmentations(image)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Get the item at the given index.

        :param index: Index of the item.
        :type index: int
        """
        return self.data[index]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return self.length
