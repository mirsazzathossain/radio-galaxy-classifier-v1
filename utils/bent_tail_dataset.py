# -*- coding: utf-8 -*-

"""
This module contains the BentTailDataset class.

The Bent-Tail Dataset is a dataset of 670 to be cont.
"""

__author__ = "Mir Sazzat Hossain"


import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BentTailDataset(Dataset):
    """Bent-Tail Dataset."""

    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose = None,
        include_small: bool = False,
    ) -> None:
        """
        Initialize the Bent-Tail Dataset.

        :param data_dir: Path to the dataset directory
        :type data_dir: str
        :param transform: Transform to be applied on the dataset
        :type transform: transforms.Compose
        :param train: Whether to use the training set or the test set
        :type train: bool
        """
        self.data_dir = data_dir
        self.transform = transform
        self.include_small = include_small

        self.data = []
        self.targets = []

        self._load_data()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        """
        Return the data and the target at the given index.

        :param index: Index of the data and the target
        :type index: int
        :return: Data and the target at the given index
        :rtype: tuple
        """
        data = self.data[index]
        target = self.targets[index]

        if self.transform:
            data = self.transform(data)

        return data, target

    def _load_data(self) -> None:
        """
        Load the data and the targets.

        Load data from folder. File names start with 100 for
        class WAT and 200 for class NAT.
        """
        data_dir = os.path.join(self.data_dir, "good")
        for path in os.listdir(data_dir):
            img = Image.open(os.path.join(data_dir, path))
            self.data.append(np.asarray(img))

            filename = path.split('/')[-1]

            if filename.startswith("100"):
                self.targets.append(0)
            else:
                self.targets.append(1)

        if self.include_small:
            data_dir = os.path.join(self.data_dir, "small")
            for path in os.listdir(data_dir):
                img = Image.open(os.path.join(data_dir, path))
                self.data.append(np.asarray(img))

                if path.startswith("100"):
                    self.targets.append(0)
                else:
                    self.targets.append(1)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)


if __name__ == "__main__":
    dataset = BentTailDataset(
        data_dir="data/bent_tail",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(padding=1, fill=0, padding_mode="constant"),
                transforms.RandomRotation(
                    degrees=360,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    expand=False
                ),
                transforms.Normalize(
                    mean=[0.0032],
                    std=[0.0376],
                ),
            ]
        ),
        include_small=False,
    )
    print(f"Length of the dataset: {len(dataset)}")
    print(f"Data: {dataset[0][0].shape}")
    print(f"Target: {dataset[0][1]}")
