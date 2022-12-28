from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UnlabeledDataset(Dataset):
    """Unlabeled dataset for image classification."""

    def __init__(self, data_path, transform=None, image_size=151):
        """
        Initialize the dataset.

        Args:
            data_path (str): Path to the data.
            transform (callable, optional): Optional transform to be applied on a sample.
            image_size (int, optional): Size of the images.
        """
        self.data_path = data_path
        self.transform = transform
        self.image_size = image_size
        self.data = self.load_data()
        self.length = len(self.data)

    def load_data(self):
        """Load the data."""
        data = []
        for path in Path(f"{self.data_path}").glob("**/*"):
            img = Image.open(path)
            img = img.resize((self.image_size, self.image_size))
            img = self.transform if self.transform else self.augment(img)
            data.append(img)
        return data

    def augment(self, image):
        """Augment the image."""

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

    def __getitem__(self, index):
        """Get the item at the given index."""
        return self.data[index]

    def __len__(self):
        """Get the length of the dataset."""
        return self.length
