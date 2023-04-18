# -*- coding: utf-8 -*-

"""BYOL Trainer."""

__author__ = "Mir Sazzat Hossain"

import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models.byol_pytorch import BYOL
from models.dstreeablelenet import DSteerableLeNet
from utils.unlabeled_dataset import UnlabeledDataset


class BYOLTrainer:
    """BYOL Trainer."""

    def __init__(
        self,
        image_size: int = 151,
        kernel_size: int = 5,
        N: int = 16,
        lr: float = 0.001,
        batch_size: int = 128,
        num_workers: int = 4,
        num_epochs: int = 10,
        device: str = "cuda",
        projection_size: int = 256,
        projection_hidden_size: int = 4096,
        moving_average_decay: float = 0.99,
        data_path: str = "data",
        results_folder: str = "results",
    ) -> None:
        """Initialize BYOL Trainer.

        :param image_size: image size
        :type image_size: int
        :param kernel_size: kernel size
        :type kernel_size: int
        :param N: number of steerable filters
        :type N: int
        :param lr: learning rate
        :type lr: float
        :param batch_size: batch size
        :type batch_size: int
        :param num_workers: number of workers
        :type num_workers: int
        :param num_epochs: number of epochs
        :type num_epochs: int
        :param device: device
        :type device: str
        :param projection_size: projection size
        :type projection_size: int
        :param projection_hidden_size: projection hidden size
        :type projection_hidden_size: int
        :param moving_average_decay: moving average decay
        :type moving_average_decay: float
        :param data_path: data path
        :type data_path: str
        :param results_folder: results folder
        :type results_folder: str
        """
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.N = N
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.device = device
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.moving_average_decay = moving_average_decay
        self.data_path = data_path
        self.results_folder = results_folder

        self.model = DSteerableLeNet(
            imsize=self.image_size, kernel_size=self.kernel_size, N=self.N
        ).to(self.device)

        self.learner = BYOL(
            self.model,
            image_size=self.image_size,
            hidden_layer="fc",
            projection_size=self.projection_size,
            projection_hidden_size=self.projection_hidden_size,
            augment_fn=self.augmentation(),
            augment_fn2=self.augmentation(gaussian_blur=True),
            moving_average_decay=self.moving_average_decay,
        ).to(self.device)

        # set up optimizer
        self.optimizer = torch.optim.Adam(
            self.learner.parameters(), lr=self.lr)

        # find run version
        self.run_version = 0
        while os.path.exists(
            self.results_folder + "/models/byol/run_" + str(self.run_version)
        ):
            self.run_version += 1

        self.model_out_dir = (
            self.results_folder + "/models/byol/run_" + str(self.run_version)
        )

        # create folder if it doesn't exist
        if not os.path.exists(self.model_out_dir):
            os.makedirs(self.model_out_dir)

        # set up tensorboard writer
        self.writer = SummaryWriter(
            self.results_folder + "/logs/byol/run_" + str(self.run_version)
        )

    def train(self) -> None:
        """Train function."""
        train_loader = self.train_dataloader()
        running_loss = 0.0
        best_loss = 1e10
        for epoch in range(self.num_epochs):
            self.learner.train()
            running_loss = 0.0

            loop = tqdm(enumerate(train_loader),
                        total=len(train_loader), leave=False)
            for batch_idx, images in loop:
                images = images.to(self.device)

                # forward pass
                loss = self.learner(images)
                running_loss += loss.item()

                self.learner.update_moving_average()

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                self.optimizer.step()

                # update progress bar
                loop.set_description(f"Epoch [{epoch + 1}/{self.num_epochs}]")
                loop.set_postfix(loss=loss.item())

            # log loss
            running_loss /= len(self.train_dataloader())
            self.writer.add_scalar("Loss/train", running_loss, epoch)

            # save model at the end of each epoch
            self.learner.eval()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": running_loss,
                },
                self.model_out_dir + "/last.pt",
            )

            # save model with the lowest loss
            if running_loss < best_loss:
                best_loss = running_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": running_loss,
                    },
                    self.model_out_dir + "/best.pt",
                )

        self.writer.close()

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Train dataloader function."""
        train_data = UnlabeledDataset(
            data_path=self.data_path,
            image_size=self.image_size,
        )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return train_loader

    def augmentation(self, gaussian_blur: bool = False) -> transforms.Compose:
        """
        Augmentations Function.

        :param gaussian_blur: whether to use gaussian blur
        :type gaussian_blur: bool
        """
        fn = [
            self.RandomApply(
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p=0.8,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                size=151, scale=(0.3, 1.0), ratio=(0.8, 1.0)),
            transforms.RandomRotation(
                360,
                interpolation=transforms.InterpolationMode.BILINEAR,
                expand=False
            ),
            transforms.Normalize((0.0033,), (0.0393,)),
        ]

        if gaussian_blur:
            fn.insert(
                2,
                self.RandomApply(
                    transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                    p=0.5,
                ),
            )

        return transforms.Compose(fn)

    class RandomApply(nn.Module):
        """Random Apply."""

        def __init__(
            self,
            fn: object,
            p: float
        ):
            """
            Initialize Random Apply.

            :param fn: transformation function
            :type fn: object
            :param p: probability
            :type p: float
            """
            super().__init__()
            self.fn = fn
            self.p = p

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            :param x: input tensor
            :type x: torch.Tensor

            :return: transformed tensor
            :rtype: torch.Tensor
            """
            if torch.rand(1) > self.p:
                return x
            return self.fn(x)
