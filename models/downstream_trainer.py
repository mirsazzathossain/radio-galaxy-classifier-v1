# -*- coding: utf-8 -*-

"""
Downstream Training/Finetuning Script.

This script is used to train a model on downstream tasks.
"""

__author__ = "Mir Sazzat Hossain"


import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models.dstreeablelenet import DSteerableLeNet
from utils.bent_tail_dataset import AugmentedDataset, BentTailDataset


class DownstreamNet(nn.Module):
    """Downstream network."""

    def __init__(
        self,
        image_size: int = 151,
        num_classes: int = 2,
        N: int = 16
    ) -> None:
        """
        Initialize the downstream network.

        :param image_size: size of the input image
        :type image_size: int
        :param num_classes: number of classes
        :type num_classes: int
        :param N: number of rotations
        :type N: int
        """
        super(DownstreamNet, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.N = N

        z = 0.5 * (self.image_size - 2)
        z = int(0.5 * (z - 2))

        self.fc1 = nn.Linear(16*z*z, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: input tensor
        :type x: torch.Tensor

        :return: output tensor
        :rtype: torch.Tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


class DownstreamTrainer:
    """Downstream Trainer."""

    def __init__(
        self,
        data_dir: str,
        mean: list[float],
        std: list[float],
        pretrained_model: str,
        image_size: int = 151,
        kernel_size: int = 5,
        N: int = 16,
        lr: float = 0.0001,
        weight_decay: float = 0.000001,
        classes: list[str] = ["WAT", "NAT"],
        batch_size: int = 16,
        num_workers: int = 4,
        include_small: bool = False,
        device: str = "cuda",
        results_folder: str = "results",
    ) -> None:
        """
        Initialize the downstream trainer.

        :param data_dir: path to the data directory
        :type data_dir: str
        :param mean: mean of the dataset
        :type mean: list[float]
        :param std: standard deviation of the dataset
        :type std: list[float]
        :param pretrained_model: path to the pretrained model
        :type pretrained_model: str
        :param image_size: size of the input image
        :type image_size: int
        :param kernel_size: size of the kernel
        :type kernel_size: int
        :param N: number of steerable filters
        :type N: int
        :param lr: learning rate
        :type lr: float
        :param weight_decay: weight decay
        :type weight_decay: float
        :param classes: list of classes
        :type classes: list[str]
        :param batch_size: batch size
        :type batch_size: int
        :param num_workers: number of workers
        :type num_workers: int
        :param include_small: whether to include small images or not
        :type include_small: bool
        :param device: device to use
        :type device: str
        :param results_folder: path to the results folder
        :type results_folder: str
        """
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.N = N
        self.lr = lr
        self.weight_decay = weight_decay
        self.classes = classes
        self.num_classes = len(self.classes)
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.include_small = include_small
        self.device = device
        self.results_folder = results_folder

        self.model = DSteerableLeNet(
            imsize=self.image_size, kernel_size=self.kernel_size, N=self.N
        ).to(self.device)

        # load the model
        self.model.eval()
        self.model.load_state_dict(
            torch.load(self.pretrained_model)["model_state_dict"]
        )

        # change the last layer
        self.model.fc = DownstreamNet(
            image_size=self.image_size,
            num_classes=self.num_classes,
            N=self.N
        ).to(self.device)

        # set up optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.9, patience=2
        )

        self.criterion = nn.NLLLoss().to(self.device)

        # set up data loader
        self.train_loader, self.test_loader = self.data_loader(
            data=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            mean=self.mean,
            std=self.std,
            include_small=self.include_small
        )

    def initiate_writer(self) -> None:
        """Initiate the tensorboard writer."""
        # find run version
        self.run_version = 0
        while os.path.exists(
            self.results_folder + "/models/downstream/run_" +
                str(self.run_version)
        ):
            self.run_version += 1

        self.model_out_dir = (
            self.results_folder + "/models/downstream/run_" +
            str(self.run_version)
        )

        # create folder if not exist
        if not os.path.exists(self.model_out_dir):
            os.makedirs(self.model_out_dir)

        # set up tensorboard
        self.writer = SummaryWriter(
            self.results_folder + "/logs/downstream/run_" +
            str(self.run_version)
        )

    def train(self, epochs: int = 100) -> None:
        """
        Train the model.

        :param epochs: number of epochs
        :type epochs: int
        """
        self.initiate_writer()
        total_loss = 0
        best_acc = 0
        correct = 0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            loop = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader), leave=False)
            for batch_idx, (data, target) in loop:
                data, target = data.to(self.device), target.to(self.device)

                # forward pass
                output = F.softmax(self.model(data), dim=1)
                loss = self.criterion(torch.log(output), target)

                # add l2 regularization
                l2_reg = None
                for W in self.model.parameters():
                    if l2_reg is None:
                        l2_reg = W.norm(2)
                    else:
                        l2_reg = l2_reg + W.norm(2)

                loss += l2_reg * 0.1

                total_loss += loss.item()

                predicted = output.argmax(dim=1, keepdim=True)
                correct += predicted.eq(target.view_as(predicted)).sum().item()

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # update weights
                self.optimizer.step()

                # update progress bar
                loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(loss=loss.item())

            # calculate training loss and accuracy
            total_loss /= len(self.train_loader.dataset)
            accuracy = correct / len(self.train_loader.dataset)

            # update learning rate
            self.scheduler.step(total_loss)

            # log training loss
            self.writer.add_scalar("Loss/train", total_loss, epoch)
            self.writer.add_scalar("Accuracy/train", accuracy, epoch)

            # evaluate on test set
            self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = F.softmax(self.model(data), dim=1)
                    test_loss += self.criterion(
                        torch.log(output), target
                    ).item()
                    predicted = output.argmax(dim=1, keepdim=True)
                    correct += predicted.eq(target.view_as(predicted)
                                            ).sum().item()

            # calculate test loss and accuracy
            test_loss /= len(self.test_loader.dataset)
            accuracy = correct / len(self.test_loader.dataset)

            # log test loss
            self.writer.add_scalar("Loss/eval", test_loss, epoch)
            self.writer.add_scalar("Accuracy/eval", accuracy, epoch)

            # save model
            torch.save(self.model.state_dict(),
                       self.model_out_dir + "/last.pt")

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.model.state_dict(),
                           self.model_out_dir + "/best.pt")

        self.writer.close()

    def test(
        self,
        model_path: str = None
    ) -> None:
        """
        Train the model.

        :param model_path: path to the model
        :type model_path: str
        """
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            self.run_version = int(model_path.split("/")[-2].split("_")[-1])
            self.writer = SummaryWriter(
                self.results_folder + "/logs/downstream/run_" +
                str(self.run_version)
            )
        elif not hasattr(self, "writer"):
            self.initiate_writer()

        predict = []
        correct = []

        self.model.eval()
        with torch.no_grad():
            loop = tqdm(enumerate(self.test_loader),
                        total=len(self.test_loader), leave=False)
            for batch_idx, (data, target) in loop:
                data = data.to(self.device)

                output = F.softmax(self.model(data), dim=1)
                predicted = output.argmax(dim=1, keepdim=True)
                predict.append(predicted.cpu().detach().numpy())
                correct.append(target)

        predict = np.concatenate(predict, axis=0)
        correct = np.concatenate(correct, axis=0)

        precision, recall, f_score, _ = precision_recall_fscore_support(
            correct, predict, average=None, labels=np.unique(predict).sort()
        )
        accuracy = accuracy_score(correct, predict)
        conf_matrix = confusion_matrix(
            correct, predict, labels=np.unique(predict).sort())

        # write matrics to tensorboard
        self.writer.add_scalar("Accuracy/test", accuracy, self.run_version)
        self.writer.add_scalar("Precision/test/WAT",
                               precision[0], self.run_version)
        self.writer.add_scalar("Precision/test/NAT",
                               precision[1], self.run_version)
        self.writer.add_scalar("Recall/test/WAT",
                               recall[0], self.run_version)
        self.writer.add_scalar("Recall/test/NAT",
                               recall[1], self.run_version)
        self.writer.add_scalar("F1/test/WAT",
                               f_score[0], self.run_version)
        self.writer.add_scalar("F1/test/NAT",
                               f_score[1], self.run_version)

        # add confusion matrix to tensorboard
        self.writer.add_figure(
            "Confusion Matrix/test",
            self.plot_confusion_matrix(
                conf_matrix,
                classes=self.classes,
                title="Confusion matrix",
            ),
            self.run_version,
        )

        self.writer.close()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        classes: list,
        title: str = "Confusion matrix",
        cmap: plt.cm = plt.cm.Blues,
    ) -> plt.figure:
        """
        Plot the confusion matrix.

        :param cm: confusion matrix
        :type cm: np.ndarray
        :param classes: class names
        :type classes: list
        :param normalize: whether to normalize the confusion matrix or not
        :type normalize: bool
        :param title: title of the plot
        :type title: str
        :param cmap: color map
        :type cmap: plt.cm

        :return: figure
        :rtype: plt.figure
        """
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )

        # Rotate the tick labels and set their alignment.
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor"
        )

        # Loop over data dimensions and create text annotations.
        fmt = "d"
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        return fig

    def data_loader(
        self,
        data: str,
        batch_size: int,
        num_workers: int,
        mean: list[float],
        std: list[float],
        include_small: bool = False,
    ):
        """
        Set up data loader.

        :param data: dataset
        :type data: torch.utils.data.Dataset
        :param batch_size: batch size
        :type batch_size: int
        :param num_workers: number of workers
        :type num_workers: int
        :param mean: mean of the dataset
        :type mean: list[float]
        :param std: standard deviation of the dataset
        :type std: list[float]
        :param include_small: whether to include small images or not
        :type include_small: bool

        :return: data loader
        :rtype: torch.utils.data.DataLoader
        """
        dataset = BentTailDataset(
            data_dir=data,
            transform=self.augmentation(mean, std),
            include_small=include_small
        )

        # split the dataset into train and test
        train_dataset, test_dataset = train_test_split(
            dataset, test_size=0.1, random_state=2171, stratify=dataset.targets
        )

        # augment the training dataset
        train_dataset = AugmentedDataset(
            train_dataset, transform=transforms.Compose([
                transforms.RandomPerspective(
                    distortion_scale=0.1, p=0.5, fill=0
                ),
                transforms.GaussianBlur(kernel_size=3),
            ])
        )

        # set up data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, test_loader

    def augmentation(
        self,
        mean: list[float],
        std: list[float]
    ):
        """
        Set up augmentations.

        :param mean: mean of the dataset
        :type mean: list[float]
        :param std: standard deviation of the dataset
        :type std: list[float]

        :return: composed transforms
        :rtype: transforms.Compose
        """
        func = [
            transforms.ToTensor(),
            transforms.Pad(padding=1, fill=0, padding_mode="constant"),
            transforms.CenterCrop(size=self.image_size),
            transforms.RandomRotation(
                degrees=360,
                interpolation=transforms.InterpolationMode.BILINEAR,
                expand=False
            ),
            transforms.Normalize(mean, std),
        ]
        return transforms.Compose(func)
