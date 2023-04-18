# -*- coding: utf-8 -*-

"""Trainings script for BYOL and downstream tasks."""

__author__ = "Mir Sazzat Hossain"


import argparse
import os
import random
import warnings

import numpy as np
import torch

from models.byol_trainer import BYOLTrainer
from models.downstream_trainer import DownstreamTrainer
from utils.setup_configs import load_config

warnings.filterwarnings("ignore")


def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility.

    :param seed: random seed
    :type seed: int
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="model name: byol or downstream")
    parser.add_argument("--seed", type=int, default=2171, help="random seed")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.model == "byol":
        configs = load_config("byol")
        trainer = BYOLTrainer(
            image_size=configs["model_params"]["image_size"],
            kernel_size=configs["model_params"]["kernel_size"],
            N=configs["exp_params"]["N"],
            lr=configs["exp_params"]["lr"],
            batch_size=configs["exp_params"]["batch_size"],
            num_workers=configs["exp_params"]["num_workers"],
            num_epochs=configs["exp_params"]["num_epochs"],
            device=configs["exp_params"]["device"],
            projection_size=configs["exp_params"]["projection_size"],
            projection_hidden_size=configs["exp_params"]["proj_hidden_size"],
            moving_average_decay=configs["exp_params"]["moving_average_decay"],
            data_path=configs["data_params"]["data_path"],
            results_folder=configs["logging_params"]["results_dir"],
        )
        trainer.train()
    elif args.model == "downstream":
        configs = load_config("downstream")
        trainer = DownstreamTrainer(
            data_dir=configs["data_params"]["data_path"],
            mean=configs["data_params"]["mean"],
            std=configs["data_params"]["std"],
            pretrained_model=configs["model_params"]["pretrained_model"],
            image_size=configs["model_params"]["image_size"],
            kernel_size=configs["model_params"]["kernel_size"],
            N=configs["model_params"]["N"],
            lr=configs["exp_params"]["lr"],
            weight_decay=configs["exp_params"]["weight_decay"],
            classes=configs["data_params"]["classes"],
            batch_size=configs["exp_params"]["batch_size"],
            num_workers=configs["exp_params"]["num_workers"],
            include_small=configs["data_params"]["include_small"],
            device=configs["exp_params"]["device"],
            results_folder=configs["logging_params"]["results_dir"],
        )
        trainer.train(epochs=configs["exp_params"]["num_epochs"])
        trainer.test()
    else:
        raise ValueError("Invalid model name")
