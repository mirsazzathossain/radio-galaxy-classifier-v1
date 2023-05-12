# -*- coding: utf-8 -*-

"""
Config file for BYOL and downstream tasks.

This file contains functions to write and load the config file for BYOL and
downstream tasks. The config file is written in YAML format.
"""

__author__ = "Mir Sazzat Hossain"

import argparse

import torch
import yaml

byol_config = {
    "exp_params": {
        "N": 16,
        "lr": 3e-4,
        "batch_size": 16,
        "num_workers": torch.multiprocessing.cpu_count(),
        "num_epochs": 500,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "projection_size": 256,
        "proj_hidden_size": 4096,
        "moving_average_decay": 0.99,
    },
    "model_params": {
        "image_size": 151,
        "kernel_size": 5,
    },
    "logging_params": {
        "results_dir": "./results/",
    },
    "data_params": {
        "data_path": "./data/unlabeled/",
    },
}

downstream_config = {
    "exp_params": {
        "lr": 3e-4,
        "weight_decay": 1e-6,
        "batch_size": 16,
        "num_workers": torch.multiprocessing.cpu_count(),
        "num_epochs": 500,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "model_params": {
        "image_size": 151,
        "kernel_size": 5,
        "N": 16,
        "pretrained_model": "./results/models/byol/run_0/best.pt",
    },
    "logging_params": {
        "results_dir": "./results/",
    },
    "data_params": {
        "data_path": "./data/bent_tail/",
        "mean": [0.0032],
        "std": [0.0376],
        "classes": ["WAT", "NAT"],
        "include_small": False,
    },
}


def write_config(config: dict, config_name: str) -> None:
    """
    Write the config file for BYOL or downstream tasks.

    :param config: Config file for BYOL or downstream tasks.
    :type config: dict
    :param config_name: Name of the config file to write.
    :type config_name: str
    """
    with open(
        "./configs/" + config_name + "_config.yaml",
            "w", encoding="utf8") as file:
        yaml.dump(config, file, default_flow_style=False)


def load_config(config_name: str) -> dict:
    """
    Load the config file for BYOL or downstream tasks.

    :param config_name: Name of the config file to load.
    :type config_name: str

    :return: Config file for BYOL.
    :rtype: dict
    """
    config_file = open("./configs/" + config_name +
                       "_config.yaml", "r", encoding="utf8")
    config = yaml.safe_load(config_file)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="byol",
        help="Name of the config file to write (byol or downstream)",
    )
    args = parser.parse_args()

    if args.config == "byol":
        write_config(byol_config, "byol")
    elif args.config == "downstream":
        write_config(downstream_config, "downstream")
    else:
        raise ValueError("Invalid config name")
