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
        "projection_hidden_size": 4096,
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


def write_byol_config():
    """
    Writes the config file for BYOL.
    Returns:
        None
    """
    with open("./configs/byol_config.yaml", "w") as f:
        yaml.dump(byol_config, f, default_flow_style=False)


def load_byol_config():
    """
    Loads the config file for BYOL.
    Returns:
        config: Dictionary containing the config file.
    """
    config_file = open("./configs/byol_config.yaml", "r")
    config = yaml.safe_load(config_file)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="which config to write. Options: byol or downstream"
    )
    args = parser.parse_args()

    if args.config == "byol":
        write_byol_config()
    elif args.config == "downstream":
        pass
    else:
        raise ValueError("Invalid config name")
