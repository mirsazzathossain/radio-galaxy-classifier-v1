import argparse

import torch

from models.byol_trainer import BYOLTrainer
from utils.setup_configs import load_byol_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model name: byol or downstream")
    parser.add_argument("--seed", type=int, default=2171, help="random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.model == "byol":
        configs = load_byol_config()
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
            projection_hidden_size=configs["exp_params"]["projection_hidden_size"],
            moving_average_decay=configs["exp_params"]["moving_average_decay"],
            data_path=configs["data_params"]["data_path"],
            results_folder=configs["logging_params"]["results_dir"],
        )
        trainer.train()
    elif args.model == "downstream":
        pass
    else:
        raise ValueError("Invalid model name")
