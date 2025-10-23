import numpy as np
import random
import torch
import time
import datetime
import os
import torch.nn as nn
import torch.optim as optim
import shutil
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
# import setcwd
# setcwd.main()

import importlib
import pickle

import gagf.group_learning.models as models
import datasets
import theory
import train
import plot
import saved_datasets

import wandb
import itertools
import logging
import default_config

today = datetime.date.today()


def main_run(config):
    """Run regression experiments."""
    full_run = True
    print(f"run_start: {today}")
    wandb.init(
        project="gagf",
        tags=[
            f"{today}",
            f"run_start_{config['run_start_time']}",
        ]
    )
    wandb_config = wandb.config
    wandb_config.update(config)

    run_name = f"run_{wandb.run.id}"
    wandb.run.name = run_name
    try:
        logging.info(f"\n\n---> START run: {run_name}.")

        # define template
        template = datasets.mnist_template(config['p'], digit=config['mnist_digit'])

        # load dataset
        X, Y, translations = datasets.load_modular_addition_dataset_2d(config['p'], template, fraction=config['dataset_fraction'], random_state=config['seed'], template_type="mnist")






        # wandb.log({"mse": mse})
        # for plot_name, plot in plots.items():
        #     wandb.log({plot_name: wandb.Image(plot)})
        # logging.info(f"mse: {mse}")

        wandb_config.update({"full_run": full_run})
        wandb.finish()




    except Exception as e:
        full_run = False
        wandb_config.update({"full_run": full_run})
        logging.exception(e)
        wandb.finish()



def main():
    """Parse the default_config file and launch all experiments.

    This launches experiments with wandb with different config parameters.
    """
    run_start_time = time.strftime("%m-%d_%H-%M-%S")
    for (
        p,
        mnist_digit,
        dataset_fraction,
        group,
        n_frequencies_to_learn,
        init_scale,
        seed,
        
    ) in itertools.product(
        default_config.p,
        default_config.mnist_digit,
        default_config.dataset_fraction,
        default_config.group,
        default_config.n_frequencies_to_learn,
        default_config.init_scale,
        default_config.seed,
    ):

        main_config = {
            "p": p,
            "mnist_digit": mnist_digit,
            "dataset_fraction": dataset_fraction,
            "group": group,
            "n_frequencies_to_learn": n_frequencies_to_learn,
            "init_scale": init_scale,
            "run_start_time": run_start_time,
            "seed": seed,
        }

        main_run(main_config)

main()