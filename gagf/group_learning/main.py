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
import gagf.group_learning.datasets as datasets
import gagf.group_learning.power as power
import gagf.group_learning.train as train
import gagf.group_learning.plot as plot
import gagf.group_learning.saved_datasets as saved_datasets

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

        template = datasets.choose_template(config['p'], template_type=config['template_type'], digit=config['mnist_digit'])

        # load dataset
        X, Y, translations = datasets.load_modular_addition_dataset_2d(config['p'], template, fraction=config['dataset_fraction'], random_state=config['seed'], template_type=config["mnist"])

        top_frequency_plot = plot.plot_top_template_components(template, config['p'])

        X, Y, device = datasets.move_dataset_to_device_and_flatten(X, Y, config['p'], device=None)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])  # if using GPU

        model = models.TwoLayerNet(p=config['p'], hidden_size=config['hidden_size'], nonlinearity='square', init_scale=config['init_scale'], output_scale=1e0)
        model = model.to(device)
        loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(config['mom'], 0.999))

        # Train the model and track training history for loss, accuracy, and parameters
        loss_history, accuracy_history, param_history = train.train(
            model,
            dataloader,
            loss,
            optimizer,
            epochs=config['epochs'],
            verbose_interval=config['verbose_interval'],
            model_save_path=config['model_save_path']
        )

        loss_plot = plot.plot_loss_curve(loss_history, template)
        power_over_training_plot = plot.plot_power_over_time(model, param_history, X, template, config['p'])
        neuron_weights_plot = plot.plot_neuron_weights_evolution(model, param_history, template, config['p'])
        wandb.log({"loss_plot": wandb.Image(loss_plot)})

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
        template_type,
        lr,
        mom,
        batch_size,
        epochs,
        verbose_interval,
        hidden_size,        
        
    ) in itertools.product(
        default_config.p,
        default_config.mnist_digit,
        default_config.dataset_fraction,
        default_config.group,
        default_config.n_frequencies_to_learn,
        default_config.init_scale,
        default_config.seed,
        default_config.template_type,
        default_config.lr,
        default_config.mom,
        default_config.batch_size,
        default_config.epochs,
        default_config.verbose_interval,
        default_config.hidden_size,
    ):

        model_save_path = (
            f"{default_config.model_save_dir}model_"
            f"p{p}_"
            f"digit{mnist_digit}_"
            f"frac{dataset_fraction}_"
            f"type{template_type}_"
            f"seed{seed}.pkl"
        )

        main_config = {
            "p": p,
            "mnist_digit": mnist_digit,
            "dataset_fraction": dataset_fraction,
            "group": group,
            "n_frequencies_to_learn": n_frequencies_to_learn,
            "init_scale": init_scale,
            "run_start_time": run_start_time,
            "seed": seed,
            "template_type": template_type,
            "lr": lr,
            "mom": mom,
            "batch_size": batch_size,
            "epochs": epochs,
            "verbose_interval": verbose_interval,
            "hidden_size": hidden_size,
            "model_save_path": model_save_path,
        }

        main_run(main_config)

main()