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

        model_save_path = get_model_save_path(config)

        print("Generating dataset...")
        X, Y, template = datasets.load_dataset(config)
        X, Y, device = datasets.move_dataset_to_device_and_flatten(X, Y, device=None)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])  # if using GPU

        model = models.TwoLayerNet(group_size=config['group_size'], hidden_size=config['hidden_size'], nonlinearity='square', init_scale=config['init_scale'], output_scale=1e0)
        model = model.to(device)
        loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(config['mom'], 0.999))

        print("Starting training...")
        loss_history, accuracy_history, param_history = train.train(
            model,
            dataloader,
            loss,
            optimizer,
            epochs=config['epochs'],
            verbose_interval=config['verbose_interval'],
            model_save_path=model_save_path
        )

        print("Training Complete. Generating plots...")
        if config['group'] == 'znz_znz':
            template_power = power.ZnZPower2D(template)
        elif config['group'] == 'dihedral':
            template_power = power.GroupPower(template, group_name='dihedral')
        else:
            raise ValueError(f"Unknown group: {config['group']}")

        loss_plot = plot.plot_loss_curve(loss_history, template_power, show=False)
        top_frequency_plot = plot.plot_top_template_components(template_power, config['group_size'], show=False)
        power_over_training_plot = plot.plot_training_power_over_time(
            template_power, 
            model, 
            device, 
            param_history, 
            X, 
            config['group'], 
            save_path=None, 
            show=False
        )
        neuron_weights_plot = plot.plot_neuron_weights(config['group'], model, config['group_size'], neuron_indices=None)
        model_predictions_plot = plot.plot_model_outputs(config['group'], config['group_size'], model, X, Y, idx=13)        
        wandb.log({
            "loss_plot": wandb.Image(loss_plot),
            "top_frequency_plot": wandb.Image(top_frequency_plot),
            "power_over_training_plot": wandb.Image(power_over_training_plot),
            "neuron_weights_plot": wandb.Image(neuron_weights_plot),
            "model_predictions_plot": wandb.Image(model_predictions_plot),
        })

        print("Plots generated and logged to wandb.")

        wandb_config.update({"full_run": full_run})
        wandb.finish()

    except Exception as e:
        full_run = False
        wandb_config.update({"full_run": full_run})
        logging.exception(e)
        wandb.finish()


def get_model_save_path(config):
    """Generate a unique model save path based on the config parameters."""
    if config['group'] == 'znz_znz':
        model_save_path = (
            f"{default_config.model_save_dir}model_"
            f"group{config['group']}_"
            f"group_size{config['group_size']}_"
            f"digit{config['mnist_digit']}_"
            f"frac{config['dataset_fraction']}_"
            f"type{config['template_type']}_"
            f"init{config['init_scale']}_"
            f"lr{config['lr']}_"
            f"mom{config['mom']}_"
            f"bs{config['batch_size']}_"
            f"epochs{config['epochs']}_"
            f"freq{config['frequencies_to_learn']}_"
            f"seed{config['seed']}.pkl"
        )
    elif config['group'] == 'dihedral':
        model_save_path = (
            f"{default_config.model_save_dir}model_"
            f"group{config['group']}_"
            f"group_size{config['group_size']}_"
            f"frac{config['dataset_fraction']}_"
            f"init{config['init_scale']}_"
            f"lr{config['lr']}_"
            f"mom{config['mom']}_"
            f"bs{config['batch_size']}_"
            f"epochs{config['epochs']}_"
            f"freq{config['frequencies_to_learn']}_"
            f"seed{config['seed']}.pkl"
        )

    return model_save_path


def main():
    """Parse the default_config file and launch all experiments.

    This launches experiments with wandb with different config parameters.
    """
    run_start_time = time.strftime("%m-%d_%H-%M-%S")
    for (
        dataset_fraction,
        group,
        init_scale,
        seed,
        lr,
        mom,
        batch_size,
        epochs,
        verbose_interval,
        frequencies_to_learn,
        
    ) in itertools.product(
        default_config.dataset_fraction,
        default_config.group,
        default_config.init_scale,
        default_config.seed,
        default_config.lr,
        default_config.mom,
        default_config.batch_size,
        default_config.epochs,
        default_config.verbose_interval,
        default_config.frequencies_to_learn,
    ):

        main_config = {
            "dataset_fraction": dataset_fraction,
            "group": group,
            "init_scale": init_scale,
            "run_start_time": run_start_time,
            "seed": seed,
            "lr": lr,
            "mom": mom,
            "batch_size": batch_size,
            "epochs": epochs,
            "verbose_interval": verbose_interval,
            "frequencies_to_learn": frequencies_to_learn,
        }

        if group == "znz_znz":
            for (
                frequencies_to_learn,
                mnist_digit,
                image_length,
            ) in itertools.product(
                default_config.frequencies_to_learn,
                default_config.mnist_digit,
                default_config.image_length,
            ):
                group_size = image_length * image_length
                template_type = 'mnist'
                hidden_size = group_size * frequencies_to_learn
                main_config["hidden_size"] = hidden_size
                main_config["template_type"] = template_type
                main_config["mnist_digit"] = mnist_digit
                main_config["group_size"] = group_size
                main_config["image_length"] = image_length
                main_run(main_config)

        elif group == "dihedral":
            for (
                frequencies_to_learn,
                signal_length_1d,
            ) in itertools.product(
                default_config.frequencies_to_learn,
                default_config.signal_length_1d,
            ):
                group_size = signal_length_1d
                template_type = 'dihedral_fixed'
                hidden_size = 6*group_size
                main_config["hidden_size"] = hidden_size
                main_config["template_type"] = template_type
                main_config["group_size"] = group_size
            main_run(main_config)

        else:
            raise ValueError(f"Unknown group: {group}")

        

main()