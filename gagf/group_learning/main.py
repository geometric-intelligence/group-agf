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
        X, Y, translations = datasets.load_modular_addition_dataset_2d(config['p'], template, fraction=config['dataset_fraction'], random_state=config['seed'], template_type=config["mnist"])

        template = choose_template(config['p'], template_type=config['template_type'], digit=config['mnist_digit'])

        top_frequency_plot = plot.plot_top_template_components(template, config['p'])

        X, Y, device = move_dataset_to_device_and_flatten(X, Y, config['p'], device=None)

        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using GPU

        model = models.TwoLayerNet(p=config['p'], hidden_size=config['hidden_size'], nonlinearity='square', init_scale=config['init_scale'], output_scale=1e0)
        model = model.to(device)
        loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(config['mom'], 0.999))

        loss_history, accuracy_history, param_history = train.train(model, dataloader, loss, optimizer, epochs=config['epochs'], verbose_interval=config['verbose_interval'])

        model_save_path = (
            f"{config['model_save_dir']}model_"
            f"p{config['p']}_"
            f"digit{config['mnist_digit']}_"
            f"frac{config['dataset_fraction']}_"
            f"type{config['template_type']}_"
            f"seed{config['seed']}.pkl"
        )

        with open(model_save_path, "wb") as f:
            pickle.dump({
                "loss_history": loss_history,
                "accuracy_history": accuracy_history,
                "param_history": param_history
            }, f)

        print(f"Training history saved to {model_save_path}. You can reload it later with pickle.load(open({model_save_path}, 'rb')).")

        


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



def choose_template(p, template_type="mnist", digit=4):
    """Choose template based on type."""
    if template_type == "znz_znz_fixed":
        template = datasets.generate_fixed_template(p)
    elif template_type == "mnist":
        template = datasets.mnist_template(p, digit=digit)
    else:
        raise ValueError(f"Unknown template_type: {template_type}")

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq
    
    return template


def move_dataset_to_device_and_flatten(X, Y, p, device=None):
    """Move dataset tensors to available or specified device.
    
    Parameters
    ----------
    X : np.ndarray
        Input data of shape (num_samples, 2, p*p)
    Y : np.ndarray
        Target data of shape (num_samples, p*p)
    p : int
        Image length. Images are of shape (p, p)
    device : torch.device, optional 
        Device to move tensors to. If None, automatically choose GPU if available.
        
    Returns
    -------
    X : torch.Tensor
        Input data tensor on specified device, flattened to (num_samples, 2*p*p)
    Y : torch.Tensor
        Target data tensor on specified device, flattened to (num_samples, p*p)
    """
    # Flatten X to shape (num_samples, 2*p*p) before converting to tensor
    X_flat = X.reshape(X.shape[0], 2 * p * p)
    Y_flat = Y.reshape(Y.shape[0], p * p)
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_flat, dtype=torch.float32)  # Targets (num_samples, p*p)
    print(f"X_tensor shape: {X_tensor.shape}, Y_tensor shape: {Y_tensor.shape}")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available. Using CUDA.")
        else:
            device = torch.device("cpu")
            print("GPU is not available. Using CPU.")

    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    return X, Y, device


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
            "model_save_path": default_config.model_save_path 
        }

        main_run(main_config)

main()