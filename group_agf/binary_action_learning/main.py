import datetime
import itertools
import logging
import time

from seaborn._core.typing import default

import default_config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from escnn.group import *
from torch.utils.data import DataLoader, TensorDataset

import group_agf.binary_action_learning.datasets as datasets
import group_agf.binary_action_learning.models as models
import group_agf.binary_action_learning.plot as plot
import group_agf.binary_action_learning.power as power
import group_agf.binary_action_learning.train as train
from group_agf.binary_action_learning.optimizer import PerNeuronScaledSGD

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
        ],
    )
    wandb_config = wandb.config
    wandb_config.update(config)

    run_name = f"run_{wandb.run.id}"
    wandb.run.name = run_name
    try:
        logging.info(f"\n\n---> START run: {run_name}.")

        print("Generating dataset...")
        X, Y, template = datasets.load_dataset(config)
        assert (
            len(template) == config["group_size"]
        ), "Template size does not match group size."

        if config["group_name"] == "cnxcn":
            template_power = power.CyclicPower(template, template_dim=2)
        elif config["group_name"] == "cn":
            template_power = power.CyclicPower(template, template_dim=1)
        else:
            template_power = power.GroupPower(template, group=config["group"])

        print(f"Template powers:\n {template_power.power}")

        X, Y, device = datasets.move_dataset_to_device_and_flatten(X, Y, device=None)

        # Determine batch size: if 'full', set to all samples
        if config["batch_size"] == "full":
            config["batch_size"] = X.shape[0]

        if default_config.resume_from_checkpoint:
            config["checkpoint_path"] = train.get_model_save_path(
                config,
                checkpoint_epoch=default_config.checkpoint_epoch,
            )
        config["run_name"] = run_name
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])  # if using GPU

        model = models.TwoLayerNet(
            group_size=config["group_size"],
            hidden_size=config["hidden_factor"] * config["group_size"],
            nonlinearity="square",
            init_scale=config["init_scale"],
            output_scale=1e0,
        )
        model = model.to(device)
        loss = nn.MSELoss()

        if config["optimizer_name"] == "Adam":
            optimizer = optim.Adam(
                model.parameters(), lr=config["lr"], betas=(config["mom"], 0.999)
            )
        elif config["optimizer_name"] == "SGD":
            optimizer = optim.SGD(
                model.parameters(), lr=config["lr"], momentum=config["mom"]
            )
        elif config["optimizer_name"] == "PerNeuronScaledSGD":
            optimizer = PerNeuronScaledSGD(model, lr=config["lr"])
        else:
            raise ValueError(
                f"Unknown optimizer: {config['optimizer_name']}. Expected one of ['Adam', 'SGD', 'PerNeuronScaledSGD']."
            )

        print("Starting training...")
        loss_history, _, param_history = train.train(
            config,
            model,
            dataloader,
            loss,
            optimizer,
        )

        print("Training Complete. Generating plots...")

        loss_plot = plot.plot_loss_curve(
            loss_history,
            template_power,
            save_path=config["model_save_dir"] + f"loss_plot_{run_name}.svg",
            show=False,
        )

        power_over_training_plot = plot.plot_training_power_over_time(
            template_power,
            model,
            device,
            param_history,
            X,
            config["group_name"],
            save_path=config["model_save_dir"]
            + f"power_over_training_plot_{run_name}.svg",
            show=False,
            logscale=config["power_logscale"],
        )
        print(
            f"loss plot and power over training plot saved to {config['model_save_dir']}"
            f" at loss_plot_{run_name}.svg and power_over_training_plot_{run_name}.svg"
        )
        neuron_weights_plot = plot.plot_neuron_weights(
            config,
            model,
            neuron_indices=None,
        )

        model_predictions_plot = plot.plot_model_outputs(
            config["group_name"], config["group_size"], model, X, Y, idx=13
        )

        wandb.log(
            {
                "loss_plot": wandb.Image(loss_plot),
                # "irreps_plot": wandb.Image(irreps_plot),
                "power_over_training_plot": wandb.Image(power_over_training_plot),
                "neuron_weights_plot": wandb.Image(neuron_weights_plot),
                "model_predictions_plot": wandb.Image(model_predictions_plot),
            }
        )

        print("Plots generated and logged to wandb.")
        if config["group_name"] not in ("cnxcn", "cn"):
            print(
                f"With irreps' sizes:\n {[irrep.size for irrep in config['group'].irreps()]}"
            )

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
        init_scale,
        hidden_factor,
        seed,
        lr,
        mom,
        optimizer_name,
        batch_size,
        epochs,
        powers,
    ) in itertools.product(
        default_config.init_scale[default_config.group_name],
        default_config.hidden_factor,
        default_config.seed,
        default_config.lr[default_config.group_name],
        default_config.mom,
        default_config.optimizer_name,
        default_config.batch_size,
        default_config.epochs,
        default_config.powers[default_config.group_name],
    ):
        group_name = default_config.group_name

        main_config = {
            "group_name": group_name,
            "init_scale": init_scale,
            "run_start_time": run_start_time,
            "hidden_factor": hidden_factor,
            "seed": seed,
            "lr": lr,
            "mom": mom,
            "optimizer_name": optimizer_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "verbose_interval": default_config.verbose_interval,
            "run_start_time": run_start_time,
            "model_save_dir": default_config.model_save_dir,
            "powers": powers,
            "dataset_fraction": default_config.dataset_fraction[group_name],
            "power_logscale": default_config.power_logscale,
            "resume_from_checkpoint": default_config.resume_from_checkpoint,
            "checkpoint_interval": default_config.checkpoint_interval,
            "checkpoint_path": None,
        }

        if group_name == "cnxcn":
            for (
                image_length,
            ) in itertools.product(
                default_config.image_length,
            ):
                group_size = image_length * image_length
                main_config["group_size"] = group_size
                main_config["image_length"] = image_length
                main_config["dataset_fraction"] = default_config.dataset_fraction[
                    "cnxcn"
                ]
                main_config["fourier_coef_diag_values"] = main_config["powers"]
                main_run(main_config)

        elif group_name == "cn":
            for (
                group_n,
            ) in itertools.product(
                default_config.group_n
            ):
                main_config["group_size"] = group_n
                main_config["group_n"] = group_n
                main_config["dataset_fraction"] = default_config.dataset_fraction[
                    "cn"
                ]
                main_config["fourier_coef_diag_values"] = main_config["powers"]
                main_run(main_config)

        elif group_name == "octahedral":
            group = Octahedral()
            group_size = group.order()
            irreps = group.irreps()
            irrep_dims = [ir.size for ir in irreps]
            print(f"Running for group: {group_name}{group_n} with irrep dims {irrep_dims}")
            main_config["group"] = group
            main_config["group_size"] = group_size
            main_config["fourier_coef_diag_values"] = [
                np.sqrt(group_size * p / dim**2)
                for p, dim in zip(main_config["powers"], irrep_dims)
            ]
            main_run(main_config)

        elif group_name == "A5":
            group = Icosahedral()
            group_size = group.order()
            irreps = group.irreps()
            irrep_dims = [ir.size for ir in irreps]
            print(f"Running for group: {group_name}{group_n} with irrep dims {irrep_dims}")
            main_config["group"] = group
            main_config["group_size"] = group_size
            main_config["fourier_coef_diag_values"] = [
                np.sqrt(group_size * p / dim**2)
                for p, dim in zip(main_config["powers"], irrep_dims)
            ]
            main_run(main_config)

        else:
            for (
                group_n,
            ) in itertools.product(
                default_config.group_n
            ):
                if group_name == "dihedral":
                    group = DihedralGroup(group_n)
                else:
                    raise ValueError(
                        f"Unknown group_name: {group_name}. "
                        f"Expected one of ['dihedral', 'cn', 'octahedral']."
                    )
                group_size = group.order()
                irreps = group.irreps()
                irrep_dims = [ir.size for ir in irreps]
                print(f"Running for group: {group_name}{group_n} with irrep dims {irrep_dims}")
                main_config["group"] = group
                main_config["group_size"] = group_size
                main_config["group_n"] = group_n
                main_config["dataset_fraction"] = default_config.dataset_fraction[
                    group_name
                ]
                main_config["fourier_coef_diag_values"] = [
                    np.sqrt(group_size * p / dim**2)
                    for p, dim in zip(main_config["powers"], irrep_dims)
                ]
            main_run(main_config)


main()
