from typing_extensions import dataclass_transform
import numpy as np
import torch
import time
import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import gagf.group_learning.models as models
import gagf.group_learning.datasets as datasets
import gagf.group_learning.power as power
import gagf.group_learning.train as train
import gagf.group_learning.plot as plot

import wandb
import itertools
import logging
import default_config

from escnn.group import *
from gagf.group_learning.optimizer import PerNeuronScaledSGD

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

        if config["group_name"] == "znz_znz":
            template_power = power.ZnZPower2D(template)
        else:
            template_power = power.GroupPower(template, group=config["group"])

        # Format template power values to only 2 decimals
        formatted_power_list = [f"{x:.2e}" for x in template_power.power]
        print("Template power:\n", formatted_power_list)
        print(
            f"With irreps' sizes:\n {[irrep.size for irrep in config['group'].irreps()]}"
        )
        # raise Exception("Stop here to check the template power.")

        X, Y, device = datasets.move_dataset_to_device_and_flatten(X, Y, device=None)

        # Determine batch size: if 'full', set to all samples
        if config["batch_size"] == "full":
            config["batch_size"] = X.shape[0]
        if default_config.resume_from_checkpoint:
            config["checkpoint_path"] = train.get_model_save_path(
                config,
                default_config.checkpoint_epoch,
                default_config.checkpoint_run_name_to_load,
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
        loss_history, accuracy_history, param_history = train.train(
            config,
            model,
            dataloader,
            loss,
            optimizer,
        )

        print("Training Complete. Generating plots...")

        loss_plot = plot.plot_loss_curve(loss_history, template_power, show=False)
        # irreps_plot = plot.plot_irreps(config['group'], show=False)
        power_over_training_plot = plot.plot_training_power_over_time(
            template_power,
            model,
            device,
            param_history,
            X,
            config["group_name"],
            save_path=None,  # TODO: Save the plot here in svg once it works
            show=False,
            logscale=config["power_logscale"],
        )
        neuron_weights_plot = plot.plot_neuron_weights(
            config[
                "group_name"
            ],  # TODO: remove this, since the group_name can be accessed from the group object:
            # group.name (will give "D3") or group.__class__.__name__ (will give "DihedralGroup")
            config["group"],
            model,
            config[
                "group_size"
            ],  # TODO: remove this, since the group_size can be accessed from the group object: group.order()
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
        print("Template power:\n", formatted_power_list)
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
        group_name,
        init_scale,
        hidden_factor,
        seed,
        lr,
        mom,
        optimizer_name,
        batch_size,
        epochs,
        verbose_interval,
        checkpoint_interval,
    ) in itertools.product(
        default_config.group_name,
        default_config.init_scale,
        default_config.hidden_factor,
        default_config.seed,
        default_config.lr,
        default_config.mom,
        default_config.optimizer_name,
        default_config.batch_size,
        default_config.epochs,
        default_config.verbose_interval,
        default_config.checkpoint_interval,
    ):

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
            "verbose_interval": verbose_interval,
            "run_start_time": run_start_time,
            "model_save_dir": default_config.model_save_dir,
            "powers": default_config.powers[group_name],
            "power_logscale": default_config.power_logscale,
            "resume_from_checkpoint": default_config.resume_from_checkpoint,
            "checkpoint_interval": checkpoint_interval,
            "checkpoint_path": None,
        }

        if group_name == "znz_znz":
            for (
                # frequencies_to_learn,
                # mnist_digit,
                image_length,
                dataset_fraction,
            ) in itertools.product(
                # default_config.frequencies_to_learn,
                # default_config.mnist_digit,
                default_config.image_length,
                default_config.dataset_fraction["znz_znz"],
            ):
                group_size = image_length * image_length
                # main_config["mnist_digit"] = mnist_digit
                main_config["group_size"] = group_size
                main_config["image_length"] = image_length
                # main_config["frequencies_to_learn"] = frequencies_to_learn
                main_config["dataset_fraction"] = dataset_fraction
                main_run(main_config)

        elif group_name == "octahedral":
            group = Octahedral()
            group_size = group.order()
            main_config["group"] = group
            main_config["group_size"] = group_size
            main_config["dataset_fraction"] = default_config.dataset_fraction[
                "octahedral"
            ]
            main_run(main_config)

        elif group_name == "A5":
            group = Icosahedral()
            group_size = group.order()
            main_config["group"] = group
            main_config["group_size"] = group_size
            main_config["dataset_fraction"] = default_config.dataset_fraction["A5"]
            main_run(main_config)

        else:
            for (
                # signal_length_1d,
                group_n,
            ) in itertools.product(
                # default_config.signal_length_1d,
                default_config.group_n
            ):
                if group_name == "dihedral":
                    group = DihedralGroup(group_n)
                elif group_name == "cyclic":
                    group = CyclicGroup(group_n)
                else:
                    raise ValueError(
                        f"Unknown group_name: {group_name}. "
                        f"Expected one of ['dihedral', 'cyclic', 'octahedral']."
                    )
                group_size = group.order()
                main_config["group"] = group
                main_config["group_size"] = group_size
                main_config["group_n"] = group_n
                main_config["dataset_fraction"] = default_config.dataset_fraction[
                    group_name
                ]
            main_run(main_config)


main()
