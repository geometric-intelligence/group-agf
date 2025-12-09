import matplotlib.pyplot as plt
import numpy as np
import collections
import copy
import torch

import group_agf.binary_action_learning.power as power

FONT_SIZES = {"title": 30, "axes_label": 30, "tick_label": 30, "legend": 15}


def plot_loss_curve(
    loss_history, template_power, save_path=None, show=False, freq_colors=None
):
    """Plot loss curve over epochs.

    Parameters
    ----------
    loss_history : list of float
        List of loss values recorded at each epoch.
    template_power : class instance
        Used to calculate theoretical plateau lines for AGF.
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    show : bool, optional
        Whether to display the plot.
    freq_colors : list of str, optional
        List of colors (in format "C0, C1, etc.") to use for different frequency intervals. 
        If None, a single color is used for the entire loss curve.
    """
    fig = plt.figure(figsize=(6, 6))

    plateau_predictions = template_power.loss_plateau_predictions()
    print(f"Plotting {len(plateau_predictions)} theoretical plateau lines.")

    for k, alpha in enumerate(plateau_predictions):
        print(f"Plotting alpha value {k}: {alpha}")
        plt.axhline(y=alpha, color="black", linestyle="--", linewidth=2, zorder=-2)

    if freq_colors is None:
        plt.plot(list(loss_history), lw=4)
    else:
        plateau_predictions = np.array(plateau_predictions)
        num_alpha_intervals = len(plateau_predictions) - 1
        grouped_epochs = [[] for _ in range(num_alpha_intervals + 1)]
        grouped_losses = [[] for _ in range(num_alpha_intervals + 1)]

        for epoch, loss in enumerate(loss_history):
            in_interval = False
            for ai in range(num_alpha_intervals):
                if (loss <= plateau_predictions[ai] + 1e-1) and (
                    loss > plateau_predictions[ai + 1] + 1e-1
                ):
                    grouped_epochs[ai].append(epoch)
                    grouped_losses[ai].append(loss)
                    in_interval = True
                    break
            # Handle losses <= to the smallest (last) alpha value - include them in last group
            if not in_interval and (loss <= plateau_predictions[-1] + 1e-1):
                grouped_epochs[-1].append(epoch)
                grouped_losses[-1].append(loss)

        print(
            f"Freq colors: {freq_colors}, number of alpha intervals: {num_alpha_intervals}"
        )
        for ai in range(num_alpha_intervals + 1):
            color = freq_colors[ai] if ai < len(freq_colors) else freq_colors[-1]
            if ai < num_alpha_intervals:
                print(
                    f"Color for alpha value {ai} (alpha={plateau_predictions[ai]}): {color}, number of points: {len(grouped_epochs[ai])}"
                )
            else:
                print(
                    f"Color for alpha values < {plateau_predictions[-1]}: {color}, number of points: {len(grouped_epochs[ai])}"
                )
            if grouped_epochs[ai]:  # only plot if group is non-empty
                plt.plot(grouped_epochs[ai], grouped_losses[ai], color=color, lw=4)

    plt.xscale("log")
    plt.yscale("log")

    ymin, ymax = plt.ylim()
    yticks = np.linspace(ymin, ymax, num=6)
    yticklabels = [t for t in yticks]
    plt.yticks(
        yticks,
        yticklabels,
        fontsize=FONT_SIZES["ticks"] if "ticks" in FONT_SIZES else 18,
    )

    tick_locs = [v for v in [100, 1000, 10000, 100000] if v < len(loss_history) - 1]
    tick_labels = [rf"$10^{{{int(np.log10(loc))}}}$" for loc in tick_locs]
    plt.xticks(
        tick_locs,
        tick_labels,
        fontsize=FONT_SIZES["ticks"] if "ticks" in FONT_SIZES else 18,
    )

    plt.xlabel("Epochs", fontsize=FONT_SIZES["axes_label"])
    plt.ylabel("Train Loss", fontsize=FONT_SIZES["axes_label"])

    # Cut off y-axis slightly below the lowest alpha value for higher resolution
    y_min = plateau_predictions[-1] * 0.7 if plateau_predictions[-1] > 0 else 1e-8
    plt.ylim(bottom=y_min)
    plt.xlim(0, len(loss_history) + 100)

    plt.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_training_power_over_time(
    template_power_object,
    model,
    device,
    param_history,
    X_tensor,
    group_name,
    save_path=None,
    logscale=False,
    show=False,
    return_freq_colors=False,
):
    """Plot the power spectrum of the model's learned weights over time.

    Parameters
    ----------
    template_power_object : class instance
        Instance of <>Power containing the template power spectrum.
    model : nn.Module
        The trained model.
    device : torch.device
        Device to run computations on.
    param_history : list of dict
        List of parameter snapshots (as yielded by model.state_dict()) during training.
    X_tensor : torch.Tensor
        Input data tensor of shape (num_samples, ...).
    group_name : str
        Name of the group (should distinguish 'znz_znz').
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    logscale : bool, optional
        Whether to use logarithmic scale for y-axis.
    show : bool, optional
        Whether to display the plot.
    return_freq_colors : bool, optional
        Whether to return the frequency colors used in the plot 
        (to optionally coordinate with loss curve).
    """
    if group_name == "znz_znz":
        escnn_group = None
        row_freqs, column_freqs = (
                template_power_object.x_freqs,
                template_power_object.y_freqs,
            )
        freq = np.array(
            [
                (row_freq, column_freq)
                for row_freq in row_freqs
                for column_freq in column_freqs
            ]
        )
    else:
        escnn_group = template_power_object.group
        freq = template_power_object.freqs

    template_power = template_power_object.power
    template_power = np.where(template_power < 1e-20, 0, template_power)
    flattened_template_power = template_power.flatten()

    
    power_idx = np.argsort(flattened_template_power)[-5:][::-1]
    model_powers_over_time, steps = power.model_power_over_time(
        group_name=group_name,
        group=escnn_group,
        model=model.to(device),
        param_history=param_history,
        model_inputs=X_tensor,
    )
        
    fig = plt.figure(figsize=(6, 7))

    for i in power_idx:
        if group_name == "znz_znz":
            label = rf"$\xi = ({freq[i][0]:.1f}, {freq[i][1]:.1f})$"
        else:
            label = rf"$\xi = {freq[i]}  (dim={escnn_group.irreps()[i].size})$"
        plt.plot(steps, model_powers_over_time[:, i], color=f"C{i}", lw=3, label=label)
        plt.axhline(
            flattened_template_power[i],
            color=f"C{i}",
            linestyle="dotted",
            linewidth=2,
            alpha=0.5,
            zorder=-10,
        )

    ymin, ymax = plt.ylim()
    if logscale:
        plt.yscale("log")
        yticks = np.logspace(np.log10(max(ymin, 1e-8)), np.log10(ymax), num=6)
        yticklabels = [f"{t:.1e}" for t in yticks]
        plt.yticks(
            yticks,
            yticklabels,
            fontsize=FONT_SIZES["ticks"] if "ticks" in FONT_SIZES else 18,
        )
    else:
        yticks = np.linspace(ymin, ymax, num=6)
        yticklabels = [t for t in yticks]
        plt.yticks(
            yticks,
            yticklabels,
            fontsize=FONT_SIZES["ticks"] if "ticks" in FONT_SIZES else 18,
        )

    plt.xscale("log")
    plt.xlim(0, len(param_history) - 1)
    tick_locs = [v for v in [100, 1000, 10000, 100000] if v < len(param_history) - 1]
    tick_labels = [rf"$10^{{{int(np.log10(loc))}}}$" for loc in tick_locs]
    plt.xticks(
        tick_locs,
        tick_labels,
        fontsize=FONT_SIZES["ticks"] if "ticks" in FONT_SIZES else 18,
    )

    plt.ylabel("Power", fontsize=FONT_SIZES["axes_label"])
    plt.xlabel("Epochs", fontsize=FONT_SIZES["axes_label"])
    plt.legend(
        fontsize=FONT_SIZES["legend"],
        title="Frequency",
        title_fontsize=FONT_SIZES["legend"],
        loc="upper left",
        labelspacing=0.25,
    )
    plt.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    if return_freq_colors:
        freq_colors = [f"C{i}" for i in power_idx]
        return fig, freq_colors

    return fig


def plot_neuron_weights(
    config,
    model,
    neuron_indices=None,
    save_path=None,
    show=False,
):
    """
    Plot the weights of specified neurons in the last linear layer of the model.
    2D visualization (imshow) if group is 'znz_znz', otherwise 1D line plot.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing 'group_name' and 'group_size'.
    model : nn.Module
        The trained model.
    neuron_indices : list of int or int, optional
        Indices of neurons to plot. If None, randomly selects 10 neurons (or all if <=16).
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    show : bool, optional
        If True, display the plot window.
    """        
    # Get the last linear layer's weights
    last_layer = None
    modules = list(model.modules())
    for module in reversed(modules):
        if hasattr(module, "weight") and hasattr(module, "bias"):
            last_layer = module
            weights = last_layer.weight.detach().cpu().numpy()
            break
    if last_layer is None:
        if hasattr(model, "U"):
            weights = model.U.detach().cpu().numpy()
        elif last_layer is not None:
            weights = last_layer.weight.detach().cpu().numpy()
        else:
            raise ValueError(
                "No suitable weights found in model (neither nn.Linear nor custom nn.Parameter 'U')."
            )


    # Select neurons
    if neuron_indices is None:
        if len(weights) <= 16:
            neuron_indices = list(range(len(weights)))
        else:
            neuron_indices = np.random.choice(range(len(weights)), 10, replace=False)
    if isinstance(neuron_indices, int):
        neuron_indices = [neuron_indices]

    # Setup subplots
    n_plots = len(neuron_indices)
    n_cols = min(5, n_plots)
    n_rows = (n_plots + 4) // 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axs = np.array(axs).reshape(-1)

    for i, idx in enumerate(neuron_indices):
        w = weights[idx]
        if w.shape[0] != config["group_size"]:
            raise ValueError(
                f"Expected weight size group_size={config['group_size']}, got {weights.shape[0]}"
            )
        if config["group_name"] is "znz_znz" or any(
            getattr(irrep, "size", 1) == 2 for irrep in config["group"].irreps()
        ):  # 2D irreps
            if config["group_name"] == "znz_znz":
                img_len = int(np.sqrt(config["group_size"]))
                w_img = w.reshape(img_len, img_len)
            else:
                w_img = w.reshape(config["group_size"], -1)
            axs[i].imshow(w_img, cmap="viridis")
            axs[i].set_title(f"Neuron {idx}")
            axs[i].axis("off")
        else:  # 1D irreps
            axs[i].plot(np.arange(config["group_size"]), w, lw=2)
            axs[i].set_title(f"Neuron {idx}")
            axs[i].set_xlabel("Input Index")
            axs[i].set_ylabel("Weight Value")
    # Hide unused subplots
    for j in range(len(neuron_indices), len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_model_outputs(
    group_name,
    group_size,
    model,
    X,
    Y,
    idx,
    step=-1,
    param_history=None,
    save_path=None,
    show=False,
):
    """
    Plot a training target vs the model output, adapting plot style for znz_znz (2D) and other groups (1D).

    Parameters
    ----------
    group_name : object or str
        The group instance or name (should distinguish 'znz_znz').
    group_size : int
        The value of group_size.
    model : nn.Module
        The trained model.
    X : torch.Tensor
        Input data of shape (num_samples, ...) for the group.
    Y : torch.Tensor
        Target data of shape (num_samples, output_size).
    idx : int or list-like
        Index/indices of the data sample(s) to plot target vs output for.
    step : int, optional
        Snapshot step from param_history to visualize. If -1, uses current model parameters.
    param_history : list of dict, optional
        List of parameter snapshots (as yielded by model.state_dict()) during training.
        If not provided, uses the latest model state.
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    show : bool, optional
        If True, display the plot window.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting matplotlib figure handle.
    """
    with torch.no_grad():
        # Accept single int or list/array of ints for idx
        if isinstance(idx, collections.abc.Sequence) and not isinstance(idx, str):
            idx_list = list(idx)
        else:
            idx_list = [idx]
        n_samples = len(idx_list)

        # Restore model parameters for a specific step if param_history is provided
        if param_history is not None and step is not None and isinstance(step, int):
            # Clamp step index
            s_idx = step
            if s_idx < 0:
                s_idx = len(param_history) + s_idx
            s_idx = max(0, min(s_idx, len(param_history) - 1))
            model.load_state_dict(param_history[s_idx])

        # Prepare output for all idx
        x = X[idx_list]
        y = Y[idx_list]

        if hasattr(model, "eval"):
            model.eval()
        output = model(x)

        # Convert tensors to numpy arrays
        def to_numpy(t):
            if torch.is_tensor(t):
                return t.detach().cpu().numpy()
            return np.array(t)

        x_np = to_numpy(x)
        y_np = to_numpy(y)
        output_np = to_numpy(output)

        plot_is_2D = group_name == "znz_znz"

        # --- 2D plotting for znz_znz ---
        if plot_is_2D:
            image_size = int(np.sqrt(group_size))
            input_flat_dim = x_np.shape[-1]
            split_point = input_flat_dim // 2 if x_np.ndim == 2 else 0

            fig, axs = plt.subplots(
                n_samples, 4, figsize=(15, 3 * n_samples), sharey=True, squeeze=False
            )

            for row, (x_item, output_item, y_item) in enumerate(
                zip(x_np, output_np, y_np)
            ):
                # Flatten and squeeze to expected shapes
                x_item = np.squeeze(x_item)
                y_item = np.squeeze(y_item)
                output_item = np.squeeze(output_item)
                # For input, show two images side by side if x_item has two images
                axs[row, 0].imshow(
                    x_item[:split_point].reshape(image_size, image_size), cmap="viridis"
                )
                axs[row, 0].set_title("Input 1")

                axs[row, 1].imshow(
                    x_item[split_point:].reshape(image_size, image_size), cmap="viridis"
                )
                axs[row, 1].set_title("Input 2")

                axs[row, 2].imshow(
                    output_item.reshape(image_size, image_size), cmap="viridis"
                )
                axs[row, 2].set_title("Output")

                axs[row, 3].imshow(
                    y_item.reshape(image_size, image_size), cmap="viridis"
                )
                axs[row, 3].set_title("Target")
                for col in range(4):
                    axs[row, col].axis("off")

            suptitle_str = f"Model Inputs, Outputs, and Targets at index {idx}"
            if param_history is not None and step is not None and isinstance(step, int):
                suptitle_str += f" (step {s_idx})"
            fig.suptitle(suptitle_str, fontsize=FONT_SIZES["title"])
            plt.tight_layout()

        # --- 1D plotting for other groups ---
        else:
            fig, axs = plt.subplots(
                n_samples, 2, figsize=(12, 3 * n_samples), sharey=True, squeeze=False
            )

            for row, (output_item, y_item) in enumerate(zip(output_np, y_np)):
                # ensure all items are 1d or flat
                y_item = np.squeeze(y_item)
                output_item = np.squeeze(output_item)

                axs[row, 0].plot(np.arange(group_size), output_item, lw=2)
                axs[row, 0].set_title("Output")

                axs[row, 1].plot(np.arange(group_size), y_item, lw=2)
                axs[row, 1].set_title("Target")
                for col in range(2):
                    axs[row, col].set_xlabel("Index")
                    axs[row, col].set_ylabel("Value")

            suptitle_str = f"Model Outputs and Targets at index {idx}"
            if param_history is not None and step is not None and isinstance(step, int):
                suptitle_str += f" (step {s_idx})"
            fig.suptitle(suptitle_str, fontsize=FONT_SIZES["title"])
            plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

    return fig


def plot_irreps(group, show=False):
    """Plot the irreducible representations (irreps) of the group and their corresponding power in the template.

    Parameters
    ----------
    group : class instance
        The group for which the irreps are being plotted. Should have a method `get_irreps()` that returns a list of irreps.
    show : bool, optional
        Whether to display the plot immediately. If False, the plot is not shown but the figure object is returned.
    """
    irreps = group.irreps()

    group_elements = group.elements
    irreps = group.irreps()

    num_irreps = len(irreps)
    fig, axs = plt.subplots(1, num_irreps, figsize=(3 * num_irreps, 4), squeeze=False)
    axs = axs[0]

    for i, irrep in enumerate(irreps):
        # Evaluate irrep on all group elements
        matrices = [irrep(g) for g in group_elements]
        matrices = np.array(matrices)  # (num_elements, d, d) or (num_elements,) for 1D

        if matrices.ndim == 1 or (matrices.ndim == 2 and matrices.shape[1] == 1):
            # 1D irrep: plot as real line (vs. group element index)
            axs[i].plot(
                range(len(group_elements)), matrices.real, marker="o", label="Re"
            )
            if np.any(np.abs(matrices.imag) > 1e-10):
                axs[i].plot(
                    range(len(group_elements)), matrices.imag, marker="x", label="Im"
                )
            axs[i].set_title(f"Irrep {i}: {str(irrep)} (dim=1)")
            axs[i].set_xlabel("Group element idx")
            axs[i].set_ylabel("Irrep value")
            axs[i].legend()
        else:
            d = matrices.shape[1]
            num_group_elements = len(group_elements)
            num_irrep_entries = d * d
            irrep_matrix_entries = matrices.real.reshape(
                num_group_elements, num_irrep_entries
            )
            im = axs[i].imshow(irrep_matrix_entries, aspect="auto", cmap="viridis")
            axs[i].set_title(f"Irrep {i}: {str(irrep)} (size={d}x{d})")
            axs[i].set_xlabel("Flattened Irreps")
            axs[i].set_ylabel("Irrep(g)")
            plt.colorbar(im, ax=axs[i])
    fig.suptitle(
        "Irreducible Representations (matrix values for all group elements)",
        fontsize=FONT_SIZES["title"],
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig