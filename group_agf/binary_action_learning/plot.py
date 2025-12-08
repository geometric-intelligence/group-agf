import matplotlib.pyplot as plt
import numpy as np
import collections
import copy
import torch

import gagf.group_learning.power as power

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
    """
    fig = plt.figure(figsize=(6, 6))

    alpha_values = template_power.loss_plateau_predictions()
    print(f"Plotting {len(alpha_values)} theoretical plateau lines.")

    for k, alpha in enumerate(alpha_values):
        print(f"Plotting alpha value {k}: {alpha}")
        plt.axhline(y=alpha, color="black", linestyle="--", linewidth=2, zorder=-2)

    if freq_colors is None:
        plt.plot(list(loss_history), lw=4)
    else:
        # For each alpha interval (between alpha_values[ai] and alpha_values[ai+1]),
        # collect contiguous loss values that fall into that interval, then plot each group in a different color.
        alpha_values = np.array(alpha_values)
        num_alpha_intervals = len(alpha_values) - 1
        grouped_epochs = [[] for _ in range(num_alpha_intervals + 1)]
        grouped_losses = [[] for _ in range(num_alpha_intervals + 1)]

        for epoch, loss in enumerate(loss_history):
            in_interval = False
            for ai in range(num_alpha_intervals):
                if (loss <= alpha_values[ai] + 1e-1) and (
                    loss > alpha_values[ai + 1] + 1e-1
                ):
                    grouped_epochs[ai].append(epoch)
                    grouped_losses[ai].append(loss)
                    in_interval = True
                    break
            # Handle losses <= to the smallest (last) alpha value - include them in last group
            if not in_interval and (loss <= alpha_values[-1] + 1e-1):
                grouped_epochs[-1].append(epoch)
                grouped_losses[-1].append(loss)

        print(
            f"Freq colors: {freq_colors}, number of alpha intervals: {num_alpha_intervals}"
        )
        for ai in range(num_alpha_intervals + 1):
            # Use the last color for the last group if not enough colors
            color = freq_colors[ai] if ai < len(freq_colors) else freq_colors[-1]
            if ai < num_alpha_intervals:
                print(
                    f"Color for alpha value {ai} (alpha={alpha_values[ai]}): {color}, number of points: {len(grouped_epochs[ai])}"
                )
            else:
                print(
                    f"Color for alpha values < {alpha_values[-1]}: {color}, number of points: {len(grouped_epochs[ai])}"
                )
            if grouped_epochs[ai]:  # only plot if group is non-empty
                plt.plot(grouped_epochs[ai], grouped_losses[ai], color=color, lw=4)

    plt.xscale("log")
    plt.yscale("log")

    # Create nice yticks for linear loss scale (not log)
    ymin, ymax = plt.ylim()
    # Compute some reasonable ticks between the min and max, at most 6 ticks
    yticks = np.linspace(ymin, ymax, num=6)

    # Format the tick labels nicely, decide if they should be int or float
    def format_tick(val):
        if abs(val) >= 1000 or abs(val) < 1e-2:
            return f"{val:.1e}"
        elif abs(val - int(val)) < 1e-6:
            return f"{int(val)}"
        else:
            return f"{val:.2f}"

    yticklabels = [format_tick(t) for t in yticks]
    plt.yticks(
        yticks,
        yticklabels,
        fontsize=FONT_SIZES["ticks"] if "ticks" in FONT_SIZES else 18,
    )

    plt.xlabel("Epochs", fontsize=FONT_SIZES["axes_label"])
    plt.ylabel("Train Loss", fontsize=FONT_SIZES["axes_label"])

    # Dynamically adjust xticks so none exceed available epochs
    tick_locs = [v for v in [100, 1000, 10000, 100000] if v < len(loss_history) - 1]
    tick_labels = [rf"$10^{{{int(np.log10(loc))}}}$" for loc in tick_locs]
    plt.xticks(
        tick_locs,
        tick_labels,
        fontsize=FONT_SIZES["ticks"] if "ticks" in FONT_SIZES else 18,
    )

    # Cut off y-axis slightly below the lowest alpha value for higher resolution
    y_min = alpha_values[-1] * 0.7 if alpha_values[-1] > 0 else 1e-8
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
    avg_power_history : list of ndarray (num_steps, num_freqs)
        List of average power spectra at each step.
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    template_power = template_power_object.power
    template_power = np.where(template_power < 1e-20, 0, template_power)
    flattened_template_power = template_power.flatten()
    if group_name == "znz_znz":
        power_idx = np.argsort(flattened_template_power)[-5:][::-1]
        model_powers_over_time, steps = power.model_power_over_time(
            group_name=group_name,
            group=None,
            model=model.to(device),
            param_history=param_history,
            model_inputs=X_tensor,
        )
    else:
        power_idx = np.argsort(flattened_template_power)[::-1]
        group = template_power_object.group
        model_powers_over_time, steps = power.model_power_over_time(
            group_name=group_name,
            group=group,
            model=model.to(device),
            param_history=param_history,
            model_inputs=X_tensor,
        )

    # Create a new figure for this plot
    fig = plt.figure(figsize=(6, 7))

    for i in power_idx:
        if group_name == "znz_znz":
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
            label = rf"$\xi = ({freq[i][0]:.1f}, {freq[i][1]:.1f})$"
        else:
            freqs = template_power_object.freqs
            label = rf"$\xi = {freqs[i]}  (dim={group.irreps()[i].size})$"
        plt.plot(steps, model_powers_over_time[:, i], color=f"C{i}", lw=3, label=label)
        plt.axhline(
            flattened_template_power[i],
            color=f"C{i}",
            linestyle="dotted",
            linewidth=2,
            alpha=0.5,
            zorder=-10,
        )

    # Labeling and formatting
    if logscale:
        plt.yscale("log")
        plt.ylim(1, 10000)
        plt.yticks(
            [1, 10, 100, 1000, 10000],
            [r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$"],
            fontsize=FONT_SIZES["ticks"] if "ticks" in FONT_SIZES else 18,
        )
    else:
        # Create nice yticks for linear loss scale (not log)
        ymin, ymax = plt.ylim()
        # Compute some reasonable ticks between the min and max, at most 6 ticks
        yticks = np.linspace(ymin, ymax, num=6)

        # Format the tick labels nicely, decide if they should be int or float
        def format_tick(val):
            if abs(val) >= 1000 or abs(val) < 1e-2:
                return f"{val:.1e}"
            elif abs(val - int(val)) < 1e-6:
                return f"{int(val)}"
            else:
                return f"{val:.2f}"

        yticklabels = [format_tick(t) for t in yticks]
        plt.yticks(
            yticks,
            yticklabels,
            fontsize=FONT_SIZES["ticks"] if "ticks" in FONT_SIZES else 18,
        )

    plt.xscale("log")
    plt.xlim(0, len(param_history) - 1)

    # Dynamically adjust xticks so none exceed available epochs
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
        # bbox_to_anchor=(1, 0.9),
        labelspacing=0.25,
    )
    ax = plt.gca()
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
    model : nn.Module
    group_size : int
        The value of group_size (weights are of size signal_len for 1D, img_len*img_len for 2D).
    group_name : str
    neuron_indices : list of int, optional
        List of neuron indices to plot. If None, random neurons are selected.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    show : bool, optional
        Whether to show the plot.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
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
            neuron_indices = random.sample(range(len(weights)), 10)
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
        if config["group_name"] is "znz_znz" or any(
            getattr(irrep, "size", 1) == 2 for irrep in config["group"].irreps()
        ):  # 2D irreps
            if w.shape[0] != config["group_size"]:
                raise ValueError(
                    f"Expected weight size img_len*img_len={config['group_size']}, got {w.shape[0]}"
                )
            if config["group_name"] == "znz_znz":
                img_len = int(np.sqrt(config["group_size"]))
                w_img = w.reshape(img_len, img_len)
            else:
                w_img = w.reshape(config["group_size"], -1)
            axs[i].imshow(w_img, cmap="viridis")
            axs[i].set_title(f"Neuron {idx}")
            axs[i].axis("off")
        else:  # 1D irreps
            if w.shape[0] != config["group_size"]:
                raise ValueError(
                    f"Expected weight size group_size={config['group_size']}, got {w.shape[0]}"
                )
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
            model_state_before = copy.deepcopy(model.state_dict())
            model.load_state_dict(param_history[s_idx])

        # Prepare output for all idx
        x = X[idx_list]
        y = Y[idx_list]

        if hasattr(model, "eval"):
            model.eval()
        output = model(x)

        # Restore model params if they were changed for plotting
        if param_history is not None and step is not None and isinstance(step, int):
            model.load_state_dict(model_state_before)

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


def plot_template(X, Y, template, template_minus_mean, indices, p, i=4):
    template_matrix = template.reshape((p, p))
    template_minus_mean_matrix = template_minus_mean.reshape((p, p))
    translation = indices[i]
    print(
        f"Translation for sample {i}: a=({translation[0][0]}, {translation[0][1]}), b=({translation[1][0]}, {translation[1][1]}), a+b=({translation[2][0]}, {translation[2][1]})"
    )

    Xi_a = X[i, 0].reshape((p, p))
    Xi_b = X[i, 1].reshape((p, p))
    Yi = Y[i].reshape((p, p))

    # Plot the original template and the matrices
    fig, axs = plt.subplots(1, 5, figsize=(16, 4))

    # Plot the original template
    im_template = axs[0].imshow(template_matrix, cmap="viridis")
    axs[0].set_title("Original Template")
    plt.colorbar(im_template, ax=axs[0])

    im_template_minus_mean = axs[1].imshow(template_minus_mean_matrix, cmap="viridis")
    axs[1].set_title("Template Minus Mean")
    plt.colorbar(im_template_minus_mean, ax=axs[1])

    # Extract translation values for titles
    a_x, a_y = translation[0]
    b_x, b_y = translation[1]
    q_x, q_y = translation[2]

    # Plot X[i][0] (a*template)
    im0 = axs[2].imshow(Xi_a, cmap="viridis")
    axs[2].set_title(f"X_a: Δx {a_x} Δy {a_y}")
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("x")
    plt.colorbar(im0, ax=axs[2])

    # Plot X[i][1] (b*template)
    im1 = axs[3].imshow(Xi_b, cmap="viridis")
    axs[3].set_title(f"X_b: Δx {b_x} Δy {b_y}")
    axs[3].set_xlabel("y")
    axs[3].set_ylabel("x")
    plt.colorbar(im1, ax=axs[3])

    # Plot Y[i] ((a+b)*template)
    im2 = axs[4].imshow(Yi, cmap="viridis")
    axs[4].set_title(f"Y: Δx {q_x} Δy {q_y}")
    axs[4].set_xlabel("y")
    axs[4].set_ylabel("x")
    plt.colorbar(im2, ax=axs[4])

    plt.tight_layout()
    plt.show()


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

    # Plot the irreducible representations (irreps) themselves, not their power.
    # Assume group.elements() returns the group elements in order, and each irrep is a function/class that can be called with a group element to get its representation matrix.

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


def plot_top_template_components(group_name, template_power, group_size, show=False):
    """Plot the top 5 Fourier components of the template."""
    if group_name == "znz_znz":
        return plot_top_template_components_2D(template_power, group_size, show)
    else:
        return plot_top_template_components_1D(template_power, group_size, show)


def plot_top_template_components_1D(template_power, group_size, show=False):
    """Plot the top 5 Fourier components of the template for 1D case.

    Parameters
    ----------
    template_power : class instance
        Instance of <>Power containing the template power spectrum.
    group_size : int
        group_size of D_p.
    """
    template_power = template_power.power

    # Get the indices of the top 5 components
    top_indices = np.argsort(template_power)[-5:][::-1]  # Indices of top 5 components

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    # Initialize cumulative spectrum
    cumulative_spectrum = np.zeros_like(template_power, dtype=complex)

    for i, idx in enumerate(top_indices):
        # Add current component to cumulative spectrum
        cumulative_spectrum[idx] = np.sqrt(
            template_power[idx] * group_size
        )  # Scale back to amplitude
        if idx != 0 and idx != template_power.shape[0] - 1:
            cumulative_spectrum[-idx] = np.conj(
                cumulative_spectrum[idx]
            )  # Add negative frequency component if not DC or Nyquist

        # Convert cumulative spectrum to spatial domain
        spatial_component = np.fft.ifft(cumulative_spectrum).real

        # Create title showing which components are included
        components_list = []
        for j in range(i + 1):
            comp_idx = top_indices[j]
            components_list.append(f"{comp_idx}")

        title = f"Components: {', '.join(components_list)}\nNew: {idx} Power: {template_power[idx]:.4f}"

        # Roll the spatial component by half the signal length
        spatial_component_rolled = np.roll(
            spatial_component, spatial_component.shape[0] // 2
        )

        im = axs[i].plot(spatial_component_rolled)
        axs[i].set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_top_template_components_2D(template_power, group_size, show=False):
    """Plot the top 5 Fourier components of the template for 2D case.

    Parameters
    ----------
    template_power : class instance
        Instance of <>Power containing the template power spectrum.
    group_size : int
        group_size of Z/pZ x Z/pZ.
    """
    template_power = template_power.power

    # Flatten the power array and get the indices of the top 5 components
    template_power_flat = template_power.flatten()
    top_indices = np.argsort(template_power_flat)[-5:][
        ::-1
    ]  # Indices of top 5 components

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    # Initialize cumulative spectrum
    cumulative_spectrum = np.zeros_like(template_power, dtype=complex)

    for i, idx in enumerate(top_indices):
        u = idx // template_power.shape[1]
        v = idx % template_power.shape[1]

        # Add current component to cumulative spectrum
        cumulative_spectrum[u, v] = np.sqrt(
            template_power[u, v] * group_size
        )  # Scale back to amplitude
        if v != 0 and v != template_power.shape[1] - 1:
            cumulative_spectrum[u, -v] = np.conj(
                cumulative_spectrum[u, v]
            )  # Add negative frequency component if not DC or Nyquist

        # Convert cumulative spectrum to spatial domain
        spatial_component = np.fft.ifft2(cumulative_spectrum).real

        # Create title showing which components are included
        components_list = []
        for j in range(i + 1):
            comp_idx = top_indices[j]
            comp_u = comp_idx // template_power.shape[1]
            comp_v = comp_idx % template_power.shape[1]
            components_list.append(f"({comp_u},{comp_v})")

        title = f"Components: {', '.join(components_list)}\nNew: ({u},{v}) Power: {template_power[u,v]:.4f}"

        # Roll the spatial component by half the image length in both x and y directions
        spatial_component_rolled = np.roll(
            spatial_component, spatial_component.shape[0] // 2, axis=0
        )
        spatial_component_rolled = np.roll(
            spatial_component_rolled, spatial_component.shape[1] // 2, axis=1
        )

        im = axs[i].imshow(spatial_component_rolled, cmap="viridis")
        axs[i].set_title(title)
        plt.colorbar(im, ax=axs[i])
    plt.tight_layout()
    if show:
        plt.show()
    return fig
