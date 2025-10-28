import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import collections.abc

import gagf.group_learning.power as power


def plot_loss_curve(loss_history, template_power, save_path=None, show=False):
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
    fig = plt.figure(figsize=(8, 6))
    plt.plot(list(loss_history), lw=4)

    alpha_values = template_power.alpha_values

    for k, alpha in enumerate(alpha_values):
        print(f"Plotting alpha value {k}: {alpha}")
        plt.axhline(y=alpha, color='black', linestyle='--', linewidth=2, zorder=-2)

    plt.xscale("log")
    plt.yscale("log")
 
    plt.xlabel('Epochs', fontsize=24)
    plt.ylabel('Train Loss', fontsize=24)

    style_axes(plt.gca())
    plt.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_training_power_over_time(template_power, model, device, param_history, X_tensor, group, save_path=None, logscale=False, show=False):
    """Plot the power spectrum of the model's learned weights over time.

    Parameters
    ----------
    template_power : class instance
        Instance of <>Power containing the template power spectrum.
    avg_power_history : list of ndarray (num_steps, num_freqs)
        List of average power spectra at each step.
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    template_power = template_power.power
    row_freqs, column_freqs = template_power.x_freqs, template_power.y_freqs
    freq = np.array([(row_freq, column_freq) for row_freq in row_freqs for column_freq in column_freqs])
    flattened_template_power = template_power.flatten()
    top_5_power_idx = np.argsort(flattened_template_power)[-5:][::-1]
    print(f"top_5_power_idx: {top_5_power_idx}")

    model_powers_over_time, steps = power.model_power_over_time(
        group=group,
        model=model.to(device),
        param_history=param_history,
        model_inputs=X_tensor,
    )

    # Create a new figure for this plot
    fig = plt.figure(figsize=(10, 6))

    for i in top_5_power_idx:
        label = fr"$\xi = ({freq[i][0]:.1f}, {freq[i][1]:.1f})$"
        plt.plot(steps, model_powers_over_time[:, i], color=f"C{i}", lw=3, label=label)
        plt.axhline(flattened_template_power[i], color=f"C{i}", linestyle='dotted', linewidth=2, alpha=0.5, zorder=-10)

    # Labeling and formatting
    if logscale:
        plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0, len(param_history) - 1)
    plt.xticks(
        [1000, 10000, 100000, len(param_history) - 1],
        [r'$10^3$', r'$10^4$', r'$10^5$', 'Final']
    )
    plt.ylabel("Power", fontsize=24)
    plt.xlabel("Epochs", fontsize=24)
    plt.legend(
        fontsize=14,
        title="Frequency",
        title_fontsize=16,
        loc='upper right',
        bbox_to_anchor=(1, 0.9),
        labelspacing=0.25
    )
    ax = plt.gca()
    style_axes(ax)
    plt.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig



def plot_neuron_weights(group, model, group_size, neuron_indices=None, save_path=None, show=False):
    """Plot the weights of specified neurons in the model."""
    if group == 'znz_znz':
        return plot_neuron_weights_2D(model, group_size, neuron_indices, save_path, show)
    elif group == 'dihedral':
        return plot_neuron_weights_1D(model, group_size, neuron_indices, save_path, show)


def plot_neuron_weights_1D(model, group_size, neuron_indices=None, save_path=None, show=False):
    """Plot the weights of specified neurons in the last linear layer of the model.

    Parameters
    ----------
    model : nn.Module
    neuron_indices : list of int, optional
        List of neuron indices to plot. If None, random neurons are selected.
    group_size : int
        The value of group_size (weights are of size signal_len).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    """
    # Get the last linear layer's weights
    last_layer = None
    # Reverse list of modules to get the last nn.Linear if present
    modules = list(model.modules())
    for module in reversed(modules):
        if hasattr(module, 'weight') and hasattr(module, 'bias'):
            last_layer = module
            weights = last_layer.weight.detach().cpu().numpy()  # shape: (out_features, in_features)
            break
    if last_layer is None:
        # Support both nn.Linear and custom nn.Parameter-based models (like TwoLayerNet)
        if hasattr(model, 'U'):
            weights = model.U.detach().cpu().numpy()
        elif last_layer is not None:
            weights = last_layer.weight.detach().cpu().numpy()
        else:
            raise ValueError("No suitable weights found in model (neither nn.Linear nor custom nn.Parameter 'U').")

    if neuron_indices is None:
        if len(weights) <= 16:
            neuron_indices = list(range(len(weights)))
        else:
            neuron_indices = random.sample(range(len(weights)), 10)

    if isinstance(neuron_indices, int):
        neuron_indices = [neuron_indices]
        
    # Determine number of rows and columns (max 5 per row)
    n_plots = len(neuron_indices)
    n_cols = min(5, n_plots)
    n_rows = (n_plots + 4) // 5  # integer division, round up
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    # axs is 2D if n_rows > 1, else 1D
    axs = np.array(axs).reshape(-1)  # flatten for easy indexing

    for i, idx in enumerate(neuron_indices):
        w = weights[idx]  # shape: (2*p,)
        if w.shape[0] != group_size:
            raise ValueError(f"Expected weight size group_size={group_size}, got {w.shape[0]}")
        axs[i].plot(np.arange(group_size), w, lw=2)
        axs[i].set_title(f'Neuron {idx}')
        axs[i].set_xlabel('Input Index')
        axs[i].set_ylabel('Weight Value')
    # Hide any unused subplots
    for j in range(len(neuron_indices), len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return fig




def plot_neuron_weights_2D(model, group_size, neuron_indices=None, save_path=None, show=False):
    """
    Plots the weights of specified neurons in the last linear layer of the model.
    
    Args:
        model: The neural network model (assumes first layer is nn.Linear).
        neuron_indices: List of neuron indices to plot.
        group_size: The value of group_size (weights are of size img_len*img_len).
        save_path: Optional path to save the figure.
    """        
    # Get the last linear layer's weights
    last_layer = None
    # Reverse list of modules to get the last nn.Linear if present
    modules = list(model.modules())
    for module in reversed(modules):
        if hasattr(module, 'weight') and hasattr(module, 'bias'):
            last_layer = module
            weights = last_layer.weight.detach().cpu().numpy()  # shape: (out_features, in_features)
            break
    if last_layer is None:
        # Support both nn.Linear and custom nn.Parameter-based models (like TwoLayerNet)
        if hasattr(model, 'U'):
            weights = model.U.detach().cpu().numpy()
        elif last_layer is not None:
            weights = last_layer.weight.detach().cpu().numpy()
        else:
            raise ValueError("No suitable weights found in model (neither nn.Linear nor custom nn.Parameter 'U').")

    if neuron_indices is None:
        if len(weights) <= 16:
            neuron_indices = list(range(len(weights)))
        else:
            neuron_indices = random.sample(range(len(weights)), 10)

    if isinstance(neuron_indices, int):
        neuron_indices = [neuron_indices]
        
    # Determine number of rows and columns (max 5 per row)
    n_plots = len(neuron_indices)
    n_cols = min(5, n_plots)
    n_rows = (n_plots + 4) // 5  # integer division, round up
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    # axs is 2D if n_rows > 1, else 1D
    axs = np.array(axs).reshape(-1)  # flatten for easy indexing

    for i, idx in enumerate(neuron_indices):
        w = weights[idx]  
        if w.shape[0] != group_size:
            raise ValueError(f"Expected weight size img_len*img_len={group_size}, got {w.shape[0]}")
        img_len = int(np.sqrt(group_size))        
        w_img = w.reshape(img_len, img_len)
        axs[i].imshow(w_img, cmap='viridis')
        axs[i].set_title(f'Neuron {idx}')
        axs[i].axis('off')
    # Hide any unused subplots
    for j in range(len(neuron_indices), len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    return fig


def plot_model_outputs(group, group_size, model, X, Y, idx, save_path=None, show=False):
    """Plot a training target vs the model output."""
    if group == 'znz_znz':
        return plot_model_outputs_2D(group_size, model, X, Y, idx, save_path, show)
    elif group == 'dihedral':
        return plot_model_outputs_1D(group_size, model, X, Y, idx, save_path, show)


def plot_model_outputs_1D(group_size, model, X, Y, idx, save_path=None, show=False):
    """Plot a training target vs the model output in 1D case.

    Parameters
    ----------
    group_size : int
        The value of group_size.
    model : nn.Module
        The trained model.
    X : torch.Tensor
        Input data of shape (num_samples, 2, input_size). 2 images.
    Y : torch.Tensor
        Target data of shape (num_samples, output_size).
    idx : int
        Index of the data sample to plot target vs output for.
    num_samples : int
        Total number of samples in the dataset.
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    with torch.no_grad():
        # Accept single int or list/array of ints for idx
        idx_list = idx if isinstance(idx, collections.abc.Sequence) and not isinstance(idx, str) else [idx]
        n_samples = len(idx_list)

        # Prepare output for all idx
        # Model may only accept batch input, so collect batch inputs
        x = X[idx_list]   # (n_samples, input_size)
        y = Y[idx_list]   # (n_samples, output_size)
        print(f"x shape: {x.shape}, y shape: {y.shape}")

        if hasattr(model, 'eval'):
            model.eval()
        output = model(x)   # (n_samples, output_size)

        # Convert tensors to numpy arrays
        def to_numpy(t):
            if torch.is_tensor(t):
                return t.detach().cpu().numpy()
            return np.array(t)

        y_np = to_numpy(y)
        output_np = to_numpy(output)

        # Plot for each index
        fig, axs = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples), sharey=True,
                                squeeze=False)

        for row, (output_item, y_item) in enumerate(zip(output_np, y_np)):
            # ensure all items are 1d or flat
            y_item = np.squeeze(y_item)
            output_item = np.squeeze(output_item)

            axs[row, 0].plot(np.arange(group_size), output_item, lw=2)
            axs[row, 0].set_title('Output')

            axs[row, 1].plot(np.arange(group_size), y_item, lw=2)
            axs[row, 1].set_title('Target')
            for col in range(2):
                axs[row, col].set_xlabel('Index')
                axs[row, col].set_ylabel('Value')

        fig.suptitle(f"Model Inputs, Outputs, and Targets at index {idx}", fontsize=20)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close(fig)
    return fig



def plot_model_outputs_2D(group_size, model, X, Y, idx, save_path=None, show=False):
    """Plot a training target vs the model output in 2D case.

    Parameters
    ----------
    group_size : int
        The value of group_size.
    model : nn.Module
        The trained model.
    X : torch.Tensor
        Input data of shape (num_samples, 2, input_size). 2 images.
    Y : torch.Tensor
        Target data of shape (num_samples, output_size).
    idx : int
        Index of the data sample to plot target vs output for.
    num_samples : int
        Total number of samples in the dataset.
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """    
    with torch.no_grad():
        # Accept single int or list/array of ints for idx
        idx_list = idx if isinstance(idx, collections.abc.Sequence) and not isinstance(idx, str) else [idx]
        n_samples = len(idx_list)

        # Prepare output for all idx
        # Model may only accept batch input, so collect batch inputs
        x = X[idx_list]   # (n_samples, input_size)
        y = Y[idx_list]   # (n_samples, output_size)
        print(f"x shape: {x.shape}, y shape: {y.shape}")

        if hasattr(model, 'eval'):
            model.eval()
        output = model(x)   # (n_samples, output_size)

        # Convert tensors to numpy arrays
        def to_numpy(t):
            if torch.is_tensor(t):
                return t.detach().cpu().numpy()
            return np.array(t)

        x_np = to_numpy(x)
        y_np = to_numpy(y)
        output_np = to_numpy(output)

        image_size = int(np.sqrt(group_size))
        # If inputs are 2*group_size (concatenated two images) shape, split accordingly
        input_flat_dim = x_np.shape[-1]
        split_point = input_flat_dim // 2

        # Plot for each index
        fig, axs = plt.subplots(n_samples, 4, figsize=(15, 3 * n_samples), sharey=True,
                                squeeze=False)

        for row, (x_item, output_item, y_item) in enumerate(zip(x_np, output_np, y_np)):
            # ensure all items are 1d or flat
            x_item = np.squeeze(x_item)
            y_item = np.squeeze(y_item)
            output_item = np.squeeze(output_item)
            # For input, show two images side by side if x_item has two images
            axs[row, 0].imshow(x_item[:split_point].reshape(image_size, image_size), cmap='viridis')
            axs[row, 0].set_title('Input 1')

            axs[row, 1].imshow(x_item[split_point:].reshape(image_size, image_size), cmap='viridis')
            axs[row, 1].set_title('Input 2')

            axs[row, 2].imshow(output_item.reshape(image_size, image_size), cmap='viridis')
            axs[row, 2].set_title('Output')

            axs[row, 3].imshow(y_item.reshape(image_size, image_size), cmap='viridis')
            axs[row, 3].set_title('Target')
            for col in range(4):
                axs[row, col].axis('off')

        fig.suptitle(f"Model Inputs, Outputs, and Targets at index {idx}", fontsize=20)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close(fig)

    return fig


def style_axes(ax, numyticks=5, numxticks=5, labelsize=24):
    # Y-axis ticks
    ax.tick_params(axis="y", which="both", bottom=True, top=False,
                labelbottom=True, left=True, right=False,
                labelleft=True, direction='out', length=7, width=1.5, pad=8, labelsize=labelsize)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=numyticks))
    
    # X-axis ticks
    ax.tick_params(axis="x", which="both", bottom=True, top=False,
                labelbottom=True, left=True, right=False,
                labelleft=True, direction='out', length=7, width=1.5, pad=8, labelsize=labelsize)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=numxticks))

    # Scientific notation formatting
    if ax.get_yscale() == 'linear':
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
    if ax.get_xscale() == 'linear':
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2))

    ax.xaxis.offsetText.set_fontsize(20)
    ax.grid()

    # Customize spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(3)


def plot_template(X, Y, template, template_minus_mean, indices, p, i=4):
    template_matrix = template.reshape((p, p))
    template_minus_mean_matrix = template_minus_mean.reshape((p, p))
    translation = indices[i]
    print(f"Translation for sample {i}: a=({translation[0][0]}, {translation[0][1]}), b=({translation[1][0]}, {translation[1][1]}), a+b=({translation[2][0]}, {translation[2][1]})")

    Xi_a = X[i, 0].reshape((p, p))
    Xi_b = X[i, 1].reshape((p, p))
    Yi = Y[i].reshape((p, p))

    # Plot the original template and the matrices
    fig, axs = plt.subplots(1, 5, figsize=(16, 4))

    # Plot the original template
    im_template = axs[0].imshow(template_matrix, cmap='viridis')
    axs[0].set_title("Original Template")
    plt.colorbar(im_template, ax=axs[0])

    im_template_minus_mean = axs[1].imshow(template_minus_mean_matrix, cmap='viridis')
    axs[1].set_title("Template Minus Mean")
    plt.colorbar(im_template_minus_mean, ax=axs[1])

    # Extract translation values for titles
    a_x, a_y = translation[0]
    b_x, b_y = translation[1]
    q_x, q_y = translation[2]

    # Plot X[i][0] (a*template)
    im0 = axs[2].imshow(Xi_a, cmap='viridis')
    axs[2].set_title(f"X_a: Δx {a_x} Δy {a_y}")
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("x")
    plt.colorbar(im0, ax=axs[2])

    # Plot X[i][1] (b*template)
    im1 = axs[3].imshow(Xi_b, cmap='viridis')
    axs[3].set_title(f"X_b: Δx {b_x} Δy {b_y}")
    axs[3].set_xlabel("y")
    axs[3].set_ylabel("x")
    plt.colorbar(im1, ax=axs[3])

    # Plot Y[i] ((a+b)*template)
    im2 = axs[4].imshow(Yi, cmap='viridis')
    axs[4].set_title(f"Y: Δx {q_x} Δy {q_y}")
    axs[4].set_xlabel("y")
    axs[4].set_ylabel("x")
    plt.colorbar(im2, ax=axs[4])

    plt.tight_layout()
    plt.show()


def plot_top_template_components(template_power, group_size, show=False):
    """Plot the top 5 Fourier components of the template.

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
    top_indices = np.argsort(template_power_flat)[-5:][::-1]  # Indices of top 5 components

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    
    # Initialize cumulative spectrum
    cumulative_spectrum = np.zeros_like(template_power, dtype=complex)
    
    for i, idx in enumerate(top_indices):
        u = idx // template_power.shape[1]
        v = idx % template_power.shape[1]
        
        # Add current component to cumulative spectrum
        cumulative_spectrum[u, v] = np.sqrt(template_power[u, v] * group_size)  # Scale back to amplitude
        if v != 0 and v != template_power.shape[1] - 1:
            cumulative_spectrum[u, -v] = np.conj(cumulative_spectrum[u, v])  # Add negative frequency component if not DC or Nyquist
        
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
        spatial_component_rolled = np.roll(spatial_component, spatial_component.shape[0]//2, axis=0)
        spatial_component_rolled = np.roll(spatial_component_rolled, spatial_component.shape[1]//2, axis=1)
        
        im = axs[i].imshow(spatial_component_rolled, cmap='viridis')
        axs[i].set_title(title)
        plt.colorbar(im, ax=axs[i])
    plt.tight_layout()
    if show:
        plt.show()
    return fig


