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

import gagf.group_learning.power as power


def plot_loss_curve(loss_history, template, save_path=None, show=False):
    """Plot loss curve over epochs.

    Parameters
    ----------
    loss_history : list of float
        List of loss values recorded at each epoch.
    template : np.ndarray
        The template used in training (used to calculate theoretical plateau lines for AGF).
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    fig = plt.figure(figsize=(8, 6))
    plt.plot(list(loss_history), lw=4)

    alpha_values = power.get_alpha_values(template)

    for k, alpha in enumerate(alpha_values):
        plt.axhline(y=alpha, color='black', linestyle='--', linewidth=2, zorder=-2)

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1e-2, 0.1])

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


def plot_training_power_over_time(template_2d, model, device, param_history, X_tensor, p, save_path=None, show=False):
    """Plot the power spectrum of the model's learned weights over time.

    Parameters
    ----------
    avg_power_history : list of ndarray (num_steps, num_freqs)
        List of average power spectra at each step.
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    row_freqs, column_freqs, template_power = power.get_power_2d(template_2d)
    freq = np.array([(row_freq, column_freq) for row_freq in row_freqs for column_freq in column_freqs])
    flattened_template_power = template_power.flatten()
    top_5_power_idx = np.argsort(flattened_template_power)[-5:][::-1]
    print(f"top_5_power_idx: {top_5_power_idx}")

    model_powers_over_time, steps = power.model_power_over_time(
        model.to(device),
        param_history=param_history,
        model_inputs = X_tensor,
        p=p,
    )

    # Create a new figure for this plot
    plt.figure(figsize=(10, 6))

    for i in top_5_power_idx:
        label = fr"$\xi = ({freq[i][0]:.1f}, {freq[i][1]:.1f})$"
        plt.plot(steps, model_powers_over_time[:, i], color=f"C{i}", lw=3, label=label)
        plt.axhline(flattened_template_power[i], color=f"C{i}", linestyle='dotted', linewidth=2, alpha=0.5, zorder=-10)

    # Labeling and formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0, len(param_history) - 1)
    plt.ylim(1e-7, 2)
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

    return plt.gcf()



def plot_neuron_weights(model, neuron_indices, p, save_path=None, show=False):
    """
    Plots the weights of specified neurons in the first linear layer of the model.
    
    Args:
        model: The neural network model (assumes first layer is nn.Linear).
        neuron_indices: List of neuron indices to plot.
        p: The value of p (weights are of size p*p).
        save_path: Optional path to save the figure.
    """

    # Get the first linear layer's weights
    first_layer = None
    for module in model.modules():
        if hasattr(module, 'weight') and hasattr(module, 'bias'):
            first_layer = module
            weights = first_layer.weight.detach().cpu().numpy()  # shape: (out_features, in_features)
            break
    if first_layer is None:
        # Support both nn.Linear and custom nn.Parameter-based models (like TwoLayerNet)
        if hasattr(model, 'W'):
            weights = model.W.detach().cpu().numpy()
        elif first_layer is not None:
            weights = first_layer.weight.detach().cpu().numpy()
        else:
            raise ValueError("No suitable weights found in model (neither nn.Linear nor custom nn.Parameter 'U').")

    # Determine number of rows and columns (max 5 per row)
    n_plots = len(neuron_indices)
    n_cols = min(5, n_plots)
    n_rows = (n_plots + 4) // 5  # integer division, round up

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    # axs is 2D if n_rows > 1, else 1D
    axs = np.array(axs).reshape(-1)  # flatten for easy indexing

    for i, idx in enumerate(neuron_indices):
        w = weights[idx]  # shape: (p*p,)
        if w.shape[0] != p*p:
            raise ValueError(f"Expected weight size p*p={p*p}, got {w.shape[0]}")
        # Reshape to (p, p)
        w_img = w.reshape(p, p)
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


def plot_model_outputs(model, X, Y, idx, num_samples=5, save_path=None):
    """Plot a training target vs the model output.

    Parameters
    ----------
    model : nn.Module
        The trained model.
    X : torch.Tensor
        Input data of shape (num_samples, input_size).
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
        x = X[idx].view(1, -1)
        y = Y[idx].view(1, -1)

        output = model(x)
        output_np = output.cpu().numpy().squeeze()
        target_np = y.cpu().numpy().squeeze() if hasattr(y, 'cpu') else y.numpy().squeeze()

        # Try to infer image size if possible
        # Ensure x, output, and target are on CPU and numpy arrays for plotting
        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.array(x)
        if torch.is_tensor(output):
            output_np = output.detach().cpu().numpy().squeeze()
        # output_np already defined above, but ensure it's on CPU and numpy
        if torch.is_tensor(y):
            target_np = y.detach().cpu().numpy().squeeze()
        else:
            target_np = np.array(y).squeeze()

        # Infer image size
        image_size = int(np.sqrt(x_np.shape[-1] // 2)) if x_np.shape[-1] % 2 == 0 else int(np.sqrt(x_np.shape[-1]))
        print(image_size)
        print(x_np.shape)
        print(output_np.shape)
        print(target_np.shape)
        x_np = x_np.squeeze()

        fig, axs = plt.subplots(1, 4, figsize=(15, 3), sharey=True)
        axs[0].imshow(x_np[:image_size*image_size].reshape(image_size, image_size))
        axs[0].set_title('Input 1')
        axs[1].imshow(x_np[image_size*image_size:].reshape(image_size, image_size))
        axs[1].set_title('Input 2')
        axs[2].imshow(output_np.reshape(image_size, image_size))
        axs[2].set_title('Output')
        axs[3].imshow(target_np.reshape(image_size, image_size))
        axs[3].set_title('Target')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close(fig)


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


def plot_top_template_components(template_flat, p):
    """Plot the top 5 Fourier components of the template.

    Parameters
    ----------
    template : np.ndarray
        A flattened 2D array of shape (p, p) representing the template.
    p : int
        p in Z/pZ x Z/pZ. Number of elements per dimension in the 2D modular addition
    """
    template_2d = template_flat.reshape((p, p))

    _, _, template_power = power.get_power_2d(template_2d)

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
        cumulative_spectrum[u, v] = np.sqrt(template_power[u, v] * p * p)  # Scale back to amplitude
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
    plt.show()


