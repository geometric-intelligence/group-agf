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

from . import theory


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


def plot_top_template_components(template_1d, p):
    """Plot the top 5 Fourier components of the template.

    Parameters
    ----------
    template : np.ndarray
        A flattened 2D array of shape (p, p) representing the template.
    p : int
        p in Z/pZ x Z/pZ. Number of elements per dimension in the 2D modular addition
    """
    template_2d = template_1d.reshape((p, p))

    freqs_u, freqs_v, power = theory.get_power_2d(template_2d)

    # Flatten the power array and get the indices of the top 5 components
    power_flat = power.flatten()
    top_indices = np.argsort(power_flat)[-5:][::-1]  # Indices of top 5 components

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    
    # Initialize cumulative spectrum
    cumulative_spectrum = np.zeros_like(power, dtype=complex)
    
    for i, idx in enumerate(top_indices):
        u = idx // power.shape[1]
        v = idx % power.shape[1]
        
        # Add current component to cumulative spectrum
        cumulative_spectrum[u, v] = np.sqrt(power[u, v] * p * p)  # Scale back to amplitude
        if v != 0 and v != power.shape[1] - 1:
            cumulative_spectrum[u, -v] = np.conj(cumulative_spectrum[u, v])  # Add negative frequency component if not DC or Nyquist
        
        # Convert cumulative spectrum to spatial domain
        spatial_component = np.fft.ifft2(cumulative_spectrum).real
        
        # Create title showing which components are included
        components_list = []
        for j in range(i + 1):
            comp_idx = top_indices[j]
            comp_u = comp_idx // power.shape[1]
            comp_v = comp_idx % power.shape[1]
            components_list.append(f"({comp_u},{comp_v})")
        
        title = f"Components: {', '.join(components_list)}\nNew: ({u},{v}) Power: {power[u,v]:.4f}"
        
        # Roll the spatial component by half the image length in both x and y directions
        spatial_component_rolled = np.roll(spatial_component, spatial_component.shape[0]//2, axis=0)
        spatial_component_rolled = np.roll(spatial_component_rolled, spatial_component.shape[1]//2, axis=1)
        
        im = axs[i].imshow(spatial_component_rolled, cmap='viridis')
        axs[i].set_title(title)
        plt.colorbar(im, ax=axs[i])
    plt.tight_layout()
    plt.show()


def plot_set_template_components(v1, v2, v3, p):
    spectrum = np.zeros((p,p), dtype=complex)

    # Mode (1,0)
    spectrum[1,0] = v1
    spectrum[-1,0] = np.conj(v1) 

    template_1 = np.fft.ifft2(spectrum).real

    # Mode (0,1)
    spectrum[0,1] = v2
    spectrum[0,-1] = np.conj(v2)

    template_2 = np.fft.ifft2(spectrum).real

    # Mode (1,1)
    spectrum[1,1] = v3
    spectrum[-1,-1] = np.conj(v3)
    
    # Generate signal from spectrum
    template_full = np.fft.ifft2(spectrum).real

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    im1 = axs[0].imshow(template_1, cmap='viridis')
    axs[0].set_title("Mode (1,0)")
    plt.colorbar(im1, ax=axs[0])
    im2 = axs[1].imshow(template_2, cmap='viridis')
    axs[1].set_title("Mode (1,0)+(0,1)")
    plt.colorbar(im2, ax=axs[1])
    im3 = axs[2].imshow(template_full, cmap='viridis')
    axs[2].set_title("Full Template")
    plt.colorbar(im3, ax=axs[2])
    plt.tight_layout()

    plt.show()