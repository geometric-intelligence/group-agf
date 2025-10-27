import numpy as np
import os
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
import setcwd
from gagf.group_learning.group_fourier_transform import compute_group_inverse_fourier_transform


def one_hot2D(p):
    """One-hot encode an integer value in R^pxp.
    
    Parameters
    ----------
    p : int
        p in Z/pZ x Z/pZ. Number of elements in the 2D modular addition
    
    Returns
    -------
    mat : np.ndarray
        A flattened one-hot encoded matrix of shape (p*p).
    """
    mat = np.zeros((p,p))
    mat[0,0] = 1
    mat = mat.flatten()
    return mat

def generate_fixed_template_znz_znz(p):
    """Generate a fixed template for the 2D modular addition dataset.

    Note: Since our input is a flattened matrix, we should un-flatten
    the weights vectors to match the shape of the template when we visualize.
    
    Parameters
    ----------
    p : int
        p in Z/pZ x Z/pZ. Number of elements per dimension in the 2D modular addition

    Returns
    -------
    template : np.ndarray
        A flattened 2D array of shape (p, p) representing the template.
        After flattening, it will have shape (p*p,).
    """
    # Generate template array from Fourier spectrum
    spectrum = np.zeros((p,p), dtype=complex)

    # Three low-frequency bins with Gaussian-ish weights
    v1 = 2.0 # 2.0
    v2 = 1.0 #0.1 # make sure this is not too close to v1
    v3 = 0.7 # 0.7 #0.01

    # plot_set_template_components(v1, v2, v3, p)

    # Mode (1,0)
    spectrum[1,0] = v1
    spectrum[-1,0] = np.conj(v1) 

    # Mode (2,1)
    spectrum[0,1] = v2
    spectrum[0,-1] = np.conj(v2)

    # Mode (1,1)
    spectrum[1,1] = v3
    spectrum[-1,-1] = np.conj(v3)
    
    # Generate signal from spectrum
    template = np.fft.ifft2(spectrum).real

    template = template.flatten()

    return template


def generate_fixed_template_dihedral(group):
    """Generate a fixed template for a group, that has non-zero Fourier coefficients 
    only for a few irreps.
    
    Parameters
    ----------
    group : Group (escnn object)
        The group.

    Returns
    -------
    template : np.ndarray, shape=[group.order()]
        The template.
    """
     # TODO: This only works for D3 for now.
    # Generate template array from Fourier spectrum
    spectrum = [np.array([[10.]]), np.array([[10.]]), np.array([[5., 5.], [5., 5.]])]
    # Generate signal from spectrum
    template = compute_group_inverse_fourier_transform(group, spectrum)

    return template



def mnist_template(p, digit=0, sample_idx=0, random_state=42):
    """Generate a template from the MNIST dataset, resized to p x p, for a specified digit.

    Parameters
    ----------
    p : int
        p in Z/pZ x Z/pZ. Number of elements per dimension in the 2D modular addition
    digit : int, optional
        The MNIST digit to use as a template (0-9). Default is 0.
    sample_idx : int, optional
        The index of the sample to use among the filtered digit images. Default is 0.
    random_state : int, optional
        Random seed for shuffling the digit images. Default is 42.

    Returns
    -------
    template : np.ndarray
        A flattened 2D array of shape (p, p) representing the template.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.utils import shuffle
    from skimage.transform import resize

    # Load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.values
    y = mnist.target.astype(int).values

    # Filter for the specified digit
    X_digit = X[y == digit]

    if X_digit.shape[0] == 0:
        raise ValueError(f"No samples found for digit {digit} in MNIST dataset.")

    # Shuffle and select the desired sample
    X_digit = shuffle(X_digit, random_state=random_state)
    if sample_idx >= X_digit.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} is out of bounds for digit {digit} (found {X_digit.shape[0]} samples).")
    sample = X_digit[sample_idx].reshape(28, 28)

    # Resize to p x p
    sample_resized = resize(sample, (p, p), anti_aliasing=True)

    # Normalize to [0, 1]
    sample_resized = (sample_resized - np.min(sample_resized)) / (np.max(sample_resized) - np.min(sample_resized))

    template = sample_resized.flatten()

    return template


def modular_addition_dataset_2d(p, template, fraction=0.3, random_state=42, save_path=None):
    """Generate a dataset for the 2D modular addition operation.

    General idea: We are generating a dataset where each sample consists of 
    two inputs (a*template and b*template) and an output (a*b)*template,
    where a, b \in Z/pZ x Z/pZ. The template is a flattened 2D array
    representing the modular addition operation in a 2D space.

    Each element X_i will contain the template with a different a_i, b_i, and
    in fact X contains the template at all possible a, b shifts.
    The output Y_i will contain the template shifted by a_i*b_i.
    * refers to the composition of two group actions (but by an abuse of notation, 
    also refers to group action on the template. oops.)
    
    Parameters
    ----------
    p : int
        p in Z/pZ x Z/pZ. Number of elements per dimension in the 2D modular addition.
    template : np.ndarray
        A flattened 2D array of shape (p*p,) representing the template.
        This should be generated using `generate_fixed_template(p)`.
        
    Returns
    -------
    X : np.ndarray
        Input data of shape (p^4, 2, p*p). 
        2 inputs (a and b), each with shape (p*p,).
         is the total number of combinations of shifted a's and b's.
    Y : np.ndarray
        Output data of shape (p^4, p*p), where each sample is the result of the modular addition."""
    # Initialize data arrays
    X = np.zeros((p ** 4, 2, p * p))  # Shape: (p^4, 2, p*p)
    Y = np.zeros((p ** 4, p * p))     # Shape: (p^4, p*p)
    translations = np.zeros((p**4, 3, 2), dtype=int)
    
    # Generate the dataset
    idx = 0
    template_2d = template.reshape((p, p))
    for a_x in range(p):
        for a_y in range(p):
            for b_x in range(p):
                for b_y in range(p):
                    q_x = (a_x + b_x) % p
                    q_y = (a_y + b_y) % p
                    X[idx, 0, :] = np.roll(np.roll(template_2d, a_x, axis=0), a_y, axis=1).flatten()
                    X[idx, 1, :] = np.roll(np.roll(template_2d, b_x, axis=0), b_y, axis=1).flatten()
                    Y[idx, :] = np.roll(np.roll(template_2d, q_x, axis=0), q_y, axis=1).flatten()
                    translations[idx, 0, :] = (a_x, a_y)
                    translations[idx, 1, :] = (b_x, b_y)
                    translations[idx, 2, :] = (q_x, q_y)
                    idx += 1

    assert 0 < fraction <= 1.0, "fraction must be in (0, 1]"
    # Sample a subset of the dataset according to the specified fraction
    N = X.shape[0]
    n_sample = int(np.ceil(N * fraction))
    rng = np.random.default_rng(random_state)
    indices = rng.choice(N, size=n_sample, replace=False)  # indices of the sampled subset

    if save_path is not None:
        # Save the dataset at the specified path using numpy
        np.savez(save_path, X=X[indices], Y=Y[indices], translations=translations[indices])

    return X[indices], Y[indices], translations[indices]



def load_modular_addition_dataset_2d(p, template, fraction=0.3, random_state=42, template_type="mnist"):
    """
    Load the modular addition 2D dataset from a file if present. 
    Otherwise, generate it and save to the given path.

    Parameters
    ----------
    p : int
        Group size (Z/pZ x Z/pZ)
    template : np.ndarray
        The fixed template (flattened, shape (p*p,))
    fraction : float
        Fraction of the dataset to use
    random_state : int
        Random seed
    template_type : str
        Type of template used ("mnist" or "fixed")

    Returns
    -------
    X : np.ndarray
    Y : np.ndarray
    translations : np.ndarray
    """
    file_name = f'modular_addition_2d_dataset_type{template_type}_p{p}_fraction{fraction:.2f}.npz'

    root_dir = setcwd.get_root_dir()
    load_path = os.path.join(root_dir, 'gagf', 'group_learning', 'saved_datasets', file_name)

    if os.path.exists(load_path):
        data = np.load(load_path)
        X = data['X']
        Y = data['Y']
        translations = data['translations']
        return X, Y, translations
    else:
        X, Y, translations = modular_addition_dataset_2d(
            p,
            template,
            fraction=fraction,
            random_state=random_state,
            save_path=load_path
        )
        return X, Y, translations


def choose_template(p, template_type="mnist", digit=4):
    """Choose template based on type."""
    if template_type == "znz_znz_fixed":
        template = generate_fixed_template(p)
    elif template_type == "mnist":
        template = mnist_template(p, digit=digit)
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

    return X_tensor, Y_tensor, device