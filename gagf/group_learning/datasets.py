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
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from skimage.transform import resize

from escnn.group import *

import gagf.group_learning.power as power
import setcwd
from gagf.group_learning.group_fourier_transform import (
    compute_group_inverse_fourier_transform,
)


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
    mat = np.zeros((p, p))
    mat[0, 0] = 1
    mat = mat.flatten()
    return mat


def generate_fixed_template_znz_znz(image_length):
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
    spectrum = np.zeros((image_length, image_length), dtype=complex)

    # Three low-frequency bins with Gaussian-ish weights
    v1 = 2.0  # 2.0
    v2 = 1.0  # 0.1 # make sure this is not too close to v1
    v3 = 0.7  # 0.7 #0.01

    # plot_set_template_components(v1, v2, v3, p)

    # Mode (1,0)
    spectrum[1, 0] = v1
    spectrum[-1, 0] = np.conj(v1)

    # Mode (2,1)
    spectrum[0, 1] = v2
    spectrum[0, -1] = np.conj(v2)

    # Mode (1,1)
    spectrum[1, 1] = v3
    spectrum[-1, -1] = np.conj(v3)

    # Generate signal from spectrum
    template = np.fft.ifft2(spectrum).real

    template = template.flatten()

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def generate_fixed_template_dihedral3(group):
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
    # spectrum = [np.array([[10.]]), np.array([[30.]]), np.array([[10., 10.], [10., 10.]])]
    # spectrum = [np.array([[20.]]), np.array([[10.]]), np.array([[5., 5.], [5., 5.]])]
    spectrum = [
        np.array([[50.0]]),
        np.array([[40.0]]),
        np.array([[5.0, 1.0], [1.0, 5.0]]),
    ]
    # Generate signal from spectrum
    template = compute_group_inverse_fourier_transform(group, spectrum)

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def generate_fixed_template_dihedral(N):
    """Generate a fixed template for the dihedral group D_N, that has non-zero Fourier coefficients
    only for a few irreps.

    DRAFT. UNTESTED.

    Parameters
    ----------
    N : int
        The order of the dihedral group D_N.

    Returns
    -------
    template : np.ndarray, shape=[group.order()]
        The mean centered template.
    """
    group = DihedralGroup(N)

    irreps = group.irreps()

    for i, irrep in enumerate(irreps):
        print(f"Irrep {i}: size {irrep.size}")

    for i, irrep in enumerate(irreps):
        print("Irrep", irrep)

    # Generate template array from Fourier spectrum
    if N % 2 == 1:
        n_1D_irreps = 2
        n_2D_irreps = int((N - 1) // 2)
    if N % 2 == 0:
        n_1D_irreps = 4
        n_2D_irreps = int(N / 2 - 1)

    spectrum = []

    for weight in np.linspace(50, 10, n_1D_irreps):
        spectrum.append(np.array([[weight]]))

    for a in np.linspace(50, 10, n_2D_irreps):
        b = 1
        mat = np.array([[a, b], [b, a]])
        spectrum.append(mat)

    # Generate signal from spectrum
    template = compute_group_inverse_fourier_transform(group, spectrum)

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template

# TODO: Design a template with a few irreps that more separation in power spectrum
def generate_fixed_group_template(group, seed):
    """Generate a fixed template for a group, that has non-zero Fourier coefficients
    only for a few irreps.

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    num_irreps : int
        Number of irreps to set non-zero Fourier coefficients for. (Default is 3.)

    Returns
    -------
    template : np.ndarray, shape=[group.order()]
        The mean centered template.
    """
    # Incorporate seed for reproducibility
    # rng = np.random.default_rng(seed)
    # Generate template array from Fourier spectrum
    spectrum = []
    num_1d_nonzero_irreps = 0
    num_multi_d_nonzero_irreps = 0
    max_num_1d_nonzero_irreps = 2
    powers_1d = [0.1, 1.]
    powers_2d = [100.]
    max_num_multi_d_nonzero_irreps = 1

    for i, irrep in enumerate(group.irreps()):
        dim = irrep.size
        diag_values = np.zeros(dim, dtype=float)

        if dim == 1 and num_1d_nonzero_irreps < max_num_1d_nonzero_irreps:
            diag_values[0] = powers_1d[num_1d_nonzero_irreps]
            num_1d_nonzero_irreps += 1

        elif dim > 1 and num_multi_d_nonzero_irreps < max_num_multi_d_nonzero_irreps:
            diag_values = np.full(dim, powers_2d[num_multi_d_nonzero_irreps], dtype=float)
            num_multi_d_nonzero_irreps += 1

        # Create a random full rank matrix with unique diagonal entries
        mat = np.zeros((dim, dim), dtype=float)
        print(f"diag_values for irrep {i} of dimension {dim} is: {diag_values}")
        np.fill_diagonal(mat, diag_values)
        print(f"mat for irrep {i} of dimension {dim} is: {mat}\n\n")

        spectrum.append(mat)

    # Generate signal from spectrum
    template = compute_group_inverse_fourier_transform(group, spectrum)

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def mnist_template(image_length, digit=0, sample_idx=0, random_state=42):
    """Generate a template from the MNIST dataset, resized to p x p, for a specified digit.

    Parameters
    ----------
    image_length : int
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
        A flattened 2D array of shape (image_length, image_length) representing the template.
    """
    # Load MNIST dataset
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.values
    y = mnist.target.astype(int).values

    # Filter for the specified digit
    X_digit = X[y == digit]

    if X_digit.shape[0] == 0:
        raise ValueError(f"No samples found for digit {digit} in MNIST dataset.")

    # Shuffle and select the desired sample
    X_digit = shuffle(X_digit, random_state=random_state)
    if sample_idx >= X_digit.shape[0]:
        raise IndexError(
            f"sample_idx {sample_idx} is out of bounds for digit {digit} (found {X_digit.shape[0]} samples)."
        )
    sample = X_digit[sample_idx].reshape(28, 28)

    # Resize to p x p
    sample_resized = resize(sample, (image_length, image_length), anti_aliasing=True)

    # Normalize to [0, 1]
    sample_resized = (sample_resized - np.min(sample_resized)) / (
        np.max(sample_resized) - np.min(sample_resized)
    )

    template = sample_resized.flatten()

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def group_dataset(group, template):
    """Generate a dataset of group elements acting on the template.

    Using the regular representation.

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    template : np.ndarray, shape=[group.order()]
        The template to generate the dataset from.

    Returns
    -------
    X : np.ndarray, shape=[group.order()**2, 2, group.order()]
        The dataset of group elements acting on the template.
    Y : np.ndarray, shape=[group.order()**2, group.order()]
        The dataset of group elements acting on the template.
    """

    # Initialize data arrays
    group_order = group.order()
    assert (
        len(template) == group_order
    ), "template must have the same length as the group order"
    n_samples = group_order**2
    X = np.zeros((n_samples, 2, group_order))
    Y = np.zeros((n_samples, group_order))
    regular_rep = group.representations["regular"]

    idx = 0
    for g1 in group.elements:
        for g2 in group.elements:
            g1_rep = regular_rep(g1)
            g2_rep = regular_rep(g2)
            g12_rep = g1_rep @ g2_rep

            X[idx, 0, :] = g1_rep @ template
            X[idx, 1, :] = g2_rep @ template
            Y[idx, :] = g12_rep @ template
            idx += 1

    return X, Y


def modular_addition_dataset_2d(
    p, template, fraction=0.3, random_state=42, save_path=None
):
    """Generate a dataset for the 2D modular addition operation.

    General idea: We are generating a dataset where each sample consists of
    two inputs (a*template and b*template) and an output (a*b)*template,
    where $a, b \in Z/pZ x Z/pZ$. The template is a flattened 2D array
    representing the modular addition operation in a 2D space.

    Each element $X_i$ will contain the template with a different $a_i$, $b_i$, and
    in fact $X$ contains the template at all possible $a$, $b$ shifts.
    The output $Y_i$ will contain the template shifted by $a_i*b_i$.
    * refers to the composition of two group actions (but by an abuse of notation,
    also refers to group action on the template.)

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
        Output data of shape (p^4, p*p), where each sample is the result of the modular addition.
    """
    # Initialize data arrays
    X = np.zeros((p**4, 2, p * p))  # Shape: (p^4, 2, p*p)
    Y = np.zeros((p**4, p * p))  # Shape: (p^4, p*p)
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
                    X[idx, 0, :] = np.roll(
                        np.roll(template_2d, a_x, axis=0), a_y, axis=1
                    ).flatten()
                    X[idx, 1, :] = np.roll(
                        np.roll(template_2d, b_x, axis=0), b_y, axis=1
                    ).flatten()
                    Y[idx, :] = np.roll(
                        np.roll(template_2d, q_x, axis=0), q_y, axis=1
                    ).flatten()
                    translations[idx, 0, :] = (a_x, a_y)
                    translations[idx, 1, :] = (b_x, b_y)
                    translations[idx, 2, :] = (q_x, q_y)
                    idx += 1

    assert 0 < fraction <= 1.0, "fraction must be in (0, 1]"
    # Sample a subset of the dataset according to the specified fraction
    N = X.shape[0]
    n_sample = int(np.ceil(N * fraction))
    rng = np.random.default_rng(random_state)
    indices = rng.choice(
        N, size=n_sample, replace=False
    )  # indices of the sampled subset

    if save_path is not None:
        # Save the dataset at the specified path using numpy
        np.savez(
            save_path, X=X[indices], Y=Y[indices], translations=translations[indices]
        )

    return X[indices], Y[indices], translations[indices]


def load_modular_addition_dataset_2d(
    image_length, template, fraction=0.3, random_state=42, template_type="mnist"
):
    """
    Load the modular addition 2D dataset from a file if present.
    Otherwise, generate it and save to the given path.

    Parameters
    ----------
    image_length : int
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
    X : np.ndarray (num_samples, 2, p*p)
    Y : np.ndarray (num_samples, p*p)
    translations : np.ndarray
    """
    file_name = f"modular_addition_2d_dataset_type{template_type}_p{image_length}_fraction{fraction:.2f}.npz"

    root_dir = setcwd.get_root_dir()
    load_path = os.path.join(
        root_dir, "gagf", "group_learning", "saved_datasets", file_name
    )

    if os.path.exists(load_path):
        data = np.load(load_path)
        X = data["X"]
        Y = data["Y"]
        translations = data["translations"]
        return X, Y, translations
    else:
        X, Y, translations = modular_addition_dataset_2d(
            image_length,
            template,
            fraction=fraction,
            random_state=random_state,
            save_path=load_path,
        )
        return X, Y, translations


def load_dataset(config):
    """Load dataset based on configuration."""

    if config["group_name"] == "znz_znz":
        # template = mnist_template(config["image_length"], digit=config["mnist_digit"])
        template = generate_fixed_template_znz_znz(config["image_length"])
        X, Y, _ = load_modular_addition_dataset_2d(
            config["image_length"],
            template,
            fraction=config["dataset_fraction"],
            random_state=config["seed"],
            # template_type="mnist",
            template_type="fixed",
        )
        return X, Y, template
    else:
        template = generate_fixed_group_template(config["group"], config["seed"])
        X, Y = group_dataset(config["group"], template)

        if config["dataset_fraction"] != 1.0:
            assert 0 < config["dataset_fraction"] <= 1.0, "fraction must be in (0, 1]"
            # Sample a subset of the dataset according to the specified fraction
            N = X.shape[0]
            n_sample = int(np.ceil(N * config["dataset_fraction"]))
            rng = np.random.default_rng(config["seed"])
            indices = rng.choice(
                N, size=n_sample, replace=False
            )  # indices of the sampled subset
            X = X[indices]
            Y = Y[indices]

        return X, Y, template


def move_dataset_to_device_and_flatten(X, Y, device=None):
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
    # Reshape X to (num_samples, 2*num_data_features), where num_data_features is inferred from len(X[0][0])
    num_data_features = len(X[0][0])
    X_flat = X.reshape(X.shape[0], 2 * num_data_features)
    Y_flat = Y.reshape(Y.shape[0], num_data_features)
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_flat, dtype=torch.float32)
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