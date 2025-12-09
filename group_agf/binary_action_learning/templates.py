import numpy as np
from skimage.transform import resize
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

from group_agf.binary_action_learning.group_fourier_transform import \
    compute_group_inverse_fourier_transform


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
    v1 = 12.0  # 2.0 10.0
    v2 = 10.0  # 0.1 # make sure this is not too close to v1 3.0
    v3 = 8.0  # 0.7 #0.01 . 7.0
    v4 = 6.0
    v5 = 4.0

    # Mode (1,0)
    spectrum[1, 0] = v1
    spectrum[-1, 0] = np.conj(v1)

    # Mode (2,1)
    spectrum[0, 1] = v2
    spectrum[0, -1] = np.conj(v2)

    # Mode (1,1)
    spectrum[1, 1] = v3
    spectrum[-1, -1] = np.conj(v3)

    # Mode (2,0)
    spectrum[2, 0] = v4
    spectrum[-2, 0] = np.conj(v4)

    # Mode (0,2)
    spectrum[0, 2] = v5
    spectrum[0, -2] = np.conj(v5)

    # Generate signal from spectrum
    template = np.fft.ifft2(spectrum).real

    template = template.flatten()

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def generate_fixed_group_template(group, seed, fourier_coef_diag_values):
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
    spectrum = []
    assert len(fourier_coef_diag_values) == len(
        group.irreps()
    ), f"Number of Fourier coef. magnitudes on the diagonal {len(fourier_coef_diag_values)} must match number of irreps {len(group.irreps())}"
    for i, irrep in enumerate(group.irreps()):
        diag_values = np.full(irrep.size, fourier_coef_diag_values[i], dtype=float)
        # Create a random full rank matrix with unique diagonal entries
        mat = np.zeros((irrep.size, irrep.size), dtype=float)
        # print(f"diag_values for irrep {i} of dimension {dim} is: {diag_values}")
        np.fill_diagonal(mat, diag_values)
        print(f"mat for irrep {i} of dimension {irrep.size} is:\n {mat}\n")

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
