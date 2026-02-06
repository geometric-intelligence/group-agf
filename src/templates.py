import numpy as np
from skimage.transform import resize
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

from src.group_fourier_transform import (
    compute_group_inverse_fourier_transform,
)


def one_hot(p):
    """One-hot encode an integer value in R^p."""
    vec = np.zeros(p)
    vec[1] = 10

    zeroth_freq = np.mean(vec)
    vec = vec - zeroth_freq
    return vec


def fixed_cn_template(group_size, fourier_coef_mags):
    """Generate a fixed template for the 1D modular addition dataset.

    Parameters
    ----------
    group_size : int
        n in Cn. Number of elements in the 1D modular addition
    fourier_coef_mags : list of float
        Magnitudes of the Fourier coefficients to set. This list can have any length, and the
        coefficients will be assigned to frequency modes in increasing order:
        0, 1, 2, ..., n-1 (and then their negative counterparts to ensure a real-valued template)
        where 0 represents the zeroth frequency mode.
    Returns
    -------
    template : np.ndarray
        A 1D array of shape (group_size,) representing the template.
    """
    # Generate template array from Fourier spectrum
    spectrum = np.zeros(group_size, dtype=complex)

    spectrum[0] = fourier_coef_mags[0]  # Zeroth frequency component
    fourier_coef_mags = fourier_coef_mags[1:]  # Exclude zeroth frequency

    for i_mag, mag in enumerate(fourier_coef_mags):
        mode = i_mag + 1  # Frequency mode starts from 1
        spectrum[mode] = mag
        spectrum[-mode] = np.conj(mag)
        print("Setting mode:", mode, "with magnitude:", mag)

    # Generate signal from spectrum
    template = np.fft.ifft(spectrum).real

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def fixed_cnxcn_template(image_length, fourier_coef_mags):
    """Generate a fixed template for the 2D modular addition dataset.

    Note: Since our input is a flattened matrix, we should un-flatten
    the weights vectors to match the shape of the template when we visualize.

    Parameters
    ----------
    image_length : int
        image_length = n in Cn x Cn. Number of elements per dimension in the 2D modular addition
    fourier_coef_mags : list of float
        Magnitudes of the Fourier coefficients to set. This list can have any length, and the
        coefficients will be assigned to frequency modes in the following order:
        (0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,2), (3,0), (0,3), (3,3), ...
        (and then their negative counterparts to ensure a real-valued template)
        where (i,j) represents the frequency mode with frequency i in the first dimension

    Returns
    -------
    template : np.ndarray
        A flattened 2D array of shape (image_length, image_length) representing the template.
        After flattening, it will have shape (image_length*image_length,).
    """
    # Generate template array from Fourier spectrum
    spectrum = np.zeros((image_length, image_length), dtype=complex)

    spectrum[0, 0] = fourier_coef_mags[0]  # Zeroth frequency component
    fourier_coef_mags = fourier_coef_mags[1:]  # Exclude zeroth frequency

    def mode_selector(i_mag):
        i_mode = 1 + i_mag // 3
        mode_type = i_mag % 3
        if mode_type == 0:
            return (i_mode, 0)
        elif mode_type == 1:
            return (0, i_mode)
        else:
            return (i_mode, i_mode)

    i_mag = 0
    while i_mag < len(fourier_coef_mags):
        mode = mode_selector(i_mag)

        spectrum[mode[0], mode[1]] = fourier_coef_mags[i_mag]
        spectrum[-mode[0], -mode[1]] = np.conj(fourier_coef_mags[i_mag])
        print("Setting mode:", mode, "with magnitude:", fourier_coef_mags[i_mag])
        i_mag += 1

    # Generate signal from spectrum
    template = np.fft.ifft2(spectrum).real

    template = template.flatten()

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def fixed_group_template(group, fourier_coef_diag_values):
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
    assert len(fourier_coef_diag_values) == len(group.irreps()), (
        f"Number of Fourier coef. magnitudes on the diagonal {len(fourier_coef_diag_values)} must match number of irreps {len(group.irreps())}"
    )
    for i, irrep in enumerate(group.irreps()):
        diag_values = np.full(irrep.size, fourier_coef_diag_values[i], dtype=float)
        mat = np.zeros((irrep.size, irrep.size), dtype=float)
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
