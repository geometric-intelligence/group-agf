import numpy as np
from skimage.transform import resize
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

import src.fourier as fourier

# ---------------------------------------------------------------------------
# Templates from the original templates.py (names kept as-is per plan)
# ---------------------------------------------------------------------------


def one_hot(p):
    """One-hot encode an integer value in R^p."""
    vec = np.zeros(p)
    vec[1] = 10

    zeroth_freq = np.mean(vec)
    vec = vec - zeroth_freq
    return vec


def fixed_cn(group_size, fourier_coef_mags):
    """Generate a fixed template for the 1D modular addition dataset.

    Parameters
    ----------
    group_size : int
        n in Cn. Number of elements in the 1D modular addition
    fourier_coef_mags : list of float
        Magnitudes of the Fourier coefficients to set.

    Returns
    -------
    template : np.ndarray
        A 1D array of shape (group_size,) representing the template.
    """
    spectrum = np.zeros(group_size, dtype=complex)

    spectrum[0] = fourier_coef_mags[0]
    fourier_coef_mags = fourier_coef_mags[1:]

    for i_mag, mag in enumerate(fourier_coef_mags):
        mode = i_mag + 1
        spectrum[mode] = mag
        spectrum[-mode] = np.conj(mag)
        print("Setting mode:", mode, "with magnitude:", mag)

    template = np.fft.ifft(spectrum).real

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def fixed_cnxcn(image_length, fourier_coef_mags):
    """Generate a fixed template for the 2D modular addition dataset.

    Parameters
    ----------
    image_length : int
        image_length = n in Cn x Cn.
    fourier_coef_mags : list of float
        Magnitudes of the Fourier coefficients to set.

    Returns
    -------
    template : np.ndarray
        A flattened 2D array of shape (image_length*image_length,).
    """
    spectrum = np.zeros((image_length, image_length), dtype=complex)

    spectrum[0, 0] = fourier_coef_mags[0]
    fourier_coef_mags = fourier_coef_mags[1:]

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

    template = np.fft.ifft2(spectrum).real

    template = template.flatten()

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def fixed_group(group, fourier_coef_diag_values):
    """Generate a fixed template for a group with non-zero Fourier coefficients for specific irreps.

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    fourier_coef_diag_values : list of float
        Diagonal values for each irrep's Fourier coefficient matrix.

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

    template = fourier.group_fourier_inverse(group, spectrum)

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def mnist(image_length, digit=0, sample_idx=0, random_state=42):
    """Generate a template from the MNIST dataset, resized to p x p.

    Parameters
    ----------
    image_length : int
        p in Z/pZ x Z/pZ.
    digit : int, optional
        The MNIST digit to use as a template (0-9).
    sample_idx : int, optional
        The index of the sample to use.
    random_state : int, optional
        Random seed for shuffling.

    Returns
    -------
    template : np.ndarray
        A flattened 2D array of shape (image_length*image_length,).
    """
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.values
    y = mnist.target.astype(int).values

    X_digit = X[y == digit]

    if X_digit.shape[0] == 0:
        raise ValueError(f"No samples found for digit {digit} in MNIST dataset.")

    X_digit = shuffle(X_digit, random_state=random_state)
    if sample_idx >= X_digit.shape[0]:
        raise IndexError(
            f"sample_idx {sample_idx} is out of bounds for digit {digit} (found {X_digit.shape[0]} samples)."
        )
    sample = X_digit[sample_idx].reshape(28, 28)

    sample_resized = resize(sample, (image_length, image_length), anti_aliasing=True)

    sample_resized = (sample_resized - np.min(sample_resized)) / (
        np.max(sample_resized) - np.min(sample_resized)
    )

    template = sample_resized.flatten()

    zeroth_freq = np.mean(template)
    template = template - zeroth_freq

    return template


def template_selector(config):
    """Select template based on configuration."""
    if config["template_type"] == "irrep_construction":
        if config["group_name"] == "cnxcn":
            template = fixed_cnxcn(config["image_length"], config["fourier_coef_diag_values"])
        elif config["group_name"] == "cn":
            template = fixed_cn(config["group_n"], config["fourier_coef_diag_values"])
        else:
            template = fixed_group(config["group"], config["fourier_coef_diag_values"])
    elif config["template_type"] == "one_hot":
        template = one_hot(config["group_size"])
    else:
        raise ValueError(f"Unknown template type: {config['template_type']}")
    return template


# ---------------------------------------------------------------------------
# Template functions moved from datamodule.py (renamed per plan)
# ---------------------------------------------------------------------------


def mnist_1d(p: int, label: int, root: str = "data", axis: int = 0):
    """Return a (p,) 1D template from a random MNIST image by taking a slice or projection.

    Args:
        p: dimension of the cyclic group
        label: MNIST digit class (0-9)
        root: MNIST data directory
        axis: 0 for row average, 1 for column average, 2 for diagonal

    Returns:
        template: (p,) array
    """
    import torch
    import torchvision
    import torchvision.transforms as transforms

    if not (0 <= int(label) <= 9):
        raise ValueError("label must be an integer in [0, 9].")

    ds = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )
    cls_idxs = (ds.targets == int(label)).nonzero(as_tuple=True)[0]
    if cls_idxs.numel() == 0:
        raise ValueError(f"No samples for label {label}.")

    idx = cls_idxs[torch.randint(len(cls_idxs), (1,)).item()].item()
    img, _ = ds[idx]
    img = img[0].numpy()

    if axis == 0:
        signal = img.mean(axis=1)
    elif axis == 1:
        signal = img.mean(axis=0)
    elif axis == 2:
        signal = np.diag(img)
    else:
        raise ValueError("axis must be 0, 1, or 2")

    from scipy.interpolate import interp1d

    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, p)
    f = interp1d(x_old, signal, kind="cubic")
    template = f(x_new)

    return template.astype(np.float32)


def mnist_2d(p1: int, p2: int, label: int, root: str = "data"):
    """Return a (p1, p2) template from a random MNIST image.

    Args:
        p1, p2: dimensions
        label: MNIST digit class (0-9)
        root: MNIST data directory

    Returns:
        template: (p1, p2) array
    """
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms

    if not (0 <= int(label) <= 9):
        raise ValueError("label must be an integer in [0, 9].")

    ds = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )
    cls_idxs = (ds.targets == int(label)).nonzero(as_tuple=True)[0]
    if cls_idxs.numel() == 0:
        raise ValueError(f"No samples for label {label}.")

    idx = cls_idxs[torch.randint(len(cls_idxs), (1,)).item()].item()
    img, _ = ds[idx]
    img = nn.functional.interpolate(
        img.unsqueeze(0), size=(p1, p2), mode="bilinear", align_corners=False
    )[0, 0]
    return img.numpy().astype(np.float32)


# --- 1D Synthetic Templates ---


def fourier_1d(p: int, n_freqs: int, amp_max: float = 100, amp_min: float = 10, seed=None):
    """Generate 1D template from random Fourier modes.

    Args:
        p: dimension of cyclic group
        n_freqs: number of frequency components to include
        amp_max: maximum amplitude
        amp_min: minimum amplitude
        seed: random seed

    Returns:
        template: (p,) real-valued array
    """
    rng = np.random.default_rng(seed)
    spectrum = np.zeros(p, dtype=np.complex128)

    available_freqs = list(range(1, p // 2 + 1))
    if len(available_freqs) < n_freqs:
        raise ValueError(
            f"Only {len(available_freqs)} non-DC frequencies available for p={p}, requested {n_freqs}"
        )

    chosen_freqs = rng.choice(
        available_freqs, size=min(n_freqs, len(available_freqs)), replace=False
    )

    amps = np.sqrt(np.linspace(amp_max, amp_min, len(chosen_freqs)))
    phases = rng.uniform(0.0, 2 * np.pi, size=len(chosen_freqs))

    for freq, amp, phi in zip(chosen_freqs, amps, phases):
        v = amp * np.exp(1j * phi)
        spectrum[freq] = v
        spectrum[-freq] = np.conj(v)

    template = np.fft.ifft(spectrum).real
    template -= template.mean()
    s = template.std()
    if s > 1e-12:
        template /= s

    return template.astype(np.float32)


def gaussian_1d(p: int, n_gaussians: int = 3, sigma_range: tuple = (0.5, 2.0), seed=None):
    """Generate 1D template as sum of Gaussians.

    Args:
        p: dimension of cyclic group
        n_gaussians: number of Gaussian bumps
        sigma_range: (min_sigma, max_sigma) for Gaussian widths
        seed: random seed

    Returns:
        template: (p,) real-valued array
    """
    rng = np.random.default_rng(seed)
    x = np.arange(p)
    template = np.zeros(p, dtype=np.float32)

    for _ in range(n_gaussians):
        center = rng.uniform(0, p)
        sigma = rng.uniform(*sigma_range)
        amplitude = rng.uniform(0.5, 1.0)

        dist = np.minimum(np.abs(x - center), p - np.abs(x - center))
        template += amplitude * np.exp(-(dist**2) / (2 * sigma**2))

    template -= template.mean()
    s = template.std()
    if s > 1e-12:
        template /= s

    return template.astype(np.float32)


def onehot_1d(p: int):
    """Generate 1D one-hot template for cyclic group C_p.

    Args:
        p: dimension of cyclic group

    Returns:
        template: (p,) array with template[0] = 1, all others = 0
    """
    template = np.zeros(p, dtype=np.float32)
    template[0] = 1.0
    return template


# --- 2D Synthetic Templates ---


def gaussian_mixture_2d(
    p1=20,
    p2=20,
    n_blobs=8,
    frac_broad=0.7,
    sigma_broad=(3.5, 6.0),
    sigma_narrow=(1.0, 2.0),
    amp_broad=1.0,
    amp_narrow=0.5,
    seed=None,
    normalize=True,
):
    """Build a (p1 x p2) template as a periodic mixture of Gaussians."""
    rng = np.random.default_rng(seed)
    H, W = p1, p2
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    k_broad = int(round(n_blobs * frac_broad))
    k_narrow = n_blobs - k_broad

    def add_blobs(k, sigma_range, amp):
        out = np.zeros((H, W), dtype=float)
        for _ in range(k):
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            sigma = rng.uniform(*sigma_range)
            dy = np.minimum(np.abs(Y - cy), H - np.abs(Y - cy))
            dx = np.minimum(np.abs(X - cx), W - np.abs(X - cx))
            out += amp * np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
        return out

    template = add_blobs(k_broad, sigma_broad, amp_broad) + add_blobs(
        k_narrow, sigma_narrow, amp_narrow
    )

    if normalize:
        template -= template.mean()
        s = template.std()
        if s > 1e-12:
            template /= s
    return template.astype(np.float32)


def unique_freqs_2d(p1, p2, n_freqs, amp_max=100, amp_min=10, seed=None):
    """Real (p1 x p2) template from n_freqs unique Fourier modes.

    Each chosen frequency bin has no conjugate collision.

    Args:
        p1, p2: spatial dims
        n_freqs: number of frequency components
        amp_max, amp_min: amplitude range
        seed: random seed

    Returns:
        template: (p1, p2) real-valued array
    """
    rng = np.random.default_rng(seed)
    spectrum = np.zeros((p1, p2), dtype=np.complex128)

    def ky_signed(ky):
        return ky if ky <= p1 // 2 else ky - p1

    def is_self_conj(ky, kx):
        on_self_kx = (kx == 0) or (p2 % 2 == 0 and kx == p2 // 2)
        if not on_self_kx:
            return False
        s = ky_signed(ky)
        return (s == 0) or (p1 % 2 == 0 and abs(s) == p1 // 2)

    cand = []
    for ky in range(p1):
        s = ky_signed(ky)
        for kx in range(p2 // 2 + 1):
            if ky == 0 and kx == 0:
                continue
            if is_self_conj(ky, kx):
                continue
            r2 = (s**2) + (kx**2)
            cand.append((r2, ky, kx))
    cand.sort(key=lambda t: (t[0], abs(ky_signed(t[1])), t[2]))

    chosen = []
    seen_axis_pairs = set()

    mid_kx = p2 // 2 if (p2 % 2 == 0) else None
    for _, ky, kx in cand:
        if len(chosen) >= n_freqs:
            break
        if (kx == 0) or (mid_kx is not None and kx == mid_kx):
            s = ky_signed(ky)
            key = (kx, min(s, -s))
            if key in seen_axis_pairs:
                continue
            seen_axis_pairs.add(key)
            chosen.append((ky, kx))
        else:
            chosen.append((ky, kx))

    if len(chosen) < n_freqs:
        raise ValueError(
            f"Could only find {len(chosen)} unique non-conjugate bins; "
            f"requested {n_freqs}. Increase grid size or reduce n_freqs."
        )

    amps = np.sqrt(np.linspace(amp_max, amp_min, n_freqs, dtype=float))
    phases = rng.uniform(0.0, 2 * np.pi, size=n_freqs)

    for (ky, kx), a, phi in zip(chosen, amps, phases):
        kyc, kxc = (-ky) % p1, (-kx) % p2
        v = a * np.exp(1j * phi)
        spectrum[ky, kx] += v
        spectrum[kyc, kxc] += np.conj(v)

    template = np.fft.ifft2(spectrum).real
    template -= template.mean()
    s = template.std()
    if s > 1e-12:
        template /= s
    return template.astype(np.float32)


def fixed_2d(p1: int, p2: int) -> np.ndarray:
    """Generate 2D template array from Fourier spectrum.

    Args:
        p1: height dimension
        p2: width dimension

    Returns:
        template: (p1, p2) real-valued array
    """
    spectrum = np.zeros((p1, p2), dtype=complex)

    assert p1 > 5 and p2 > 5, "p1 and p2 must be greater than 5"

    spectrum[1, 0] = 10.0
    spectrum[-1, 0] = 10.0

    spectrum[0, 3] = 7.5
    spectrum[0, -3] = 7.5

    spectrum[2, 1] = 5.0
    spectrum[-2, -1] = 5.0

    template = np.fft.ifft2(spectrum).real

    return template


def _fft_indices(n):
    """Return integer-like frequency indices aligned with numpy's FFT layout."""
    k = np.fft.fftfreq(n) * n
    return k.astype(int)


def hexagon_tie_2d(p1: int, p2: int, k0: float = 6.0, amp: float = 1.0):
    """Real template with hexagonal Fourier spectrum.

    Args:
        p1, p2: spatial dims
        k0: desired radius (index units)
        amp: amplitude per spike

    Returns:
        template: (p1, p2) real-valued array
    """
    assert p1 > 5 and p2 > 5, "p1 and p2 must be > 5"
    spec = np.zeros((p1, p2), dtype=np.complex128)

    thetas = np.arange(6) * (np.pi / 3.0)

    Kx = _fft_indices(p1)
    Ky = _fft_indices(p2)

    def put(kx, ky, val):
        spec[int(kx) % p1, int(ky) % p2] += val

    used = set()
    for th in thetas:
        kx_f = k0 * np.cos(th)
        ky_f = k0 * np.sin(th)
        kx = int(np.round(kx_f))
        ky = int(np.round(ky_f))
        if (kx, ky) == (0, 0):
            if abs(np.cos(th)) > abs(np.sin(th)):
                kx = 1 if kx >= 0 else -1
            else:
                ky = 1 if ky >= 0 else -1
        if (kx, ky) in used:
            continue
        used.add((kx, ky))
        used.add((-kx, -ky))

        put(kx, ky, amp)
        put(-kx, -ky, np.conjugate(amp))

    spec[0, 0] = 0.0

    x = np.fft.ifft2(spec).real
    return x


def ring_isotropic_2d(
    p1: int, p2: int, r0: float = 6.0, sigma: float = 0.5, total_power: float = 1.0
):
    """Real template with an isotropic ring in the 2D spectrum.

    Args:
        p1, p2: spatial dims
        r0: target radius (index units)
        sigma: radial width of the ring
        total_power: scales overall energy

    Returns:
        template: (p1, p2) real-valued array
    """
    assert p1 > 5 and p2 > 5, "p1 and p2 must be > 5"

    kx = _fft_indices(p1)[:, None]
    ky = _fft_indices(p2)[None, :]
    R = np.sqrt(kx**2 + ky**2)

    mag = np.exp(-0.5 * ((R - r0) / max(sigma, 1e-6)) ** 2)

    mag[0, 0] = 0.0

    power = np.sum(mag**2)
    if power > 0:
        mag *= np.sqrt(total_power / power)

    spec = mag.astype(np.complex128)

    x = np.fft.ifft2(spec).real
    return x


def gaussian_2d(
    p1: int,
    p2: int,
    center: tuple = None,
    sigma: float = 2.0,
    k_freqs: int = None,
) -> np.ndarray:
    """Generate 2D template with a single Gaussian, optionally band-limited.

    Args:
        p1: height dimension
        p2: width dimension
        center: (cx, cy) center position
        sigma: standard deviation of Gaussian
        k_freqs: if not None, keep only the top k frequencies by power

    Returns:
        template: (p1, p2) real-valued array
    """
    if center is None:
        center = (p1 / 2, p2 / 2)
    cx, cy = center
    x = np.arange(p1)
    y = np.arange(p2)
    X, Y = np.meshgrid(x, y, indexing="ij")
    template = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
    if k_freqs is not None:
        spectrum = np.fft.fft2(template)
        power = np.abs(spectrum) ** 2
        power_flat = power.flatten()
        kx_indices = np.arange(p1)
        ky_indices = np.arange(p2)
        KX, KY = np.meshgrid(kx_indices, ky_indices, indexing="ij")
        all_indices = list(zip(KX.flatten(), KY.flatten()))
        sorted_idx = np.argsort(-power_flat)
        top_k_idx = sorted_idx[:k_freqs]
        top_k_freqs = set([all_indices[i] for i in top_k_idx])
        mask = np.zeros((p1, p2), dtype=complex)
        for kx, ky in top_k_freqs:
            mask[kx, ky] = 1.0
        spectrum_masked = spectrum * mask
        template = np.fft.ifft2(spectrum_masked).real
    return template
