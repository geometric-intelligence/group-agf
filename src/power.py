import numpy as np
import torch

import src.fourier as fourier


class CyclicPower:
    """Compute and store the power spectrum of the template, which can be used
    to compute theoretical loss plateau predictions for the ZnZ group and compare to learned power spectrum.

    Parameters
    ----------
    template : ndarray (p*p)
        Flattened 2D template array.
    """

    def __init__(self, template, template_dim):
        self.template = template
        self.template_dim = template_dim
        if template_dim == 2:
            self.group_size = int(np.sqrt(len(template)))
            self.template_2D = template.reshape((self.group_size, self.group_size))
            self.x_freqs, self.y_freqs, self.power = self.cnxcn_power_spectrum(return_freqs=True)
        else:
            self.group_size = len(template)
            self.freqs, self.power = self.cn_power_spectrum(return_freqs=True)

    def cn_power_spectrum(self, return_freqs=False):
        """Compute the 1D power spectrum of 1D FT."""
        num_coefficients = (self.group_size // 2) + 1

        # Perform FFT and calculate power spectrum
        ft = np.fft.fft(
            self.template
        )  # Could consider using np.fft.rfft which is designed for real valued input.
        power = np.abs(ft[:num_coefficients]) ** 2 / self.group_size

        # Double power for frequencies strictly between 0 and Nyquist (Nyquist is not doubled if p is even)
        if (
            self.group_size % 2 == 0
        ):  # group size is even, Nyquist frequency at index num_coefficients - 1
            power[1 : num_coefficients - 1] *= 2
        else:  # p is odd, no Nyquist frequency
            power[1:] *= 2

        # Confirm the power sum approximates the squared norm of points
        total_power = np.sum(power)
        norm_squared = np.linalg.norm(self.template) ** 2
        if not np.isclose(total_power, norm_squared, rtol=1e-3):
            print(
                f"Warning: Total power {total_power:.3f} does not match norm squared {norm_squared:.3f}"
            )

        if return_freqs:
            freqs = np.fft.rfftfreq(self.group_size)
            return freqs, power

        return power

    def cnxcn_power_spectrum(self, return_freqs=False):
        """
        Compute the 2D power spectrum of 2D FT.

        Why are some powers doubled?
        rfft2 removes redundant frequencies along first axis automatically
        but does not truncate the second axis
        Therefore, the output shape is (M, N//2 + 1).
        This eliminates redundancy, save for a specific cases:
        --> All frequencies along the first axis at (u, 0) for u = N//2 + 1, ..., N - 1
        are redundant and contain the same information as (u, 0) for u = 1, ..., N//2 - 1.

        Since most of the power coefficients now represnet 2 frequencies (positive and negative),
        we double all the power coefficients to conserve total power.
        However, we do not double the DC component (0, 0) and the Nyquist frequency (N/2, 0) if N is even,
        since these are unique and do not have a negative counterpart.

        Parameters
        ----------
        template : ndarray (M, N)
            Real-valued 2D input array.

        Returns
        -------
        row_freqs : ndarray (M,)
            Frequency bins for the first axis (rows).
        column_freqs : ndarray (N//2 + 1,)
            Frequency bins for the second axis (columns).
        power : ndarray (M, N//2 + 1)
            Power spectrum of the input.
        """
        M, N = self.template_2D.shape

        # Perform 2D rFFT
        ft = np.fft.rfft2(self.template_2D)

        # Power spectrum normalized by total number of samples
        power = np.abs(ft) ** 2 / (M * N)

        # For the first row (u=0), remove redundant frequencies and double the appropriate ones
        power[(N // 2 + 1) :, 0] = 0

        power *= 2
        power[0, 0] /= 2
        if N % 2 == 0:
            power[N // 2, 0] /= 2

        # Check Parseval’s theorem
        total_power = np.sum(power)
        norm_squared = np.linalg.norm(self.template_2D) ** 2
        if not np.isclose(total_power, norm_squared, rtol=1e-3):
            print(
                f"Warning: Total power {total_power:.3f} does not match norm squared {norm_squared:.3f}"
            )

        if return_freqs:
            # Frequency bins
            row_freqs = np.fft.fftfreq(M)  # full symmetric frequencies (rows)
            column_freqs = np.fft.rfftfreq(N)  # only non-negative frequencies (columns)

            return row_freqs, column_freqs, power

        return power

    def loss_plateau_predictions(self):
        """Compute theoretical loss plateau predictions from the template's power spectrum.
        (as predicted by AGF)

        Returns
        -------
        plateau_predictions : list of float
            Theoretical loss plateau predictions for each nonzero power, in descending order.
        """
        power = self.power

        if self.template_dim == 2:
            img_size = int(np.sqrt(len(self.template)))
            print("Computing loss plateau predictions for template of shape:", (img_size, img_size))
            power = power.flatten()

        nonzero_power_mask = power > 1e-20
        power = power[nonzero_power_mask]
        i_power_descending_order = np.argsort(power)[::-1]
        power = power[i_power_descending_order]

        plateau_predictions = [np.sum(power[k:]) for k in range(len(power))]
        coef = 1 / self.group_size
        plateau_predictions = [alpha * coef for alpha in plateau_predictions]

        return plateau_predictions


class GroupPower:
    """Compute and store the power spectrum of the template for the Dihedral group.

    Parameters
    ----------
    template : ndarray (p,)
        1D template array.
    group : Group (escnn object)
        The group the template is defined over.
        Also specifies which fourier transform to apply, and thus
        which transform to compute the power spectrum for.
    """

    def __init__(self, template, group):
        self.template = template
        self.group_size = len(template)
        self.group = group
        self.power = self.group_power_spectrum()
        self.freqs = list(range(len(self.power)))

    def group_power_spectrum(self):
        """Compute the (group) power spectrum of the template.

        For each irrep rho, the power is given by:
        ||hat x(rho)||_rho = dim(rho) * Tr(hat x(rho)^dagger * hat x(rho))
        where hat x(rho) is the (matrix) Fourier coefficient of template x at irrep rho.

        We multiply by the dimension of the irrep because for 2D irreps, the power
        would otherwise be split across two dimensions, so we must double it to get the correct
        total power.

        Returns
        -------
        power_spectrum : np.ndarray, shape=[len(group.irreps())]
            The power spectrum of the template.
        """
        fourier_coefs = fourier.group_fourier(self.group, self.template)
        irreps = self.group.irreps()

        power_spectrum = np.zeros(len(irreps))
        for i, irrep in enumerate(irreps):
            fc = fourier_coefs[i]
            power_spectrum[i] = irrep.size * np.trace(fc.conj().T @ fc)
        power_spectrum = power_spectrum / self.group.order()

        return np.array(power_spectrum)

    def loss_plateau_predictions(self):
        """Compute theoretical loss plateau predictions from the template's power spectrum.

        The loss plateau predictions give the levels of the loss plot.

        Returns
        -------
        plateau_predictions : list of float
            Theoretical loss plateau predictions for each nonzero power, in descending order.
        """
        p = len(self.template)
        print("Computing loss plateau predictions for template of shape:", (p,))
        power = self.power
        nonzero_power_mask = power > 1e-20
        power = power[nonzero_power_mask]
        print("Found ", len(power), "non-zero power coefficients.")
        i_power_descending_order = np.argsort(power)[::-1]
        power = power[i_power_descending_order]
        plateau_predictions = [np.sum(power[k:]) for k in range(len(power))]
        coef = 1 / p
        plateau_predictions = [alpha * coef for alpha in plateau_predictions]
        return plateau_predictions


def model_power_over_time(group_name, model, param_history, model_inputs, group=None):
    """Compute the power spectrum of the model's learned weights over time.

    Parameters
    ----------
    group_name : str
        Group type (e.g., 'cnxcn').
    model : TwoLayerNet
        The trained model.
    param_history : list of dict
        List of model parameters at each training step.
    model_inputs : torch.Tensor
        Input data tensor.
    group : Group (escnn object)
        The escnn group object. Optional, since we don't use escnn for cnxcn.

    Returns
    -------
    avg_power_history : list of ndarray (num_steps, num_freqs)
        List of average power spectra at each step.
    """
    # Determine output shape: support both 1D and 2D
    model.eval()
    with torch.no_grad():
        test_output = model(model_inputs[:1])
    output_shape = test_output.shape[1:]

    if group_name == "cnxcn":  # 2D template
        p1 = int(np.sqrt(output_shape[0]))
        p2 = p1
        template_power_length = p1 * (p2 // 2 + 1)
        reshape_dims = (-1, p1, p2)
    elif group_name == "cn":  # 1D template
        template_power_length = (output_shape[0] // 2) + 1
        p1 = output_shape[0]
        reshape_dims = (-1, p1)
    else:  # other groups are 1D signals
        template_power_length = len(group.irreps())
        p1 = output_shape[0]
        reshape_dims = (-1, p1)

    num_points = 200
    max_step = len(param_history) - 1
    num_inputs_to_compute_power = max(1, len(model_inputs) // 50)  # Ensure at least 1 input
    X_tensor = model_inputs[
        :num_inputs_to_compute_power
    ]  # Added by Nina to speed up computation with octahedral.
    if max_step <= 1:
        # Very short training: just use all available checkpoints
        steps = np.arange(max_step + 1)
    else:
        steps = np.unique(np.logspace(1, np.log10(max_step), num_points, dtype=int))
        steps = steps[steps > 50]
        steps = np.hstack([np.linspace(1, min(50, max_step), 5).astype(int), steps])
    # Ensure all indices are within bounds
    steps = np.unique(steps)
    steps = steps[steps <= max_step]
    powers_over_time = np.zeros([len(steps), template_power_length])

    for i_step, step in enumerate(steps):
        model.load_state_dict(param_history[step])

        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            outputs_arr = outputs.detach().cpu().numpy().reshape(reshape_dims)

            if i_step % 10 == 0:
                print("Computing power at step", step, "with output shape", outputs_arr.shape)

            powers = []
            for out in outputs_arr:
                if group_name == "cnxcn":
                    output_power = CyclicPower(out.flatten(), template_dim=2)
                elif group_name == "cn":
                    output_power = CyclicPower(out.flatten(), template_dim=1)
                else:
                    output_power = GroupPower(out.flatten(), group=group)

                one_power = output_power.power
                # flatten to 1D for both 1D and 2D cases
                one_power_flat = one_power.flatten()
                powers.append(one_power_flat)
            powers = np.array(powers)

            average_power = np.mean(powers, axis=0)  # shape: (num_samples, template_power_length)
            powers_over_time[i_step, :] = average_power

    powers_over_time = np.array(powers_over_time)  # shape: (steps, num_freqs)
    powers_over_time[powers_over_time < 1e-20] = 0

    return powers_over_time, steps


# ---------------------------------------------------------------------------
# Power spectrum computation functions (moved from utils.py)
# ---------------------------------------------------------------------------


def get_power_1d(points_1d):
    """Compute 1D power spectrum using rfft (for real-valued inputs).

    Args:
        points_1d: (p,) array

    Returns:
        power: (p//2+1,) array of power values
        freqs: frequency indices
    """
    p = len(points_1d)

    ft = np.fft.rfft(points_1d)
    power = np.abs(ft) ** 2 / p

    power = 2 * power.copy()
    power[0] = power[0] / 2  # DC component
    if p % 2 == 0:
        power[-1] = power[-1] / 2  # Nyquist frequency

    freqs = np.fft.rfftfreq(p, 1.0) * p

    return power, freqs


def topk_template_freqs_1d(template_1d: np.ndarray, K: int, min_power: float = 1e-20):
    """Return top-K frequency indices by power for 1D template.

    Args:
        template_1d: 1D template array (p,)
        K: Number of top frequencies to return
        min_power: Minimum power threshold

    Returns:
        List of frequency indices (as integers)
    """
    power, _ = get_power_1d(template_1d)
    mask = power > min_power
    if not np.any(mask):
        return []
    valid_power = power[mask]
    valid_indices = np.flatnonzero(mask)
    top_idx = valid_indices[np.argsort(valid_power)[::-1]][:K]
    return top_idx.tolist()


def topk_template_freqs(template_2d: np.ndarray, K: int, min_power: float = 1e-20):
    """Return top-K (kx, ky) rFFT2 bins by power from get_power_2d(template_2d)."""
    freqs_u, freqs_v, power = get_power_2d(template_2d)
    shp = power.shape
    flat = power.ravel()
    mask = flat > min_power
    if not np.any(mask):
        return []
    top_idx = np.flatnonzero(mask)[np.argsort(flat[mask])[::-1]][:K]
    kx, ky = np.unravel_index(top_idx, shp)
    return list(zip(kx.tolist(), ky.tolist()))


def get_power_2d(points, no_freq=False):
    """Compute 2D power spectrum using rfft2 with proper symmetry handling.

    Args:
        points: (M, N) array, the 2D signal
        no_freq: if True, only return power (no frequency arrays)

    Returns:
        freqs_u: frequency bins for rows (if no_freq=False)
        freqs_v: frequency bins for columns (if no_freq=False)
        power: 2D power spectrum (M, N//2 + 1)
    """
    M, N = points.shape

    ft = np.fft.rfft2(points)
    power = np.abs(ft) ** 2 / (M * N)

    weight = 2 * np.ones((M, N // 2 + 1))
    weight[0, 0] = 1
    weight[(M // 2 + 1) :, 0] = 0
    if M % 2 == 0:
        weight[M // 2, 0] = 1
    if N % 2 == 0:
        weight[(M // 2 + 1) :, N // 2] = 0
        weight[0, N // 2] = 1
    if (M % 2 == 0) and (N % 2 == 0):
        weight[M // 2, N // 2] = 1

    power = weight * power

    total_power = np.sum(power)
    norm_squared = np.linalg.norm(points) ** 2
    if not np.isclose(total_power, norm_squared, rtol=1e-6):
        print(
            f"Warning: Total power {total_power:.3f} does not match norm squared {norm_squared:.3f}"
        )

    if no_freq:
        return power

    freqs_u = np.fft.fftfreq(M)
    freqs_v = np.fft.rfftfreq(N)

    return freqs_u, freqs_v, power


def _tracked_power_from_fft2(power2d, kx, ky, p1, p2):
    """Sum power at (kx, ky) and its real-signal mirror (-kx, -ky).

    Args:
        power2d: 2D power spectrum from fft2 (shape: p1, p2)
        kx, ky: Frequency indices
        p1, p2: Dimensions of the signal

    Returns:
        float: Total power at this frequency (including mirror)
    """
    i0, j0 = kx % p1, ky % p2
    i1, j1 = (-kx) % p1, (-ky) % p2
    if (i0, j0) == (i1, j1):
        return float(power2d[i0, j0])
    return float(power2d[i0, j0] + power2d[i1, j1])


def theoretical_loss_levels_2d(template_2d):
    """Compute theoretical MSE loss levels based on 2D template power spectrum.

    Args:
        template_2d: 2D template array (p1, p2)

    Returns:
        dict with 'initial', 'final', and 'levels' keys
    """
    p1, p2 = template_2d.shape
    power = get_power_2d(template_2d, no_freq=True)

    power_flat = power.flatten()
    power_flat = np.sort(power_flat[power_flat > 1e-20])[::-1]

    coef = 1.0 / (p1 * p2)
    levels = [coef * np.sum(power_flat[k:]) for k in range(len(power_flat) + 1)]

    return {
        "initial": levels[0] if levels else 0.0,
        "final": 0.0,
        "levels": levels,
    }


def theoretical_loss_levels_1d(template_1d):
    """Compute theoretical MSE loss levels based on 1D template power spectrum.

    Args:
        template_1d: 1D template array (p,)

    Returns:
        dict with 'initial', 'final', and 'levels' keys
    """
    p = len(template_1d)
    power, _ = get_power_1d(template_1d)

    power = np.sort(power[power > 1e-20])[::-1]

    coef = 1.0 / p
    levels = [coef * np.sum(power[k:]) for k in range(len(power) + 1)]

    return {
        "initial": levels[0] if levels else 0.0,
        "final": 0.0,
        "levels": levels,
    }


# Backward compatibility aliases
def theoretical_final_loss_2d(template_2d):
    """Returns expected initial loss (for setting convergence targets)."""
    return theoretical_loss_levels_2d(template_2d)["initial"]


def theoretical_final_loss_1d(template_1d):
    """Returns expected initial loss (for setting convergence targets)."""
    return theoretical_loss_levels_1d(template_1d)["initial"]


def group_power_spectrum(group, template):
    """Compute the (group) power spectrum of the template.

    For each irrep rho, the power is given by:
    ||hat x(rho)||_rho = dim(rho) * Tr(hat x(rho)^dagger * hat x(rho))

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    template : np.ndarray, shape=[group.order()]
        The template to compute the power spectrum of.

    Returns
    -------
    power_spectrum : np.ndarray, shape=[len(group.irreps())]
        The power spectrum of the template.
    """
    fourier_coefs = fourier.group_fourier(group, template)
    irreps = group.irreps()

    power_spectrum = np.zeros(len(irreps))
    for i, irrep in enumerate(irreps):
        fc = fourier_coefs[i]
        power_spectrum[i] = irrep.size * np.trace(fc.conj().T @ fc)
    power_spectrum = power_spectrum / group.order()
    return np.array(power_spectrum)
