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


def get_power_2d(points, no_freq=False):
    """
    Compute the 2D power spectrum of a real-valued array.

    Note on redundant frequencies:
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
    points : ndarray (M, N)
        Real-valued 2D input array.

    Returns
    -------
    freqs_u : ndarray (M,)
        Frequency bins for the first axis (rows).
    freqs_v : ndarray (N//2 + 1,)
        Frequency bins for the second axis (columns).
    power : ndarray (M, N//2 + 1)
        Power spectrum of the input.
    """
    M, N = points.shape
    num_coefficients = N // 2 + 1
    
    # Perform 2D rFFT
    ft = np.fft.rfft2(points)
    
    # Power spectrum normalized by total number of samples
    power = np.abs(ft)**2 / (M * N)

    # For the first row (u=0), remove redundant frequencies and double the appropriate ones
    power[(N//2 + 1):, 0] = 0 

    # Since (almost) all frequencies contribute twice (positive and negative), double the power
    power *= 2
    # Except the DC component
    power[0, 0] /= 2
    # Except the Nyquist frequency if N is even
    if N % 2 == 0:
        power[N//2, 0] /= 2

    # Check Parseval’s theorem
    total_power = np.sum(power)
    norm_squared = np.linalg.norm(points)**2
    if not np.isclose(total_power, norm_squared, rtol=1e-3):
        print(f"Warning: Total power {total_power:.3f} does not match norm squared {norm_squared:.3f}")


    if no_freq:
        return power

    # Frequency bins
    freqs_u = np.fft.fftfreq(M)          # full symmetric frequencies (rows)
    freqs_v = np.fft.rfftfreq(N)         # only non-negative frequencies (columns)

    return freqs_u, freqs_v, power


def get_alpha_values_and_valid_freq(template):
    p = int(np.sqrt(len(template)))
    x_freq, y_freq, power = get_power_2d(template.reshape((p, p)))
    print(power)
    power = power.flatten()

    valid = power > 1e-20
    power = power[valid]
    sorted_idx = np.argsort(power)[::-1]  # np.argsort with [::-1] gives descending order
    power = power[sorted_idx]

    # Plot theoretical lines
    alpha_values = [np.sum(power[k:]) for k in range(len(power))]
    coef = 1 / (p * p)
    alpha_values = [alpha * coef for alpha in alpha_values]

    return alpha_values, power, valid, sorted_idx


def model_power_over_time(model, param_history, model_inputs, template_power_length, p, valid_freqs=None):
    """Compute the power spectrum of the model's learned weights over time.

    Parameters
    ----------
    model : TwoLayerNet
        The trained model.
    valid_freqs : list of tuple, optional
        List of frequency indices to track. If None, track all frequencies.

    Returns
    -------
    avg_power_history : list of ndarray (num_steps, num_freqs)
        List of average power spectra at each step.
    """
    # Compute output power over time (GD)
    num_points = 1000
    X_tensor = model_inputs
    steps = np.unique(np.logspace(0, np.log10(len(param_history) - 1), num_points, dtype=int))
    powers_over_time = np.zeros([len(steps), template_power_length])  # shape: (steps, num_freqs) np.zeros([len(steps), len(X_tensor), template_power_length]) 
    # powers_over_time = np.zeros([len(steps), len(X_tensor), template_power_length]) 

    for i_step, step in enumerate(steps):
        model.load_state_dict(param_history[step])
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            # Compute 2D FFT for each output, then flatten and take rfft2
            outputs_2d = outputs.detach().cpu().numpy().reshape(-1, p, p)
            powers = np.array([get_power_2d(out, no_freq=True) for out in outputs_2d])
            # Don't flatten the batch dim; keep as (num_samples, p* p//2 + 1)
            powers = powers.reshape(outputs_2d.shape[0], -1)
            average_power = np.mean(powers, axis=0)
            powers_over_time[i_step, :] = average_power

    powers_over_time = np.array(powers_over_time)  # shape: (steps, num_freqs)
    print("Powers over time shape:", powers_over_time.shape)

    if valid_freqs is not None:
        valid_indices = [idx for idx, freq in enumerate(freq_tuples) if freq in valid_freqs]
        powers_over_time = powers_over_time[:, valid_indices]
        print(f"Filtered to {len(valid_indices)} valid frequencies.")
    
    return powers_over_time, steps
    