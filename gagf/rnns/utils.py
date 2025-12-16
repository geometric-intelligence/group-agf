from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt



### ----- VISUALIZATION FUNCTIONS ----- ###

def style_axes(ax, numyticks=5, numxticks=5, labelsize=24):
    # Y-axis ticks
    ax.tick_params(
        axis="y",
        which="both",
        bottom=True,
        top=False,
        labelbottom=True,
        left=True,
        right=False,
        labelleft=True,
        direction="out",
        length=7,
        width=1.5,
        pad=8,
        labelsize=labelsize,
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=numyticks))

    # X-axis ticks
    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=True,
        left=True,
        right=False,
        labelleft=True,
        direction="out",
        length=7,
        width=1.5,
        pad=8,
        labelsize=labelsize,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=numxticks))

    ax.xaxis.offsetText.set_fontsize(20)
    ax.grid()

    # Customize spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(3)



def plot_train_val_loss(
    train_loss_history, 
    val_loss_history,
    save_path=None,
    show=True,
    xlabel='Step'
):
    """
    Plot training and validation loss vs steps.
    
    Args:
        train_loss_history: List of training loss values
        val_loss_history: List of validation loss values
        save_path: Optional path to save figure
        show: Whether to display the plot
        xlabel: Label for x-axis (e.g., 'Step' or 'Epoch')
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    steps = np.arange(len(train_loss_history))
    
    ax.plot(steps, train_loss_history, lw=2, color='#1f77b4', label='Training Loss', alpha=0.7)
    ax.plot(steps, val_loss_history, lw=2, color='#ff7f0e', label='Validation Loss')
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training vs Validation Loss', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale often helps see loss curves
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  ✓ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_2d_signal(
    signal_2d,
    title="",
    cmap="RdBu_r",
    colorbar=True,
):
    """Plot a 2D signal as a heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(signal_2d, cmap=cmap, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("y", fontsize=12)
    ax.set_ylabel("x", fontsize=12)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, ax


def plot_2d_power_spectrum(
    ax,
    power_2d,
    fx=None,
    fy=None,
    title="Power Spectrum",
    cmap="viridis",
    log_scale=True,
    shift=True,
):
    """Plot 2D power spectrum with proper frequency axes."""
    if log_scale:
        power_plot = np.log10(power_2d + 1e-12)
        title = f"{title} (log₁₀)"
    else:
        power_plot = power_2d

    # Optionally shift to center zero frequency
    if shift:
        power_plot = np.fft.fftshift(power_plot)
        if fx is not None and fy is not None:
            fx = np.fft.fftshift(fx)
            fy = np.fft.fftshift(fy)

    # Set up extent for proper frequency axis labeling
    if fx is not None and fy is not None:
        extent = [fy.min(), fy.max(), fx.min(), fx.max()]
    else:
        extent = None

    im = ax.imshow(
        power_plot,
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
        origin="lower",
        extent=extent,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("k_y (frequency)", fontsize=12)
    ax.set_ylabel("k_x (frequency)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


### ----- POWER SPECTRUM FUNCTIONS ----- ###

def get_power_1d(points_1d):
    """
    Compute 1D power spectrum using rfft (for real-valued inputs).

    Args:
        points_1d: (p,) array

    Returns:
        power: (p//2+1,) array of power values
        freqs: frequency indices
    """
    p = len(points_1d)

    # Perform 1D FFT
    ft = np.fft.rfft(points_1d)
    power = np.abs(ft) ** 2 / p
    
    # Handle conjugate symmetry for real signals
    power = 2 * power.copy()
    power[0] = power[0] / 2  # DC component
    if p % 2 == 0:
        power[-1] = power[-1] / 2  # Nyquist frequency

    freqs = np.fft.rfftfreq(p, 1.0) * p

    return power, freqs


def topk_template_freqs_1d(template_1d: np.ndarray, K: int, min_power: float = 1e-20):
    """
    Return top-K frequency indices by power for 1D template.
    
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
    """
    Return top-K (kx, ky) rFFT2 bins by power from get_power_2d_adele(template_2d).
    """
    freqs_u, freqs_v, power = get_power_2d_adele(template_2d)  # power shape: (p1, p2//2 + 1)
    shp = power.shape
    flat = power.ravel()
    mask = flat > min_power
    if not np.any(mask):
        return []
    top_idx = np.flatnonzero(mask)[np.argsort(flat[mask])[::-1]][:K]
    kx, ky = np.unravel_index(top_idx, shp)
    return list(zip(kx.tolist(), ky.tolist()))


def get_power_2d_adele(points, no_freq=False):
    """
    Compute 2D power spectrum using rfft2 with proper symmetry handling.
    
    Args:
        points: (M, N) array, the 2D signal
        no_freq: if True, only return power (no frequency arrays)
    
    Returns:
        freqs_u: frequency bins for rows (if no_freq=False)
        freqs_v: frequency bins for columns (if no_freq=False)
        power: 2D power spectrum (M, N//2 + 1)
    """
    M, N = points.shape
    
    # Perform 2D rFFT
    ft = np.fft.rfft2(points)
    
    # Power spectrum normalized by total number of samples
    power = np.abs(ft)**2 / (M * N)

    # Construct weighting to handle real conjugate symmetry
    weight = 2 * np.ones((M, N // 2 + 1))
    weight[0, 0] = 1  # handles DC component
    weight[(M//2 + 1):, 0] = 0  # handles DC frequency in second axis
    if M % 2 == 0:
        weight[M//2, 0] = 1
    if N % 2 == 0:
        weight[(M//2 + 1):, N//2] = 0
        weight[0, N//2] = 1
    if (M % 2 == 0) and (N % 2 == 0):
        weight[M//2, N//2] = 1

    # Reweight power to account for redundancies
    power = weight * power

    # Check Parseval's theorem
    total_power = np.sum(power)
    norm_squared = np.linalg.norm(points)**2
    if not np.isclose(total_power, norm_squared, rtol=1e-6):
        print(f"Warning: Total power {total_power:.3f} does not match norm squared {norm_squared:.3f}")

    if no_freq:
        return power

    # Frequency bins
    freqs_u = np.fft.fftfreq(M)          # full symmetric frequencies (rows)
    freqs_v = np.fft.rfftfreq(N)         # only non-negative frequencies (columns)

    return freqs_u, freqs_v, power


def compute_theoretical_loss_levels_2d(template_2d):
    """
    Compute theoretical MSE loss levels based on template power spectrum.
    
    Returns both the initial loss (before learning) and final loss (fully converged).
    The theory predicts step-wise loss reductions as each Fourier mode is learned.
    
    Args:
        template_2d: 2D template array (p1, p2)
    
    Returns:
        dict with:
            'initial': Expected MSE before any learning (= Var(template))
            'final': Expected MSE when fully converged (~0)
            'levels': All intermediate loss plateaus
    """
    p1, p2 = template_2d.shape
    power = get_power_2d_adele(template_2d, no_freq=True)
    
    power_flat = power.flatten()
    power_flat = np.sort(power_flat[power_flat > 1e-20])[::-1]  # Descending
    
    coef = 1.0 / (p1 * p2)
    
    # Theory levels: cumulative tail sums
    levels = [coef * np.sum(power_flat[k:]) for k in range(len(power_flat) + 1)]
    
    return {
        'initial': levels[0] if levels else 0.0,  # Before learning any mode
        'final': 0.0,  # When all modes are learned
        'levels': levels,
    }


def compute_theoretical_loss_levels_1d(template_1d):
    """
    Compute theoretical MSE loss levels based on 1D template power spectrum.
    
    Args:
        template_1d: 1D template array (p,)
    
    Returns:
        dict with:
            'initial': Expected MSE before any learning
            'final': Expected MSE when fully converged (~0)
            'levels': All intermediate loss plateaus
    """
    p = len(template_1d)
    power, _ = get_power_1d(template_1d)
    
    power = np.sort(power[power > 1e-20])[::-1]  # Descending
    
    coef = 1.0 / p
    
    # Theory levels: cumulative tail sums
    levels = [coef * np.sum(power[k:]) for k in range(len(power) + 1)]
    
    return {
        'initial': levels[0] if levels else 0.0,
        'final': 0.0,
        'levels': levels,
    }


# Backward compatibility aliases
def compute_theoretical_final_loss_2d(template_2d):
    """Returns expected initial loss (for setting convergence targets)."""
    return compute_theoretical_loss_levels_2d(template_2d)['initial']


def compute_theoretical_final_loss_1d(template_1d):
    """Returns expected initial loss (for setting convergence targets)."""
    return compute_theoretical_loss_levels_1d(template_1d)['initial']


def _tracked_power_from_fft2(power2d, kx, ky, p1, p2):
    """
    Sum power at (kx, ky) and its real-signal mirror (-kx, -ky).
    
    For real signals, the full FFT has conjugate symmetry, so power at (kx, ky)
    and (-kx, -ky) are equal. This helper sums both for consistent power measurement.
    
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


def _squareish_grid(n):
    """Compute nearly-square grid dimensions for n items."""
    c = int(np.ceil(np.sqrt(n)))
    r = int(np.ceil(n / c))
    return r, c


def _fourier_mode_2d(p1: int, p2: int, kx: int, ky: int, phase: float = 0.0):
    """Generate a 2D Fourier mode (cosine wave), normalized to [0, 1]."""
    y = np.arange(p1)[:, None]
    x = np.arange(p2)[None, :]
    mode = np.cos(2 * np.pi * (ky * y / p1 + kx * x / p2) + phase)
    mmin, mmax = mode.min(), mode.max()
    return (mode - mmin) / (mmax - mmin) if mmax > mmin else mode


def _signed_k(k: int, n: int) -> int:
    """Convert frequency index to signed representation (-n/2 to n/2)."""
    return k if k <= n // 2 else k - n


def _pretty_k(k: int, n: int) -> str:
    """Format frequency for display (handles Nyquist frequency with ± symbol)."""
    if n % 2 == 0 and k == n // 2:
        return r"\pm{}".format(n // 2)
    return f"{_signed_k(k, n)}"


def _permutation_from_groups_with_dead(
    dom_idx, phase, dom_power, l2, *,
    within="phase", dead_l2_thresh=1e-1
):
    """
    Create neuron permutation grouped by dominant frequency.
    
    Args:
        dom_idx: Dominant frequency index for each neuron
        phase: Phase at dominant frequency for each neuron
        dom_power: Power at dominant frequency for each neuron
        l2: L2 norm of each neuron's weights
        within: How to order within groups ('phase', 'power', 'phase_power', 'none')
        dead_l2_thresh: L2 threshold below which neurons are "dead"
    
    Returns:
        perm: Permutation indices
        ordered_keys: Ordered list of group keys (-1 for dead)
        boundaries: Cumulative indices where groups end
    """
    dead_mask = l2 < float(dead_l2_thresh)
    groups = {}
    for i, f in enumerate(dom_idx):
        key = -1 if dead_mask[i] else int(f)
        groups.setdefault(key, []).append(i)
    
    freq_keys = sorted([k for k in groups.keys() if k >= 0])
    ordered_keys = freq_keys + ([-1] if -1 in groups else [])
    
    perm, boundaries = [], []
    for f in ordered_keys:
        idxs = groups[f]
        if f == -1:
            idxs = sorted(idxs, key=lambda i: l2[i])
        else:
            if within == "phase" and phase is not None:
                idxs = sorted(idxs, key=lambda i: (phase[i] + 2*np.pi) % (2*np.pi))
            elif within == "power" and dom_power is not None:
                idxs = sorted(idxs, key=lambda i: -dom_power[i])
            elif within == "phase_power":
                idxs = sorted(idxs, key=lambda i: 
                             ((phase[i] + 2*np.pi) % (2*np.pi), -dom_power[i]))
        perm.extend(idxs)
        boundaries.append(len(perm))
    
    return np.array(perm, dtype=int), ordered_keys, boundaries


def plot_training_loss_with_theory(
    loss_history, 
    template_2d, 
    p1, 
    p2,
    x_values=None,
    x_label="Step",
    save_path=None,
    show=True
):
    """
    Plot training loss with theoretical power spectrum lines.
    
    Args:
        loss_history: List of loss values
        template_2d: The 2D template array (p1, p2)
        p1, p2: Dimensions
        x_values: X-axis values (if None, uses indices 0, 1, 2, ...)
        x_label: Label for x-axis (e.g., "Samples Seen", "Fraction of Space")
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Use provided x_values or default to indices
    if x_values is None:
        x_values = np.arange(len(loss_history))
    
    # Plot loss
    ax.plot(x_values, loss_history, lw=4, color='#1f77b4', label='Training Loss')
    
    # Compute power spectrum of template
    x_freq, y_freq, power = get_power_2d_adele(template_2d)
    power = power.flatten()
    valid = power > 1e-20
    power = power[valid]
    power = np.sort(power)[::-1]  # Descending order
    
    # Plot theoretical lines (cumulative tail sums)
    alpha_values = [np.sum(power[k:]) for k in range(len(power))]
    coef = 1 / (p1 * p2)
    for k, alpha in enumerate(alpha_values):
        ax.axhline(y=coef * alpha, color="black", linestyle="--", linewidth=2, zorder=-2)
    
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel("Train Loss", fontsize=24)
    
    style_axes(ax)
    ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved loss plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

def plot_model_predictions_over_time(
    model,
    param_history,
    X_data,
    Y_data,
    p1,
    p2,
    steps=None,
    example_idx=None,
    cmap="gray",
    save_path=None,
    show=False
):
    """
    Plot model predictions at different training steps vs ground truth.
    
    Args:
        model: The trained model
        param_history: List of parameter snapshots from training
        X_data: Input tensor (N, k, p1*p2)
        Y_data: Target tensor (N, p1*p2)
        p1, p2: Dimensions
        steps: List of epoch indices to plot (default: [1, 5, 10, final])
        example_idx: Index of example to visualize (default: random)
        cmap: Colormap to use
        save_path: Path to save figure
        show: Whether to display the plot
    """
    import torch
    
    # Default steps
    if steps is None:
        final_step = len(param_history) - 1
        steps = [1, min(5, final_step), min(10, final_step), final_step]
        steps = sorted(list(set(steps)))  # Remove duplicates
    
    # Random example if not specified
    if example_idx is None:
        example_idx = int(np.random.randint(len(Y_data)))
    
    device = next(model.parameters()).device
    model.to(device).eval()
    
    # Ground truth
    if Y_data.dim() == 3:
        Y_data = Y_data[:,-1,:] # only final time step
    with torch.no_grad():
        truth_2d = Y_data[example_idx].reshape(p1, p2).cpu().numpy()
    
    # Collect predictions at each step
    preds = []
    for step in steps:
        model.load_state_dict(param_history[step], strict=True)
        with torch.no_grad():
            x = X_data[example_idx:example_idx+1].to(device)
            pred_2d = model(x)
            if pred_2d.dim() == 3:
                pred_2d = pred_2d[:,-1,:] # only final time step
            
            pred_2d = pred_2d.reshape(p1, p2).detach().cpu().numpy()
            
            preds.append(pred_2d)
    
    # Shared color scale based on ground truth
    vmin = np.min(truth_2d)
    vmax = np.max(truth_2d)
    
    # Plot: rows = [Prediction, Target], cols = time steps
    fig, axes = plt.subplots(
        2, len(steps), 
        figsize=(3.5*len(steps), 6), 
        layout="constrained"
    )
    
    # Handle case where there's only one step
    if len(steps) == 1:
        axes = axes.reshape(2, 1)
    
    for col, (step, pred_2d) in enumerate(zip(steps, preds)):
        # Prediction
        im = axes[0, col].imshow(
            pred_2d, 
            vmin=vmin, 
            vmax=vmax, 
            cmap=cmap, 
            origin="upper"
        )
        axes[0, col].set_title(f"Epoch {step}", fontsize=12)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        
        # Target (same for all columns)
        axes[1, col].imshow(
            truth_2d, 
            vmin=vmin, 
            vmax=vmax, 
            cmap=cmap, 
            origin="upper"
        )
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
    
    axes[0, 0].set_ylabel("Prediction", fontsize=14)
    axes[1, 0].set_ylabel("Target", fontsize=14)
    
    # Single shared colorbar on the right
    fig.colorbar(
        im, 
        ax=axes, 
        location="right", 
        shrink=0.9, 
        pad=0.02
    ).set_label("Value", fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved predictions plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes


def plot_model_predictions_over_time_1d(
    model,
    param_history,
    X_data,
    Y_data,
    p,
    steps=None,
    example_idx=None,
    save_path=None,
    show=False
):
    """
    Plot model predictions at different training steps vs ground truth (1D version).
    
    Args:
        model: The trained model
        param_history: List of parameter snapshots from training
        X_data: Input tensor (N, k, p)
        Y_data: Target tensor (N, p)
        p: Dimension
        steps: List of epoch indices to plot (default: [1, 5, 10, final])
        example_idx: Index of example to visualize (default: random)
        save_path: Path to save figure
        show: Whether to display the plot
    """
    import torch
    
    # Default steps
    if steps is None:
        final_step = len(param_history) - 1
        steps = [1, min(5, final_step), min(10, final_step), final_step]
        steps = sorted(list(set(steps)))
    
    # Random example if not specified
    if example_idx is None:
        example_idx = int(np.random.randint(len(Y_data)))
    
    device = next(model.parameters()).device
    model.to(device).eval()
    
    # Ground truth
    if Y_data.dim() == 3:
        Y_data = Y_data[:, -1, :]  # only final time step
    with torch.no_grad():
        truth_1d = Y_data[example_idx].cpu().numpy()
    
    # Collect predictions at each step
    preds = []
    for step in steps:
        model.load_state_dict(param_history[step], strict=True)
        with torch.no_grad():
            x = X_data[example_idx:example_idx+1].to(device)
            pred = model(x)
            if pred.dim() == 3:
                pred = pred[:, -1, :]  # only final time step
            pred_1d = pred.squeeze().detach().cpu().numpy()
            preds.append(pred_1d)
    
    # Plot: rows = [Prediction, Target], cols = time steps
    fig, axes = plt.subplots(
        2, len(steps),
        figsize=(3.5*len(steps), 4),
        layout="constrained"
    )
    
    # Handle case where there's only one step
    if len(steps) == 1:
        axes = axes.reshape(2, 1)
    
    x = np.arange(p)
    
    for col, (step, pred_1d) in enumerate(zip(steps, preds)):
        # Prediction
        axes[0, col].plot(x, pred_1d, 'b-', lw=2)
        axes[0, col].set_title(f"Epoch {step}", fontsize=12)
        axes[0, col].set_ylim(truth_1d.min() - 0.1 * np.abs(truth_1d.min()), 
                               truth_1d.max() + 0.1 * np.abs(truth_1d.max()))
        axes[0, col].set_xticks([])
        axes[0, col].grid(True, alpha=0.3)
        
        # Target (same for all columns)
        axes[1, col].plot(x, truth_1d, 'k-', lw=2)
        axes[1, col].set_ylim(truth_1d.min() - 0.1 * np.abs(truth_1d.min()), 
                               truth_1d.max() + 0.1 * np.abs(truth_1d.max()))
        axes[1, col].set_xticks([])
        axes[1, col].grid(True, alpha=0.3)
    
    axes[0, 0].set_ylabel("Prediction", fontsize=14)
    axes[1, 0].set_ylabel("Target", fontsize=14)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved predictions plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes


def plot_prediction_power_spectrum_over_time(
    model,
    param_history,
    X_data,
    Y_data,
    template_2d,
    p1,
    p2,
    loss_history,
    param_save_indices=None,
    num_freqs_to_track=10,
    checkpoint_indices=None,
    num_samples=100,
    save_path=None,
    show=False
):
    """
    Plot training loss with power spectrum analysis of predictions over time.
    
    Creates a two-panel plot:
    - Top: Training loss with colored bands for theory lines
    - Bottom: Power in tracked frequencies over time (computed at ALL saved checkpoints)
    
    Args:
        model: The trained model
        param_history: List of parameter snapshots (includes epoch 0)
        X_data: Input tensor (N, k, p1*p2)
        Y_data: Target tensor (N, p1*p2) - used to compute loss history
        template_2d: The 2D template array
        p1, p2: Dimensions
        loss_history: List of loss values over training steps/epochs
        param_save_indices: List of step/epoch numbers where params were saved (for x-axis alignment)
        num_freqs_to_track: Number of top frequencies to track
        checkpoint_indices: (deprecated/unused) - now analyzes ALL checkpoints
        num_samples: Number of samples to average for power computation
        save_path: Path to save figure
        show: Whether to display the plot
    """
    import torch
    from matplotlib.ticker import FormatStrFormatter
    from tqdm import tqdm
    
    device = next(model.parameters()).device
    
    # Identify top-K frequencies from template
    tracked_freqs = topk_template_freqs(template_2d, K=num_freqs_to_track)
    template_power_2d = get_power_2d_adele(template_2d, no_freq=True)
    target_powers = {(kx, ky): template_power_2d[kx, ky] for (kx, ky) in tracked_freqs}
    
    # Analyze ALL saved parameter checkpoints for full temporal resolution
    T = len(param_history)

    steps_analysis = list(range(len(param_history)))  # Analyze ALL saved params
    
    # Get the actual step/epoch numbers for x-axis plotting
    if param_save_indices is not None:
        actual_steps = param_save_indices  # All the actual step numbers
    else:
        actual_steps = list(range(len(param_history)))  # If None, indices = steps
    
    # Track average output power at those frequencies over training
    powers_over_time = {freq: [] for freq in tracked_freqs}
    
    print(f"  Analyzing {len(steps_analysis)} checkpoints for power spectrum...")
    
    with torch.no_grad():
        for step in tqdm(steps_analysis, desc="  Computing power spectra", leave=False):
            model.load_state_dict(param_history[step], strict=True)
            model.eval()
            
            # Get predictions for a batch
            outputs_flat = (
                model(X_data[:num_samples].to(device)).detach().cpu().numpy()
            )  # (num_samples, p1*p2)
            
            # Compute power spectrum for each sample, then average
            powers_batch = []
            for i in range(outputs_flat.shape[0]):
                if outputs_flat.ndim == 3:
                    out_2d = outputs_flat[i][-1,:] # only final time step
                else:
                    out_2d = outputs_flat[i]
                out_2d = out_2d.reshape(p1, p2)
                power_i = get_power_2d_adele(out_2d, no_freq=True)  # (p1, p2//2+1)
                powers_batch.append(power_i)
            avg_power = np.mean(powers_batch, axis=0)  # (p1, p2//2+1)
            
            # Record power at each tracked frequency
            for kx, ky in tracked_freqs:
                powers_over_time[(kx, ky)].append(avg_power[kx, ky])
    
    # Convert lists to arrays
    for freq in tracked_freqs:
        powers_over_time[freq] = np.array(powers_over_time[freq])
    
    if param_save_indices is None:
        # Assume params were saved at every step (old behavior)
        loss_epochs = np.arange(len(param_history))
        loss_history_subset = loss_history
    else:
        # Use the provided indices
        loss_epochs = np.array(param_save_indices)
        # Extract only the loss values at those indices
        loss_history_subset = [loss_history[i] for i in param_save_indices]
    
    # --- Create the plot ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_freqs)))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.10, hspace=0.12)
    
    # --- Top panel: Training loss with theory bands ---
    ax1.plot(loss_epochs, loss_history_subset, lw=4, color='#1f77b4', label='Training Loss')
    
    # Compute power spectrum of template for theory lines
    _, _, power = get_power_2d_adele(template_2d)
    power_flat = np.sort(power.flatten()[power.flatten() > 1e-20])[::-1]
    
    # Theory levels (cumulative tail sums)
    alpha_values = np.array([np.sum(power_flat[k:]) for k in range(len(power_flat))])
    coef = 1.0 / (p1 * p2)
    y_levels = coef * alpha_values  # strictly decreasing
    
    # Shade horizontal bands between successive theory lines
    n_bands = min(len(tracked_freqs), len(y_levels) - 1)
    for i in range(n_bands):
        y_top = y_levels[i]
        y_bot = y_levels[i + 1]
        ax1.axhspan(y_bot, y_top, facecolor=colors[i], alpha=0.15, zorder=-3)
    
    # Draw the black theory lines
    for y in y_levels[:n_bands + 1]:
        ax1.axhline(y=y, color="black", linestyle="--", linewidth=2, zorder=-2)
    
    ax1.set_ylabel("Theory Loss Levels", fontsize=20)
    ax1.set_ylim(y_levels[n_bands], y_levels[0] * 1.1)
    style_axes(ax1)
    ax1.grid(False)
    ax1.tick_params(labelbottom=False)
    
    # --- Bottom panel: Tracked mode power over time ---
    for i, (kx, ky) in enumerate(tracked_freqs):
        ax2.plot(
            actual_steps,  # Use actual step/epoch numbers, not indices
            powers_over_time[(kx, ky)], 
            color=colors[i], 
            lw=3,
            label=f"({kx},{ky})"
        )
        ax2.axhline(
            target_powers[(kx, ky)],
            color=colors[i],
            linestyle="dotted",
            linewidth=2,
            alpha=0.5,
        )
    
    ax2.set_xlabel("Steps", fontsize=20)
    ax2.set_ylabel("Power in Prediction", fontsize=20)
    ax2.grid(True, alpha=0.3)
    style_axes(ax2)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved power spectrum plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, (ax1, ax2), powers_over_time, tracked_freqs


def plot_prediction_power_spectrum_over_time_1d(
    model,
    param_history,
    X_data,
    Y_data,
    template_1d,
    p,
    loss_history,
    param_save_indices=None,
    num_freqs_to_track=10,
    checkpoint_indices=None,
    num_samples=100,
    save_path=None,
    show=False
):
    """
    Plot training loss with power spectrum analysis of predictions over time (1D version).
    
    Creates a two-panel plot:
    - Top: Training loss with colored bands for theory lines
    - Bottom: Power in tracked frequencies over time (computed at ALL saved checkpoints)
    
    Args:
        model: The trained model
        param_history: List of parameter snapshots (includes epoch 0)
        X_data: Input tensor (N, k, p)
        Y_data: Target tensor (N, p)
        template_1d: The 1D template array (p,)
        p: Dimension of the template
        loss_history: List of loss values over training steps/epochs
        param_save_indices: List of step/epoch numbers where params were saved
        num_freqs_to_track: Number of top frequencies to track
        checkpoint_indices: (deprecated/unused) - now analyzes ALL checkpoints
        num_samples: Number of samples to average for power computation
        save_path: Path to save figure
        show: Whether to display the plot
    """
    import torch
    from matplotlib.ticker import FormatStrFormatter
    from tqdm import tqdm
    
    device = next(model.parameters()).device
    
    # Identify top-K frequencies from template
    tracked_freqs = topk_template_freqs_1d(template_1d, K=num_freqs_to_track)
    template_power, _ = get_power_1d(template_1d)
    target_powers = {k: template_power[k] for k in tracked_freqs}
    
    # Analyze ALL saved parameter checkpoints
    T = len(param_history)
    steps_analysis = list(range(len(param_history)))
    
    # Get the actual step/epoch numbers for x-axis
    if param_save_indices is not None:
        actual_steps = param_save_indices
    else:
        actual_steps = list(range(len(param_history)))
    
    # Track average output power at those frequencies over training
    powers_over_time = {freq: [] for freq in tracked_freqs}
    
    print(f"  Analyzing {len(steps_analysis)} checkpoints for power spectrum (1D)...")
    
    with torch.no_grad():
        for step in tqdm(steps_analysis, desc="  Computing power spectra", leave=False):
            model.load_state_dict(param_history[step], strict=True)
            model.eval()
            
            # Get predictions for a batch
            outputs_flat = (
                model(X_data[:num_samples].to(device)).detach().cpu().numpy()
            )  # (num_samples, p)
            
            # Compute power spectrum for each sample, then average
            powers_batch = []
            for i in range(outputs_flat.shape[0]):
                if outputs_flat.ndim == 3:
                    out_1d = outputs_flat[i, -1, :]  # only final time step
                else:
                    out_1d = outputs_flat[i]
                power_i, _ = get_power_1d(out_1d)
                powers_batch.append(power_i)
            avg_power = np.mean(powers_batch, axis=0)  # (p//2+1,)
            
            # Record power at each tracked frequency
            for k in tracked_freqs:
                powers_over_time[k].append(avg_power[k])
    
    # Convert lists to arrays
    for freq in tracked_freqs:
        powers_over_time[freq] = np.array(powers_over_time[freq])
    
    if param_save_indices is None:
        loss_epochs = np.arange(len(param_history))
        loss_history_subset = loss_history
    else:
        loss_epochs = np.array(param_save_indices)
        loss_history_subset = [loss_history[i] for i in param_save_indices]
    
    # --- Create the plot ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_freqs)))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.10, hspace=0.12)
    
    # --- Top panel: Training loss with theory bands ---
    ax1.plot(loss_epochs, loss_history_subset, lw=4, color='#1f77b4', label='Training Loss')
    
    # Compute power spectrum of template for theory lines
    power, _ = get_power_1d(template_1d)
    power_sorted = np.sort(power[power > 1e-20])[::-1]
    
    # Theory levels (cumulative tail sums)
    alpha_values = np.array([np.sum(power_sorted[k:]) for k in range(len(power_sorted))])
    coef = 1.0 / p
    y_levels = coef * alpha_values  # strictly decreasing
    
    # Shade horizontal bands between successive theory lines
    n_bands = min(len(tracked_freqs), len(y_levels) - 1)
    for i in range(n_bands):
        y_top = y_levels[i]
        y_bot = y_levels[i + 1]
        ax1.axhspan(y_bot, y_top, facecolor=colors[i], alpha=0.15, zorder=-3)
    
    # Draw the black theory lines
    for y in y_levels[:n_bands + 1]:
        ax1.axhline(y=y, color="black", linestyle="--", linewidth=2, zorder=-2)
    
    ax1.set_ylabel("Theory Loss Levels", fontsize=20)
    ax1.set_ylim(y_levels[n_bands], y_levels[0] * 1.1)
    style_axes(ax1)
    ax1.grid(False)
    ax1.tick_params(labelbottom=False)
    
    # --- Bottom panel: Tracked mode power over time ---
    for i, k in enumerate(tracked_freqs):
        ax2.plot(
            actual_steps,
            powers_over_time[k], 
            color=colors[i], 
            lw=3,
            label=f"k={k}"
        )
        ax2.axhline(
            target_powers[k],
            color=colors[i],
            linestyle="dotted",
            linewidth=2,
            alpha=0.5,
        )
    
    ax2.set_xlabel("Steps", fontsize=20)
    ax2.set_ylabel("Power in Prediction", fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best', ncol=2)
    style_axes(ax2)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved power spectrum plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, (ax1, ax2), powers_over_time, tracked_freqs



def plot_fourier_modes_reference(
    tracked_freqs,
    colors,
    p1,
    p2,
    save_path=None,
    save_individual=False,
    individual_dir=None,
    show=False
):
    """
    Create a reference visualization of tracked Fourier modes.
    
    Generates a stacked vertical image showing all tracked frequency modes
    with colored borders matching the power spectrum analysis.
    
    Args:
        tracked_freqs: List of (kx, ky) tuples for tracked frequencies
        colors: Array of colors for each frequency (from plt.cm.tab10 or similar)
        p1, p2: Dimensions of the template
        save_path: Path to save the stacked visualization
        save_individual: Whether to also save individual mode images
        individual_dir: Directory for individual mode images (if save_individual=True)
        show: Whether to display the plot
    
    Returns:
        fig: The matplotlib figure
    """
    import matplotlib.patheffects as pe
    import matplotlib.gridspec as gridspec
    from pathlib import Path
    
    # --- Save individual mode images (optional) ---
    if save_individual and individual_dir is not None:
        individual_dir = Path(individual_dir)
        individual_dir.mkdir(exist_ok=True)
        
        for i, (kx, ky) in enumerate(tracked_freqs):
            img = _fourier_mode_2d(p1, p2, kx, ky)
            
            fig_ind, ax = plt.subplots(figsize=(3.2, 2.2))
            ax.imshow(img, cmap="RdBu_r", origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Colored border
            for side in ("left", "right", "top", "bottom"):
                ax.spines[side].set_edgecolor(colors[i])
                ax.spines[side].set_linewidth(8)
            
            # Frequency label
            kx_label = _pretty_k(kx, p2)
            ky_label = _pretty_k(ky, p1)
            ax.text(
                0.5, 0.5,
                f"$k=({kx_label},{ky_label})$",
                color=colors[i],
                fontsize=25,
                fontweight="bold",
                ha="center",
                va="center",
                transform=ax.transAxes,
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)],
            )
            
            plt.tight_layout()
            
            # Save with signed indices in filename
            kx_signed, ky_signed = _signed_k(kx, p2), _signed_k(ky, p1)
            base = f"mode_{i:03d}_kx{kx}_ky{ky}_signed_{kx_signed}_{ky_signed}"
            fig_ind.savefig(individual_dir / f"{base}.png", dpi=300, bbox_inches="tight")
            np.save(individual_dir / f"{base}.npy", img)
            plt.close(fig_ind)
        
        print(f"  ✓ Saved {len(tracked_freqs)} individual mode images to {individual_dir}")
    
    # --- Create stacked vertical visualization ---
    n = len(tracked_freqs)
    
    # Panel geometry and spacing
    panel_h_in = 2.2
    gap_h_in = 0.35  # whitespace between rows
    fig_w_in = 4.6
    fig_h_in = n * panel_h_in + (n - 1) * gap_h_in
    
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=150)
    
    # Rows alternate: [panel, gap, panel, gap, ..., panel]
    rows = 2 * n - 1
    height_ratios = []
    for i in range(n):
        height_ratios.append(panel_h_in)
        if i < n - 1:
            height_ratios.append(gap_h_in)
    
    # Layout: image on LEFT, label on RIGHT
    gs = gridspec.GridSpec(
        nrows=rows, ncols=2,
        width_ratios=[1.0, 0.46],
        height_ratios=height_ratios,
        wspace=0.0, hspace=0.0
    )
    
    for i, (kx, ky) in enumerate(tracked_freqs):
        r = 2 * i  # even rows are content; odd rows are spacers
        
        # Image axis (left)
        ax_img = fig.add_subplot(gs[r, 0])
        img = _fourier_mode_2d(p1, p2, kx, ky)
        ax_img.imshow(img, cmap="RdBu_r", origin="upper", aspect="equal")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # Colored border around the image
        for side in ("left", "right", "top", "bottom"):
            ax_img.spines[side].set_edgecolor(colors[i])
            ax_img.spines[side].set_linewidth(8)
        
        # Label axis (right)
        ax_label = fig.add_subplot(gs[r, 1])
        ax_label.set_axis_off()
        kx_label = _pretty_k(kx, p2)
        ky_label = _pretty_k(ky, p1)
        ax_label.text(
            0.0, 0.5, f"$k=({kx_label},{ky_label})$",
            color=colors[i], fontsize=45, fontweight="bold",
            ha="left", va="center", transform=ax_label.transAxes,
            path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)]
        )
    
    # Adjust to prevent clipping of thick borders
    fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.015)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.12)
        print(f"  ✓ Saved Fourier modes reference to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def plot_wout_neuron_specialization(
    param_history,
    tracked_freqs,
    colors,
    p1,
    p2,
    steps=None,
    dead_thresh_l2=0.25,
    save_dir=None,
    show=False
):
    """
    Visualize W_out neurons colored by their dominant tracked frequency.
    
    Creates grid visualizations of output weight neurons at different training steps,
    with colored borders indicating which Fourier mode each neuron is tuned to.
    
    Args:
        param_history: List of parameter snapshots from training
        tracked_freqs: List of (kx, ky) tuples for tracked frequencies
        colors: Array of colors for each frequency (from plt.cm.tab10 or similar)
        p1, p2: Dimensions of the template
        steps: List of epoch indices to plot (default: [1, 5, final])
        dead_thresh_l2: L2 norm threshold below which neurons are considered "dead"
        save_dir: Directory to save figures (Path object)
        show: Whether to display the plots
    
    Returns:
        List of figure objects
    """
    from matplotlib.patches import Patch
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    import matplotlib.gridspec as gridspec
    from pathlib import Path
    
    # Default steps
    if steps is None:
        final_step = len(param_history) - 1
        steps = [1, min(5, final_step), final_step]
        steps = sorted(list(set(steps)))
    
    # Get dimensions
    W0 = param_history[steps[0]]["W_out"].detach().cpu().numpy().T  # (H, D)
    H, D = W0.shape
    assert p1 * p2 == D, f"p1*p2 ({p1*p2}) must equal D ({D})."
    
    # Compute global color limits across all steps
    vmin, vmax = np.inf, -np.inf
    for step in steps:
        W = param_history[step]["W_out"].detach().cpu().numpy().T
        vmin = min(vmin, W.min())
        vmax = max(vmax, W.max())
    
    # Grid layout
    R_ner, C_ner = _squareish_grid(H)
    tile_w, tile_h = 2, 2  # inches per neuron tile
    figsize = (C_ner * tile_w, R_ner * tile_h)
    
    heat_cmap = "RdBu_r"
    border_lw = 5.0
    dead_color = (0.6, 0.6, 0.6, 1.0)
    
    figures = []
    
    # Create one figure per time step
    for step in steps:
        W = param_history[step]["W_out"].detach().cpu().numpy().T  # (H, D)
        
        # Determine dominant frequency for each neuron
        dom_idx = np.empty(H, dtype=int)
        l2 = np.linalg.norm(W, axis=1)
        dead_mask = l2 < dead_thresh_l2
        
        for j in range(H):
            m = W[j].reshape(p1, p2)
            F = np.fft.fft2(m)
            P = (F.conj() * F).real
            tp = [_tracked_power_from_fft2(P, kx, ky, p1, p2) 
                  for (kx, ky) in tracked_freqs]
            dom_idx[j] = int(np.argmax(tp))
        
        # Assign colors
        edge_colors = colors[dom_idx].copy()
        edge_colors[dead_mask] = dead_color
        
        # Build figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(R_ner, C_ner, figure=fig, wspace=0.06, hspace=0.06)
        
        # Plot neuron tiles
        for j in range(R_ner * C_ner):
            ax = fig.add_subplot(gs[j // C_ner, j % C_ner])
            if j < H:
                m = W[j].reshape(p1, p2)
                ax.imshow(
                    m, vmin=vmin, vmax=vmax, origin="lower", 
                    aspect="equal", cmap=heat_cmap
                )
                # Colored border
                ec = edge_colors[j]
                for sp in ax.spines.values():
                    sp.set_edgecolor(ec)
                    sp.set_linewidth(border_lw)
            else:
                ax.axis("off")
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        if save_dir:
            save_path = Path(save_dir) / f"wout_neurons_epoch_{step:04d}.pdf"
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
            print(f"  ✓ Saved W_out visualization for epoch {step}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        figures.append(fig)
    
    # Create standalone colorbar figure
    fig_cb = plt.figure(figsize=(6, 1.2))
    ax_cb = fig_cb.add_axes([0.1, 0.35, 0.8, 0.3])
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=heat_cmap)
    cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cbar.set_label("Weight value", fontsize=12)
    
    if save_dir:
        save_path = Path(save_dir) / "wout_colorbar.pdf"
        fig_cb.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved colorbar")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    figures.append(fig_cb)
    
    # Create standalone legend figure
    fig_legend = plt.figure(figsize=(6, 2.0))
    ax_leg = fig_legend.add_subplot(111)
    ax_leg.axis("off")
    
    # Colored edge patches (matching tile borders)
    handles = [
        Patch(
            facecolor="white", 
            edgecolor=colors[i], 
            linewidth=2.5, 
            label=f"k={tracked_freqs[i]}"
        )
        for i in range(len(tracked_freqs))
    ]
    handles.append(
        Patch(
            facecolor="white", 
            edgecolor=dead_color, 
            linewidth=2.5, 
            label="dead"
        )
    )
    
    ax_leg.legend(
        handles=handles, 
        ncol=min(4, len(handles)), 
        frameon=True, 
        loc="center", 
        title="Dominant frequency",
        fontsize=10
    )
    
    if save_dir:
        save_path = Path(save_dir) / "wout_legend.pdf"
        fig_legend.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved legend")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    figures.append(fig_legend)
    
    return figures


def plot_wout_neuron_specialization_1d(
    param_history,
    tracked_freqs,
    colors,
    p,
    steps=None,
    dead_thresh_l2=0.25,
    save_dir=None,
    show=False
):
    """
    Visualize W_out neurons colored by their dominant tracked frequency (1D version).
    
    Creates visualizations of output weight neurons at different training steps,
    with colored borders indicating which Fourier mode each neuron is tuned to.
    For 1D, neurons are shown as line plots.
    
    Args:
        param_history: List of parameter snapshots from training
        tracked_freqs: List of frequency indices (integers)
        colors: Array of colors for each frequency
        p: Dimension of the template
        steps: List of epoch indices to plot (default: [1, 5, final])
        dead_thresh_l2: L2 norm threshold below which neurons are considered "dead"
        save_dir: Directory to save figures (Path object)
        show: Whether to display the plots
    
    Returns:
        List of figure objects
    """
    from matplotlib.patches import Patch
    from pathlib import Path
    
    def tracked_power_from_fft(power1d, k):
        """Get power at frequency k."""
        return float(power1d[k])
    
    # Default steps
    if steps is None:
        final_step = len(param_history) - 1
        steps = [1, min(5, final_step), final_step]
        steps = sorted(list(set(steps)))
    
    # Get dimensions
    W0 = param_history[steps[0]]["W_out"].detach().cpu().numpy().T  # (H, p)
    H, D = W0.shape
    assert p == D, f"p ({p}) must equal D ({D})."
    
    figures = []
    
    # Create one figure per time step
    for step in steps:
        W = param_history[step]["W_out"].detach().cpu().numpy().T  # (H, p)
        
        # Determine dominant frequency for each neuron
        dom_idx = np.empty(H, dtype=int)
        l2 = np.linalg.norm(W, axis=1)
        dead_mask = l2 < dead_thresh_l2
        
        for j in range(H):
            neuron_weights = W[j]
            power, _ = get_power_1d(neuron_weights)
            tp = [tracked_power_from_fft(power, k) for k in tracked_freqs]
            dom_idx[j] = int(np.argmax(tp))
        
        # Assign colors
        edge_colors = colors[dom_idx].copy()
        edge_colors[dead_mask] = (0.6, 0.6, 0.6, 1.0)
        
        # Create grid of subplots
        ncols = min(6, H)
        nrows = int(np.ceil(H / ncols))
        
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(2.5 * ncols, 1.5 * nrows),
            squeeze=False
        )
        
        x = np.arange(p)
        
        for j in range(nrows * ncols):
            row = j // ncols
            col = j % ncols
            ax = axes[row, col]
            
            if j < H:
                # Plot neuron weights
                ax.plot(x, W[j], color=edge_colors[j], lw=1.5)
                ax.set_xlim(0, p - 1)
                ax.set_ylim(W.min(), W.max())
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(edge_colors[j])
                    spine.set_linewidth(3)
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f"wout_neurons_1d_epoch_{step:04d}.pdf"
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
            print(f"  ✓ Saved W_out 1D visualization for epoch {step}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        figures.append(fig)
    
    # Create legend figure
    fig_legend = plt.figure(figsize=(8, 2.0))
    ax_leg = fig_legend.add_subplot(111)
    ax_leg.axis("off")
    
    handles = [
        Patch(
            facecolor="white",
            edgecolor=colors[i],
            linewidth=2.5,
            label=f"k={tracked_freqs[i]}"
        )
        for i in range(len(tracked_freqs))
    ]
    handles.append(
        Patch(
            facecolor="white",
            edgecolor=(0.6, 0.6, 0.6, 1.0),
            linewidth=2.5,
            label="dead"
        )
    )
    
    ax_leg.legend(
        handles=handles,
        ncol=min(5, len(handles)),
        frameon=True,
        loc="center",
        title="Dominant frequency",
        fontsize=10
    )
    
    if save_dir:
        save_path = Path(save_dir) / "wout_legend_1d.pdf"
        fig_legend.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved legend")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    figures.append(fig_legend)
    
    return figures

def analyze_wout_frequency_dominance(state_dict, tracked_freqs, p1, p2):
    """
    Analyze W_out to find dominant frequency for each neuron.
    
    Args:
        state_dict: Model parameters (expects 'W_out' key)
        tracked_freqs: List of (kx, ky) tuples
        p1, p2: Template dimensions
    
    Returns:
        dom_idx: Dominant frequency index for each neuron
        phase: Phase at dominant frequency for each neuron
        dom_power: Power at dominant frequency for each neuron
        l2: L2 norm of each neuron's weights
    """
    Wo = state_dict["W_out"].detach().cpu().numpy()  # (p, H)
    W = Wo.T  # (H, p)
    H, D = W.shape
    assert D == p1 * p2
    
    dom_idx = np.empty(H, dtype=int)
    dom_pow = np.empty(H, dtype=float)
    phase = np.empty(H, dtype=float)
    l2 = np.linalg.norm(W, axis=1)
    
    for j in range(H):
        m = W[j].reshape(p1, p2)
        F = np.fft.fft2(m)
        P = (F.conj() * F).real
        # Power at tracked frequencies
        tp = [_tracked_power_from_fft2(P, kx, ky, p1, p2) 
              for (kx, ky) in tracked_freqs]
        jj = int(np.argmax(tp))
        dom_idx[j] = jj
        # Phase at representative bin
        i0, j0 = tracked_freqs[jj][0] % p1, tracked_freqs[jj][1] % p2
        phase[j] = np.angle(F[i0, j0])
        dom_pow[j] = tp[jj]
    
    return dom_idx, phase, dom_pow, l2


def plot_wmix_frequency_structure(
    param_history,
    tracked_freqs,
    colors,
    p1,
    p2,
    steps=None,
    within_group_order="phase",
    dead_l2_thresh=0.1,
    save_path=None,
    show=False
):
    """
    Visualize W_mix structure grouped by W_out frequency specialization.
    
    Creates heatmaps of W_mix reordered to show block structure based on
    which Fourier mode each neuron is tuned to in W_out.
    
    Args:
        param_history: List of parameter snapshots
        tracked_freqs: List of (kx, ky) frequency tuples
        colors: Array of colors for each frequency
        p1, p2: Template dimensions
        steps: List of epoch indices to plot (default: [1, 5, final])
        within_group_order: How to order neurons within each frequency group
                           ('phase', 'power', 'phase_power', 'none')
        dead_l2_thresh: L2 threshold for dead neurons
        save_path: Path to save figure
        show: Whether to display plot
    
    Returns:
        fig, axes
    """
    from matplotlib.patches import Rectangle
    
    # Default steps
    if steps is None:
        final_step = len(param_history) - 1
        steps = [1, min(5, final_step), final_step]
        steps = sorted(list(set(steps)))
    
    # Labels for frequencies
    tracked_labels = [
        ("DC" if (kx, ky) == (0, 0) else f"({kx},{ky})") 
        for (kx, ky) in tracked_freqs
    ]
    
    # Analyze and reorder for each step
    Wmix_perm_list = []
    group_info_list = []
    
    for s in steps:
        sd = param_history[s]
        
        # Analyze W_out
        dom_idx, phase, dom_power, l2 = analyze_wout_frequency_dominance(
            sd, tracked_freqs, p1, p2
        )
        
        # Get W_mix (fallback to W_h for compatibility)
        if "W_mix" in sd:
            M = sd["W_mix"].detach().cpu().numpy()
        elif "W_h" in sd:
            M = sd["W_h"].detach().cpu().numpy()
        else:
            raise KeyError("Neither 'W_mix' nor 'W_h' found in state dict.")
        
        # Compute permutation
        perm, group_keys, boundaries = _permutation_from_groups_with_dead(
            dom_idx, phase, dom_power, l2,
            within=within_group_order,
            dead_l2_thresh=dead_l2_thresh
        )
        
        # Reorder
        M_perm = M[perm][:, perm]
        Wmix_perm_list.append(M_perm)
        group_info_list.append((group_keys, boundaries))
    
    # Shared color limits
    vmax = max(np.max(np.abs(M)) for M in Wmix_perm_list)
    vmin = -vmax if vmax > 0 else 0.0
    
    # Create figure
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.8), constrained_layout=True)
    if n == 1:
        axes = [axes]
    
    cmap = "RdBu_r"
    dead_gray = "0.35"
    
    im = None
    for j, (s, M_perm) in enumerate(zip(steps, Wmix_perm_list)):
        ax = axes[j]
        im = ax.imshow(
            M_perm, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="nearest"
        )
        
        ax.set_yticks([])
        ax.tick_params(axis="x", bottom=False)
        
        group_keys, boundaries = group_info_list[j]
        
        # Draw separators between groups
        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color="k", lw=0.9, alpha=0.65)
            ax.axvline(b - 0.5, color="k", lw=0.9, alpha=0.65)
        
        # Draw colored boxes around frequency groups
        starts = [0] + boundaries[:-1]
        ends = [b - 1 for b in boundaries]
        for kk, s0, e0 in zip(group_keys, starts, ends):
            if kk == -1:  # Skip dead neurons
                continue
            size = e0 - s0 + 1
            rect = Rectangle(
                (s0 - 0.5, s0 - 0.5),
                width=size, height=size,
                fill=False,
                linewidth=2.0,
                edgecolor=colors[kk],
                alpha=0.95,
                joinstyle="miter"
            )
            ax.add_patch(rect)
        
        # Add labels at top
        centers = [(s + e) / 2.0 for s, e in zip(starts, ends)]
        sizes = [e - s + 1 for s, e in zip(starts, ends)]
        
        labels = []
        label_colors = []
        for kk, nn in zip(group_keys, sizes):
            if kk == -1:
                labels.append(f"DEAD\n(n={nn})")
                label_colors.append(dead_gray)
            else:
                labels.append(f"{tracked_labels[kk]}\n(n={nn})")
                label_colors.append(colors[kk])
        
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, fontsize=11, ha="center")
        ax.tick_params(
            axis="x", bottom=False, top=True,
            labelbottom=False, labeltop=True, labelsize=11
        )
        for lbl, clr in zip(ax.get_xticklabels(), label_colors):
            lbl.set_color(clr)
        
        ax.set_xlabel(f"Epoch {s}", fontsize=18, labelpad=8)
    
    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=1.0, pad=0.012, aspect=18)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Weight value", fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"  ✓ Saved W_mix structure plot")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes