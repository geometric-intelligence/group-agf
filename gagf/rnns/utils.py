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

    # # Scientific notation formatting
    # if ax.get_yscale() == "linear":
    #     ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
    # if ax.get_xscale() == "linear":
    #     ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 2))

    ax.xaxis.offsetText.set_fontsize(20)
    ax.grid()

    # Customize spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(3)


def plot_2d_signal(
    ax,
    signal_2d,
    title="",
    cmap="RdBu_r",
    colorbar=True,
):
    """Plot a 2D signal as a heatmap."""
    im = ax.imshow(signal_2d, cmap=cmap, aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("y", fontsize=12)
    ax.set_ylabel("x", fontsize=12)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


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

def get_power_2d(points_2d):
    """
    Compute 2D power spectrum using rfft2 (for real-valued inputs).

    Args:
        points_2d: (p1, p2) array

    Returns:
        power: (p1//2+1, p2) array of power values
        freqs_x: frequency indices along x
        freqs_y: frequency indices along y
    """
    p1, p2 = points_2d.shape

    # Perform 2D FFT (rfft2 for real input)
    ft = np.fft.fft2(points_2d)  # shape: (p1//2+1, p2) for rfft, (p1, p2) for fft
    power = np.abs(ft) ** 2 / (p1 * p2)

    freqs_x = np.fft.fftfreq(p1, 1.0) * p1  # or np.arange(p1) for index-based
    freqs_y = np.fft.fftfreq(p2, 1.0) * p2  # or np.arange(p2) for index-based

    return power, freqs_x, freqs_y



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



def plot_training_loss_with_theory(
    loss_history, 
    template_2d, 
    p1, 
    p2, 
    save_path=None,
    show=True
):
    """
    Plot training loss with theoretical power spectrum lines.
    
    Args:
        loss_history: List of loss values over epochs
        template_2d: The 2D template array (p1, p2)
        p1, p2: Dimensions
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot loss
    epochs = np.arange(len(loss_history))
    ax.plot(epochs, loss_history, lw=4, color='#1f77b4', label='Training Loss')
    
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
    
    ax.set_xlabel("Epochs", fontsize=24)
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