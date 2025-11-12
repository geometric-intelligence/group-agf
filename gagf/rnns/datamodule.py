import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


from torch.utils.data import IterableDataset

class OnlineModularAdditionDataset2D(IterableDataset):
    """
    Online dataset that generates 2D modular addition samples on-the-fly.
    Fully GPU-accelerated for maximum throughput.
    """
    def __init__(
        self, 
        p1: int,
        p2: int, 
        template: np.ndarray,
        k: int,
        batch_size: int,
        device: str,
        return_all_outputs: bool = False,
    ):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.k = k
        self.batch_size = batch_size
        self.p_flat = p1 * p2
        self.device = device
        self.return_all_outputs = return_all_outputs
        
        # Store template on GPU for fast rolling
        self.template_gpu = torch.tensor(template, device=device, dtype=torch.float32)
        
        # Pre-compute coordinate grids on GPU for efficient rolling
        x_coords = torch.arange(p1, device=device)
        y_coords = torch.arange(p2, device=device)
        self.x_grid, self.y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
    def _roll_2d_batch(self, shifts_x, shifts_y):
        """
        Roll the template by different amounts for each sample in a batch.
        Fully vectorized on GPU.
        
        Args:
            shifts_x: (batch_size,) or (batch_size, k) tensor of row shifts
            shifts_y: (batch_size,) or (batch_size, k) tensor of col shifts
        
        Returns:
            Rolled templates: (batch_size, p1, p2) or (batch_size, k, p1, p2)
        """
        # Determine output shape based on input
        if shifts_x.dim() == 1:
            # Single roll per sample: (batch_size,)
            batch_size = shifts_x.shape[0]
            # Broadcast: (1, p1, p2) -> (batch_size, p1, p2)
            x_grid = self.x_grid.unsqueeze(0)  # (1, p1, p2)
            y_grid = self.y_grid.unsqueeze(0)  # (1, p1, p2)
            shifts_x = shifts_x.view(batch_size, 1, 1)  # (batch_size, 1, 1)
            shifts_y = shifts_y.view(batch_size, 1, 1)  # (batch_size, 1, 1)
        else:
            # Multiple rolls per sample: (batch_size, k)
            batch_size, k = shifts_x.shape
            # Broadcast: (1, 1, p1, p2) -> (batch_size, k, p1, p2)
            x_grid = self.x_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, p1, p2)
            y_grid = self.y_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, p1, p2)
            shifts_x = shifts_x.view(batch_size, k, 1, 1)  # (batch_size, k, 1, 1)
            shifts_y = shifts_y.view(batch_size, k, 1, 1)  # (batch_size, k, 1, 1)
        
        # Compute shifted coordinates with modular arithmetic
        x_shifted = (x_grid - shifts_x) % self.p1
        y_shifted = (y_grid - shifts_y) % self.p2
        
        # Index into template using advanced indexing
        rolled = self.template_gpu[x_shifted.long(), y_shifted.long()]
        
        return rolled
        
    def __iter__(self):
        """Generate batches indefinitely on GPU."""
        while True:
            # Generate random shifts on GPU: (batch_size, k)
            shifts_x = torch.randint(0, self.p1, (self.batch_size, self.k), 
                                    device=self.device, dtype=torch.long)
            shifts_y = torch.randint(0, self.p2, (self.batch_size, self.k), 
                                    device=self.device, dtype=torch.long)
            
            # Generate X: roll template for each time step
            # Shape: (batch_size, k, p1, p2)
            X_rolled = self._roll_2d_batch(shifts_x, shifts_y)
            
            # Reshape to (batch_size, k, p_flat)
            X = X_rolled.reshape(self.batch_size, self.k, self.p_flat)
            
            if self.return_all_outputs:
                # Generate Y for ALL cumulative sums (intermediate targets)
                # Compute cumulative sum at each timestep
                sx_cumsum = torch.cumsum(shifts_x, dim=1) % self.p1  # (batch_size, k)
                sy_cumsum = torch.cumsum(shifts_y, dim=1) % self.p2  # (batch_size, k)
                
                # Roll by all cumulative sums: (batch_size, k, p1, p2)
                Y_rolled = self._roll_2d_batch(sx_cumsum, sy_cumsum)
                
                # Reshape to (batch_size, k, p_flat)
                Y = Y_rolled.reshape(self.batch_size, self.k, self.p_flat)
                Y = Y[:, 1:, :]

            else:
                # Generate Y: only final cumulative sum (current behavior)
                sx_cumsum = shifts_x.sum(dim=1) % self.p1  # (batch_size,)
                sy_cumsum = shifts_y.sum(dim=1) % self.p2  # (batch_size,)
                
                # Shape: (batch_size, p1, p2)
                Y_rolled = self._roll_2d_batch(sx_cumsum, sy_cumsum)
                
                # Reshape to (batch_size, p_flat)
                Y = Y_rolled.reshape(self.batch_size, self.p_flat)
                
            yield X, Y


def build_modular_addition_sequence_dataset_2d(
    p1: int,
    p2: int,
    template: np.ndarray,
    k: int,
    mode: str = "sampled",
    num_samples: int = 65536,
    return_all_outputs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2D modular addition dataset.

    Args:
        p1: height (rows) dimension
        p2: width  (cols) dimension
        template: (p1, p2) template array
        k: sequence length
        mode: "sampled" or "exhaustive"
        num_samples: number of samples for "sampled" mode

    Returns:
        X:           (N, k, p1*p2) where token t is template rolled by (ax_t, ay_t), then flattened
        Y:           (N, k, p1*p2) target at each step, rolled by (sum_t ax_t mod p1, sum_t ay_t mod p2), flattened
        sequence_xy: (N, k, 2) integer group elements (ax_t, ay_t) per token (NOT cumulative)

    Notes:
        - axis 0 (rows) is shifted by ax; axis 1 (cols) by ay.
        - To get cumulative positions after each token, use `sequence_to_paths_xy(sequence_xy, p1, p2)`.
    """
    assert template.shape == (p1, p2), f"template must be ({p1}, {p2}), got {template.shape}"
    p_flat = p1 * p2

    if mode == "exhaustive":
        total = (p1 * p2) ** k
        if total > 1_000_000:
            raise ValueError(f"(p1*p2)**k = {total} is huge; use mode='sampled' instead.")
        N = total
        sequence_xy = np.zeros((N, k, 2), dtype=np.int64)
        for idx in range(N):
            for t in range(k):
                flat_idx = (idx // (p_flat ** t)) % p_flat
                ax = flat_idx // p2  # rows
                ay = flat_idx %  p2  # cols
                sequence_xy[idx, t, 0] = ax
                sequence_xy[idx, t, 1] = ay
    else:
        N = int(num_samples)
        sequence_xy = np.empty((N, k, 2), dtype=np.int64)
        sequence_xy[:, :, 0] = np.random.randint(0, p1, size=(N, k))  # ax (rows)
        sequence_xy[:, :, 1] = np.random.randint(0, p2, size=(N, k))  # ay (cols)

    X = np.zeros((N, k, p_flat), dtype=np.float32)
    Y = np.zeros((N, k, p_flat), dtype=np.float32)

    for i in range(N):
        sx, sy = 0, 0  # cumulative shift for Y
        for t in range(k):
            ax, ay = int(sequence_xy[i, t, 0]), int(sequence_xy[i, t, 1])
            rolled = np.roll(np.roll(template, shift=ax, axis=0), shift=ay, axis=1)
            X[i, t, :] = rolled.ravel()
            sx = (sx + ax) % p1
            sy = (sy + ay) % p2
            Y[i, t, :] = np.roll(np.roll(template, shift=sx, axis=0), shift=sy, axis=1).ravel()
    
    if not return_all_outputs:
        Y = Y[:, -1, :]

    return X, Y, sequence_xy


def sequence_to_paths_xy(sequence_xy: np.ndarray, p1: int, p2: int) -> np.ndarray:
    """
    Convert a sequence of group elements (ax_t, ay_t) into cumulative positions
    after each token, modulo (p1, p2).

    Args:
        sequence_xy: (N, k, 2) integers, NOT cumulative
        p1, p2: moduli for rows/cols

    Returns:
        paths_xy: (N, k, 2) where paths_xy[n, t] = (sum_{u<=t} ax_u mod p1, sum_{u<=t} ay_u mod p2)
    """
    seq = sequence_xy.astype(np.int64, copy=False)
    N, k, _ = seq.shape
    paths_xy = np.empty_like(seq)

    # cumulative along time axis, modulo each dimension
    paths_xy[:, :, 0] = np.mod(np.cumsum(seq[:, :, 0], axis=1, dtype=np.int64), p1)
    paths_xy[:, :, 1] = np.mod(np.cumsum(seq[:, :, 1], axis=1, dtype=np.int64), p2)
    return paths_xy

def mnist_template_2d(p1: int, p2: int, label: int, root: str = "data"):
    """
    Return a (p1, p2) template from a random MNIST image of the given class label (0–9).
    Values are float32 in [0, 1].
    """
    if not (0 <= int(label) <= 9):
        raise ValueError("label must be an integer in [0, 9].")

    ds = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    cls_idxs = (ds.targets == int(label)).nonzero(as_tuple=True)[0]
    if cls_idxs.numel() == 0:
        raise ValueError(f"No samples for label {label}.")

    idx = cls_idxs[torch.randint(len(cls_idxs), (1,)).item()].item()
    img, _ = ds[idx]  # img: (1, 28, 28) in [0,1]
    img = nn.functional.interpolate(img.unsqueeze(0), size=(p1, p2), mode="bilinear", align_corners=False)[0, 0]
    return img.numpy().astype(np.float32)  # (p1, p2)


### ----- SYNTHETIC TEMPLATES ----- ###

def gaussian_mixture_template(
    p1=20, 
    p2=20, 
    n_blobs=8, 
    frac_broad=0.7,
    sigma_broad=(3.5, 6.0), 
    sigma_narrow=(1.0, 2.0),
    amp_broad=1.0, 
    amp_narrow=0.5,
    seed=None, 
    normalize=True):
    """
    Build a (p1 x p2) template as a periodic mixture of Gaussians.
    Broad Gaussians (low-frequency) get higher weight; a few narrow ones add detail.
    """
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
            dy = np.minimum(np.abs(Y - cy), H - np.abs(Y - cy))  # periodic (torus) distance
            dx = np.minimum(np.abs(X - cx), W - np.abs(X - cx))
            out += amp * np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
        return out

    template = (
        add_blobs(k_broad, sigma_broad, amp_broad) +   # broad, low-freq power
        add_blobs(k_narrow, sigma_narrow, amp_narrow)  # a bit of high-freq detail
    )

    if normalize:
        template -= template.mean()
        s = template.std()
        if s > 1e-12:
            template /= s
    return template.astype(np.float32)

def generate_template_unique_freqs(p1, p2, n_freqs, amp_max=100, amp_min=10, seed=None):
    """
    Real (p1 x p2) template from n_freqs Fourier modes where:
      - No two selected bins are conjugates of each other.
      - Self-conjugate singletons are excluded.
      - Frequencies are chosen (low→high) by radial order from the rfft-style half-plane.

    Conjugate symmetry: F[ky,kx] = conj( F[-ky mod p1, -kx mod p2] ).
    On the rfft half-plane kx ∈ [0, p2//2]:
      - If 0 < kx < p2//2, the conjugate sits at kx' = p2 - kx (outside the half-plane) → safe.
      - If kx in {0, p2//2 (when even)}, the conjugate keeps the same kx and flips ky → avoid picking both ky and -ky.
      - Self-conjugate happens only if kx in {0, p2//2 (when even)} AND ky in {0, p1//2 (when even)} → exclude.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.zeros((p1, p2), dtype=np.complex128)

    # Helpers
    def ky_signed(ky):  # map ky ∈ [0..p1-1] to signed range
        return ky if ky <= p1 // 2 else ky - p1

    def is_self_conj(ky, kx):
        on_self_kx = (kx == 0) or (p2 % 2 == 0 and kx == p2 // 2)
        if not on_self_kx:
            return False
        s = ky_signed(ky)
        return (s == 0) or (p1 % 2 == 0 and abs(s) == p1 // 2)

    # Build candidate list on rfft half-plane, skip DC and self-conjugate singletons
    cand = []
    for ky in range(p1):
        s = ky_signed(ky)
        for kx in range(p2 // 2 + 1):
            if ky == 0 and kx == 0:
                continue  # DC
            if is_self_conj(ky, kx):
                continue  # exclude singletons
            r2 = (s ** 2) + (kx ** 2)
            cand.append((r2, ky, kx))
    cand.sort(key=lambda t: (t[0], abs(ky_signed(t[1])), t[2]))

    # Select without conjugate collisions
    chosen = []
    seen_axis_pairs = set()  # for kx in {0, mid}, prevent picking both ky and -ky

    mid_kx = p2 // 2 if (p2 % 2 == 0) else None
    for _, ky, kx in cand:
        if len(chosen) >= n_freqs:
            break
        if (kx == 0) or (mid_kx is not None and kx == mid_kx):
            s = ky_signed(ky)
            key = (kx, min(s, -s))  # canonicalize ±ky
            if key in seen_axis_pairs:
                continue
            seen_axis_pairs.add(key)
            chosen.append((ky, kx))
        else:
            # 0 < kx < mid_kx (or no mid): conjugate lives outside half-plane → always safe
            chosen.append((ky, kx))

    if len(chosen) < n_freqs:
        raise ValueError(f"Could only find {len(chosen)} unique non-conjugate bins; "
                         f"requested {n_freqs}. Increase grid size or reduce n_freqs.")

    # Amplitudes + random phases, then place each bin + its conjugate
    amps = np.sqrt(np.linspace(amp_max, amp_min, n_freqs, dtype=float))
    phases = rng.uniform(0.0, 2*np.pi, size=n_freqs)

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


def generate_fixed_template_2d(p1: int, p2: int) -> np.ndarray:
    """
    Generate 2D template array from Fourier spectrum.
    
    Args:
        p1: height dimension
        p2: width dimension
    
    Returns:
        template: (p1, p2) real-valued array
    """
    # Generate template array from 2D Fourier spectrum
    spectrum = np.zeros((p1, p2), dtype=complex)
    
    assert p1 > 5 and p2 > 5, "p1 and p2 must be greater than 5"
    
    # Set 2D frequency components with specific amplitudes
    # Format: spectrum[kx, ky] where kx is "vertical freq", ky is "horizontal freq"
    
    # Axis-aligned frequencies
    spectrum[1, 0] = 10.0      # vertical frequency 1
    spectrum[-1, 0] = 10.0     # conjugate
    # spectrum[0, 1] = 10.0      # horizontal frequency 1  
    # spectrum[0, -1] = 10.0     # conjugate
    
    # Higher frequency components
    # spectrum[3, 0] = 7.5
    # spectrum[-3, 0] = 7.5
    spectrum[0, 3] = 7.5
    spectrum[0, -3] = 7.5

    # Diagonal/mixed frequencies
    spectrum[2, 1] = 5.0
    spectrum[-2, -1] = 5.0    # conjugate
    # spectrum[1, 2] = 5.0
    # spectrum[-1, -2] = 5.0    # conjugate
    
    # Generate signal from spectrum
    template = np.fft.ifft2(spectrum).real
    
    return template



# Spherically Symmetric Templates

def _fft_indices(n):
    """
    Return integer-like frequency indices aligned with numpy's FFT layout.
    Example: n=8 -> [0,1,2,3,4,-3,-2,-1]
    """
    k = np.fft.fftfreq(n) * n
    return k.astype(int)

def generate_hexagon_tie_template_2d(p1: int, p2: int, k0: float = 6.0, amp: float = 1.0):
    """
    Real template whose 2D Fourier spectrum has equal maxima at six directions
    (0°, 60°, 120°, 180°, 240°, 300°) with radius ~ k0 (in FFT index units).
    
    Args:
        p1, p2: spatial dims (height, width). Require > 5 recommended.
        k0: desired radius (index units). Not necessarily integer; we round.
        amp: amplitude per spike (before conjugate pairing)
        
    Returns:
        template: (p1, p2) real-valued array
    """
    assert p1 > 5 and p2 > 5, "p1 and p2 must be > 5"
    spec = np.zeros((p1, p2), dtype=np.complex128)

    # Six target angles for a hexagon
    thetas = np.arange(6) * (np.pi / 3.0)

    # FFT index grids
    Kx = _fft_indices(p1)
    Ky = _fft_indices(p2)

    # Map from (kx,ky) in index space to array indices (wrapped)
    def put(kx, ky, val):
        spec[int(kx) % p1, int(ky) % p2] += val

    used = set()
    for th in thetas:
        # Target continuous coordinates at radius k0
        kx_f = k0 * np.cos(th)
        ky_f = k0 * np.sin(th)
        # Round to nearest integer grid point
        kx = int(np.round(kx_f))
        ky = int(np.round(ky_f))
        # Avoid (0,0) and duplicates
        if (kx, ky) == (0, 0):
            # nudge radius by 1 if rounding hit DC
            if abs(np.cos(th)) > abs(np.sin(th)):
                kx = 1 if kx >= 0 else -1
            else:
                ky = 1 if ky >= 0 else -1
        if (kx, ky) in used:
            continue
        used.add((kx, ky))
        used.add((-kx, -ky))

        # Place equal-amplitude spikes with Hermitian symmetry
        put(kx, ky, amp)                     # +k
        put(-kx, -ky, np.conjugate(amp))     # -k (conjugate)

    # Remove DC (optional) to avoid mean offset
    spec[0, 0] = 0.0

    # Real template
    x = np.fft.ifft2(spec).real
    return x
    
def generate_ring_isotropic_template_2d(p1: int, p2: int, r0: float = 6.0, sigma: float = 0.5, total_power: float = 1.0):
    """
    Real template with a narrow, isotropic ring in the 2D spectrum: |X(k)| ≈ exp(- (||k||-r0)^2 / (2 sigma^2)).
    This produces a spherical (circular) symmetry -> orientation tie across the ring.

    Args:
        p1, p2: spatial dims
        r0: target radius (index units)
        sigma: radial width of the ring
        total_power: scales overall energy (roughly)

    Returns:
        template: (p1, p2) real-valued array
    """
    assert p1 > 5 and p2 > 5, "p1 and p2 must be > 5"

    # Build index grids in FFT layout
    kx = _fft_indices(p1)[:, None]  # (p1,1)
    ky = _fft_indices(p2)[None, :]  # (1,p2)
    R = np.sqrt(kx**2 + ky**2)

    # Radial Gaussian ring (real, even -> already Hermitian when phases are 0)
    mag = np.exp(-0.5 * ((R - r0) / max(sigma, 1e-6))**2)

    # Optional: zero DC
    mag[0, 0] = 0.0

    # Normalize to desired total power (approximate; ifft2 has 1/(p1*p2) factor)
    power = np.sum(mag**2)
    if power > 0:
        mag *= np.sqrt(total_power / power)

    # Real, symmetric spectrum (phase = 0 everywhere)
    spec = mag.astype(np.complex128)

    x = np.fft.ifft2(spec).real
    return x

def generate_gaussian_template_2d(
    p1: int,
    p2: int,
    center: tuple[float, float] = None,
    sigma: float = 2.0,
    k_freqs: int = None,
) -> np.ndarray:
    """
    Generate 2D template with a single Gaussian, optionally band-limited to top-k frequencies.
    Args:
        p1: height dimension
        p2: width dimension
        center: (cx, cy) center position, defaults to center of grid
        sigma: standard deviation of Gaussian
        k_freqs: if not None, keep only the top k frequencies by power (band-limit)
    Returns:
        template: (p1, p2) real-valued array
    """
    if center is None:
        center = (p1 / 2, p2 / 2)
    cx, cy = center
    # Create coordinate grids
    x = np.arange(p1)
    y = np.arange(p2)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # Compute Gaussian
    template = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
    # If k_freqs specified, band-limit to top-k frequencies
    if k_freqs is not None:
        # Take DFT
        spectrum = np.fft.fft2(template)
        # Compute power for each frequency
        power = np.abs(spectrum) ** 2
        power_flat = power.flatten()
        # Get indices of all frequencies
        kx_indices = np.arange(p1)
        ky_indices = np.arange(p2)
        KX, KY = np.meshgrid(kx_indices, ky_indices, indexing="ij")
        all_indices = list(zip(KX.flatten(), KY.flatten()))
        # Sort by power and select top-k
        sorted_idx = np.argsort(-power_flat)
        top_k_idx = sorted_idx[:k_freqs]
        top_k_freqs = set([all_indices[i] for i in top_k_idx])
        # Create mask: 1 for top-k frequencies, 0 for others
        mask = np.zeros((p1, p2), dtype=complex)
        for kx, ky in top_k_freqs:
            mask[kx, ky] = 1.0
        # Apply mask and take IDFT
        spectrum_masked = spectrum * mask
        template = np.fft.ifft2(spectrum_masked).real
    return template
