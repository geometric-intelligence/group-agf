import numpy as np
import torch
import gagf.group_learning.group_fourier_transform as gft


class ZnZPower2D:
    """Compute and store the power spectrum of the template, which can be used
    to compute theoretical loss plateau predictions for the ZnZ group and compare to learned power spectrum.

    Parameters
    ----------
    template : ndarray (p*p)
        Flattened 2D template array.
    """

    def __init__(self, template):
        self.template = template
        self.p = int(np.sqrt(len(template)))
        self.template_2D = template.reshape((self.p, self.p))
        self.x_freqs, self.y_freqs, self.power = self.cnxcn_power_spectrum()

    def cnxcn_power_spectrum(self, no_freq=False):
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

        if no_freq:
            return power

        # Frequency bins
        row_freqs = np.fft.fftfreq(M)  # full symmetric frequencies (rows)
        column_freqs = np.fft.rfftfreq(N)  # only non-negative frequencies (columns)

        return row_freqs, column_freqs, power

    def loss_plateau_predictions(self):
        """Compute theoretical loss plateau predictions from the template's power spectrum.
        (as predicted by AGF)

        Returns
        -------
        plateau_predictions : list of float
            Theoretical loss plateau predictions for each nonzero power, in descending order.
        """
        p = int(np.sqrt(len(self.template)))
        print("Computing loss plateau predictions for template of shape:", (p, p))
        _, _, power = self.cnxcn_power_spectrum()
        power = power.flatten()

        nonzero_power_mask = power > 1e-20
        power = power[nonzero_power_mask]
        i_power_descending_order = np.argsort(power)[
            ::-1
        ] 
        power = power[i_power_descending_order]

        plateau_predictions = [np.sum(power[k:]) for k in range(len(power))]
        coef = 1 / (p * p)
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
        self.p = len(template)
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
        irreps = self.group.irreps()

        power_spectrum = np.zeros(len(irreps))
        for i, irrep in enumerate(irreps):
            fourier_coef = gft.compute_group_fourier_coef(
                self.group, self.template, irrep
            )
            power_spectrum[i] = irrep.size * np.trace(
                fourier_coef.conj().T @ fourier_coef
            )  
        power_spectrum = (
            power_spectrum / self.group.order()
        ) 

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
        Group type (e.g., 'znz_znz').
    model : TwoLayerNet
        The trained model.
    param_history : list of dict
        List of model parameters at each training step.
    model_inputs : torch.Tensor
        Input data tensor.
    group : Group (escnn object)
        The escnn group object. Optional, since we don't use escnn for znz_znz.

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

    if group_name == "znz_znz":  # 2D template
        p1 = int(np.sqrt(output_shape[0]))
        p2 = p1
        template_power_length = p1 * (p2 // 2 + 1)
        reshape_dims = (-1, p1, p2)
    else:  # other groups are 1D signals
        template_power_length = len(group.irreps())
        p1 = output_shape[0]
        reshape_dims = (-1, p1)

    # Compute output power over time (GD)
    # TODO: the number of points and the number of inputs to compute the power over time should depend on the group.
    # For example, for Octahedral and A5 groups, we should compute the power over time for a smaller number of points.
    # Because the dataset is very large for these groups, so we don't need to compute the power over time for all steps.
    num_points = 200
    num_inputs_to_compute_power = len(model_inputs) // 50
    X_tensor = model_inputs[
        :num_inputs_to_compute_power
    ]  # Added by Nina to speed up computation with octahedral.
    steps = np.unique(
        np.logspace(1, np.log10(len(param_history) - 1), num_points, dtype=int)
    )
    steps = steps[steps > 50]
    steps = np.hstack([np.linspace(1, 50, 5).astype(int), steps])
    print(f"Computing power over time for {num_inputs_to_compute_power} inputs and {len(steps)} steps: {steps} ")
    powers_over_time = np.zeros([len(steps), template_power_length])

    for i_step, step in enumerate(steps):
        model.load_state_dict(param_history[step])

        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            print("outputs dtype", outputs.dtype)
            outputs_arr = outputs.detach().cpu().numpy().reshape(reshape_dims)

            print(
                "Computing power at step", step, "with output shape", outputs_arr.shape
            )

            powers = []
            for out in outputs_arr:
                if group_name == "znz_znz":
                    output_power = ZnZPower2D(out.flatten())
                else:
                    output_power = GroupPower(out.flatten(), group=group)

                this_power = output_power.power
                # flatten to 1D for both 1D and 2D cases
                this_power_flat = this_power.flatten()
                powers.append(this_power_flat)
            powers = np.array(powers)

            average_power = np.mean(powers, axis=0) # shape: (num_samples, template_power_length)
            powers_over_time[i_step, :] = average_power

    powers_over_time = np.array(powers_over_time)  # shape: (steps, num_freqs)
    powers_over_time[powers_over_time < 1e-20] = 0

    return powers_over_time, steps
