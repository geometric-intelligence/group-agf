import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from src.datamodule import (
    generate_fourier_template_1d,
    generate_gaussian_template_1d,
    generate_onehot_template_1d,
    generate_template_unique_freqs,
    mnist_template_1d,
    mnist_template_2d,
)
from src.model import QuadraticRNN, SequentialMLP
from src.optimizers import HybridRNNOptimizer, PerNeuronScaledSGD
from src.utils import (
    plot_2d_signal,
    plot_model_predictions_over_time,
    plot_model_predictions_over_time_1d,
    plot_prediction_power_spectrum_over_time_1d,
    plot_training_loss_with_theory,
    plot_wmix_frequency_structure,
    topk_template_freqs,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_run_directory(base_dir: str = "runs") -> Path:
    """Create timestamped run directory."""
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    return Path(run_dir)


def save_results(
    run_dir: Path,
    config: dict,
    model,
    train_loss_hist,
    val_loss_hist,
    param_hist,
    template: np.ndarray,
    training_time: float,
    device: str,
) -> dict:
    """Save all experiment results."""
    print(f"Saving results to {run_dir}...")

    # Ensure checkpoints directory exists
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save template
    np.save(run_dir / "template.npy", template)

    # Save training history
    np.save(run_dir / "train_loss_history.npy", np.array(train_loss_hist))
    np.save(run_dir / "val_loss_history.npy", np.array(val_loss_hist))
    torch.save(param_hist, run_dir / "param_history.pt")

    # Save final model
    torch.save(model.state_dict(), checkpoints_dir / "final_model.pt")

    # Save metadata
    metadata = {
        "final_train_loss": float(train_loss_hist[-1]),
        "final_val_loss": float(val_loss_hist[-1]),
        "training_time_seconds": training_time,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": device,
        "description": config.get("description", ""),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("  ✓ All results saved")
    return metadata


def produce_plots_2d(
    run_dir: Path,
    config: dict,
    model,
    param_hist,
    param_save_indices,
    train_loss_hist,
    template_2d: np.ndarray,
    training_mode: str,
    device: str,
):
    """
    Generate all analysis plots after training (2D only).

    Note: This function currently only supports 2D templates with p1 and p2 dimensions.
    For 1D templates, basic plots are generated separately in train_single_run.

    Some plots are model-specific:
    - W_mix frequency structure: QuadraticRNN only (skipped for SequentialMLP)
    - W_out neuron specialization: All models
    - Power spectrum, predictions, loss curves: All models

    Args:
        run_dir: Directory to save plots
        config: Configuration dictionary (must have dimension=2)
        model: Trained model (QuadraticRNN or SequentialMLP)
        param_hist: List of parameter snapshots
        param_save_indices: Indices where params were saved
        train_loss_hist: Training loss history
        template_2d: 2D template array (p1, p2)
        training_mode: 'online' or 'offline'
        device: Device string ('cpu' or 'cuda')
    """
    print("\n=== Generating Analysis Plots ===")

    ### ----- COMPUTE X-AXIS VALUES ----- ###
    dimension = config["data"]["dimension"]
    if dimension == 1:
        p_flat = config["data"]["p"]
    else:
        p_flat = config["data"]["p1"] * config["data"]["p2"]

    k = config["data"]["k"]
    batch_size = config["data"]["batch_size"]
    total_space_size = p_flat**k

    # Calculate different x-axis values for plotting
    if training_mode == "online":
        steps = np.arange(len(train_loss_hist))
        samples_seen = batch_size * steps
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Step"
    else:  # offline
        epochs = np.arange(len(train_loss_hist))
        samples_seen = config["data"]["num_samples"] * epochs
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Epoch"

    # Save x-axis data
    np.save(run_dir / "samples_seen.npy", samples_seen)
    np.save(run_dir / "fraction_of_space_seen.npy", fraction_of_space)

    print(f"Total data space: {total_space_size:,} sequences")
    print(f"Samples seen: {samples_seen[-1]:,} ({fraction_of_space[-1] * 100:.4f}% of space)")

    ### ----- GENERATE EVALUATION DATA ----- ###
    print("Generating evaluation data for visualization...")
    from src.datamodule import build_modular_addition_sequence_dataset_2d

    X_seq_2d, Y_seq_2d, _ = build_modular_addition_sequence_dataset_2d(
        config["data"]["p1"],
        config["data"]["p2"],
        template_2d,
        config["data"]["k"],
        mode="sampled",
        num_samples=min(config["data"]["num_samples"], 1000),
        return_all_outputs=config["model"]["return_all_outputs"],
    )
    X_seq_2d_t = torch.tensor(X_seq_2d, dtype=torch.float32, device=device)
    Y_seq_2d_t = torch.tensor(Y_seq_2d, dtype=torch.float32, device=device)
    print(f"  Generated {X_seq_2d_t.shape[0]} samples for visualization")

    ### ----- COMPUTE CHECKPOINT INDICES ----- ###
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config["analysis"]["checkpoints"]
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]

    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")
    print(
        f"  Corresponding to step/epoch indices: {[param_save_indices[i] for i in checkpoint_indices]}"
    )

    ### ----- PLOT TRAINING LOSS ----- ###
    print("\nPlotting training loss...")

    # Plot 1: Loss vs Steps/Epochs
    plot_training_loss_with_theory(
        loss_history=train_loss_hist,
        template_2d=template_2d,
        p1=config["data"]["p1"],
        p2=config["data"]["p2"],
        x_values=None,
        x_label=x_label_steps,
        save_path=os.path.join(run_dir, "training_loss_vs_steps.pdf"),
        show=False,
    )

    # Plot 2: Loss vs Samples Seen
    plot_training_loss_with_theory(
        loss_history=train_loss_hist,
        template_2d=template_2d,
        p1=config["data"]["p1"],
        p2=config["data"]["p2"],
        x_values=samples_seen,
        x_label="Samples Seen",
        save_path=os.path.join(run_dir, "training_loss_vs_samples.pdf"),
        show=False,
    )

    # Plot 3: Loss vs Fraction of Space
    plot_training_loss_with_theory(
        loss_history=train_loss_hist,
        template_2d=template_2d,
        p1=config["data"]["p1"],
        p2=config["data"]["p2"],
        x_values=fraction_of_space,
        x_label="Samples Seen / Data Space Size",
        save_path=os.path.join(run_dir, "training_loss_vs_fraction.pdf"),
        show=False,
    )

    ### ----- PLOT MODEL PREDICTIONS ----- ###
    print("Plotting model predictions over time...")
    plot_model_predictions_over_time(
        model,
        param_hist,
        X_seq_2d_t,
        Y_seq_2d_t,
        config["data"]["p1"],
        config["data"]["p2"],
        steps=checkpoint_indices,
        save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
        show=False,
    )

    # ### ----- PLOT POWER SPECTRUM ANALYSIS ----- ###
    # print("Analyzing power spectrum of predictions over training...")
    # plot_prediction_power_spectrum_over_time(
    #     model,
    #     param_hist,
    #     X_seq_2d_t,
    #     Y_seq_2d_t,
    #     template_2d,
    #     config['data']['p1'],
    #     config['data']['p2'],
    #     loss_history=train_loss_hist,
    #     param_save_indices=param_save_indices,
    #     num_freqs_to_track=10,
    #     checkpoint_indices=checkpoint_indices,
    #     num_samples=100,
    #     save_path=os.path.join(run_dir, "power_spectrum_analysis.pdf"),
    #     show=False
    # )

    ### ----- PLOT FOURIER MODES REFERENCE ----- ###
    print("Creating Fourier modes reference...")
    tracked_freqs = topk_template_freqs(template_2d, K=10)
    colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_freqs)))

    # plot_fourier_modes_reference(
    #     tracked_freqs,
    #     colors,
    #     config['data']['p1'],
    #     config['data']['p2'],
    #     save_path=os.path.join(run_dir, "fourier_modes_reference.pdf"),
    #     save_individual=True,
    #     individual_dir=os.path.join(run_dir, "fourier_modes"),
    #     show=False
    # )

    # ### ----- PLOT W_OUT NEURON SPECIALIZATION ----- ###
    # print("Visualizing W_out neuron specialization...")
    # plot_wout_neuron_specialization(
    #     param_hist,
    #     tracked_freqs,
    #     colors,
    #     config['data']['p1'],
    #     config['data']['p2'],
    #     steps=checkpoint_indices,
    #     dead_thresh_l2=0.25,
    #     save_dir=run_dir,
    #     show=False
    # )

    ### ----- PLOT W_MIX FREQUENCY STRUCTURE (QuadraticRNN only) ----- ###
    model_type = config["model"]["model_type"]
    if model_type == "QuadraticRNN":
        print("Visualizing W_mix frequency structure...")
        plot_wmix_frequency_structure(
            param_hist,
            tracked_freqs,
            colors,
            config["data"]["p1"],
            config["data"]["p2"],
            steps=checkpoint_indices,
            within_group_order="phase",
            dead_l2_thresh=0.1,
            save_path=os.path.join(run_dir, "wmix_frequency_structure.pdf"),
            show=False,
        )
    else:
        print("Skipping W_mix frequency structure plot (not applicable for SequentialMLP)")

    print("\n✓ All plots generated successfully!")


def produce_plots_1d(
    run_dir: Path,
    config: dict,
    model,
    param_hist,
    param_save_indices,
    train_loss_hist,
    template_1d: np.ndarray,
    training_mode: str,
    device: str,
):
    """
    Generate all analysis plots after training (1D version).

    Args:
        run_dir: Directory to save plots
        config: Configuration dictionary (must have dimension=1)
        model: Trained model (QuadraticRNN or SequentialMLP)
        param_hist: List of parameter snapshots
        param_save_indices: Indices where params were saved
        train_loss_hist: Training loss history
        template_1d: 1D template array (p,)
        training_mode: 'online' or 'offline'
        device: Device string ('cpu' or 'cuda')
    """
    print("\n=== Generating Analysis Plots (1D) ===")

    ### ----- COMPUTE X-AXIS VALUES ----- ###
    p = config["data"]["p"]
    k = config["data"]["k"]
    batch_size = config["data"]["batch_size"]
    total_space_size = p**k

    # Calculate different x-axis values for plotting
    if training_mode == "online":
        steps = np.arange(len(train_loss_hist))
        samples_seen = batch_size * steps
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Step"
    else:  # offline
        epochs = np.arange(len(train_loss_hist))
        samples_seen = config["data"]["num_samples"] * epochs
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Epoch"

    # Save x-axis data
    np.save(run_dir / "samples_seen.npy", samples_seen)
    np.save(run_dir / "fraction_of_space_seen.npy", fraction_of_space)

    print(f"Total data space: {total_space_size:,} sequences")
    print(f"Samples seen: {samples_seen[-1]:,} ({fraction_of_space[-1] * 100:.4f}% of space)")

    ### ----- GENERATE EVALUATION DATA ----- ###
    print("Generating evaluation data for visualization...")
    from src.datamodule import build_modular_addition_sequence_dataset_1d

    X_seq_1d, Y_seq_1d, _ = build_modular_addition_sequence_dataset_1d(
        config["data"]["p"],
        template_1d,
        config["data"]["k"],
        mode="sampled",
        num_samples=min(config["data"]["num_samples"], 1000),
        return_all_outputs=config["model"]["return_all_outputs"],
    )
    X_seq_1d_t = torch.tensor(X_seq_1d, dtype=torch.float32, device=device)
    Y_seq_1d_t = torch.tensor(Y_seq_1d, dtype=torch.float32, device=device)
    print(f"  Generated {X_seq_1d_t.shape[0]} samples for visualization")

    ### ----- COMPUTE CHECKPOINT INDICES ----- ###
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config["analysis"]["checkpoints"]
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]

    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")
    print(
        f"  Corresponding to step/epoch indices: {[param_save_indices[i] for i in checkpoint_indices]}"
    )

    ### ----- PLOT TRAINING LOSS ----- ###
    print("\nPlotting training loss...")

    # Create a 2x2 subplot for different scale combinations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x_values = steps if training_mode == "online" else epochs

    scale_configs = [
        ("linear", "linear", "Linear Scale"),
        ("linear", "log", "Log Y"),
        ("log", "linear", "Log X"),
        ("log", "log", "Log-Log"),
    ]

    for ax, (xscale, yscale, title) in zip(axes.flat, scale_configs):
        ax.plot(x_values, train_loss_hist, lw=2, color="#1f77b4")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(x_label_steps)
        ax.set_ylabel("Training Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_loss.pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓ Saved training loss plot (all scales)")

    ### ----- PLOT MODEL PREDICTIONS ----- ###
    print("Plotting model predictions over time...")
    plot_model_predictions_over_time_1d(
        model,
        param_hist,
        X_seq_1d_t,
        Y_seq_1d_t,
        p,
        steps=checkpoint_indices,
        save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
        show=False,
    )

    ### ----- PLOT POWER SPECTRUM ANALYSIS ----- ###
    print("Analyzing power spectrum of predictions over training...")
    plot_prediction_power_spectrum_over_time_1d(
        model,
        param_hist,
        X_seq_1d_t,
        Y_seq_1d_t,
        template_1d,
        p,
        loss_history=train_loss_hist,
        param_save_indices=param_save_indices,
        num_freqs_to_track=min(10, p // 4),
        checkpoint_indices=checkpoint_indices,
        num_samples=100,
        save_path=os.path.join(run_dir, "power_spectrum_analysis.pdf"),
        show=False,
    )

    # ### ----- PLOT W_OUT NEURON SPECIALIZATION ----- ###
    # print("Visualizing W_out neuron specialization...")
    # tracked_freqs = topk_template_freqs_1d(template_1d, K=min(10, p // 4))
    # colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_freqs)))

    # plot_wout_neuron_specialization_1d(
    #     param_hist,
    #     tracked_freqs,
    #     colors,
    #     p,
    #     steps=checkpoint_indices,
    #     dead_thresh_l2=0.25,
    #     save_dir=run_dir,
    #     show=False
    # )

    print("\n✓ All 1D plots generated successfully!")


def plot_model_predictions_over_time_D3(
    model,
    param_hist,
    X_eval,
    Y_eval,
    group_order: int,
    checkpoint_indices: list,
    save_path: str = None,
    num_samples: int = 5,
):
    """
    Plot model predictions vs targets at different training checkpoints for D3.

    Args:
        model: Trained model
        param_hist: List of parameter snapshots
        X_eval: Input evaluation tensor (N, k, group_order)
        Y_eval: Target evaluation tensor (N, group_order)
        group_order: Order of D3 group (6)
        checkpoint_indices: Indices into param_hist to visualize
        save_path: Path to save the plot
        num_samples: Number of samples to show
    """
    n_checkpoints = len(checkpoint_indices)

    fig, axes = plt.subplots(
        num_samples, n_checkpoints, figsize=(4 * n_checkpoints, 3 * num_samples)
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if n_checkpoints == 1:
        axes = axes.reshape(-1, 1)

    # Select random sample indices
    sample_indices = np.random.choice(
        len(X_eval), size=min(num_samples, len(X_eval)), replace=False
    )

    for col, ckpt_idx in enumerate(checkpoint_indices):
        # Load parameters for this checkpoint
        model.load_state_dict(param_hist[ckpt_idx])
        model.eval()

        with torch.no_grad():
            outputs = model(X_eval[sample_indices])
            outputs_np = outputs.cpu().numpy()
            targets_np = Y_eval[sample_indices].cpu().numpy()

        for row, (output, target) in enumerate(zip(outputs_np, targets_np)):
            ax = axes[row, col]
            x_axis = np.arange(group_order)

            ax.bar(x_axis - 0.15, target, width=0.3, label="Target", alpha=0.7, color="#2ecc71")
            ax.bar(x_axis + 0.15, output, width=0.3, label="Output", alpha=0.7, color="#e74c3c")

            if row == 0:
                ax.set_title(f"Checkpoint {ckpt_idx}")
            if col == 0:
                ax.set_ylabel(f"Sample {sample_indices[row]}")
            if row == num_samples - 1:
                ax.set_xlabel("Group element")
            if row == 0 and col == n_checkpoints - 1:
                ax.legend(loc="upper right", fontsize=8)

            ax.set_xticks(x_axis)
            ax.grid(True, alpha=0.3)

    plt.suptitle("D3 Model Predictions vs Targets Over Training", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_power_spectrum_over_time_D3(
    model,
    param_hist,
    param_save_indices,
    X_eval,
    template: np.ndarray,
    D3,
    k: int,
    optimizer: str,
    init_scale: float,
    save_path: str = None,
    num_samples_for_power: int = 100,
    num_checkpoints_to_sample: int = 50,
):
    """
    Plot power spectrum of model outputs vs template power spectrum over training for D3.

    Args:
        model: Trained model
        param_hist: List of parameter snapshots
        param_save_indices: List mapping param_hist index to epoch number
        X_eval: Input evaluation tensor
        template: Template array (group_order,)
        D3: DihedralGroup object from escnn
        k: Sequence length
        optimizer: Optimizer name (e.g., 'per_neuron', 'adam')
        init_scale: Initialization scale
        save_path: Path to save the plot
        num_samples_for_power: Number of samples to average power over
        num_checkpoints_to_sample: Number of checkpoints to sample for the evolution plot
    """
    from group_agf.binary_action_learning.group_fourier_transform import compute_group_fourier_coef

    group_order = D3.order()
    irreps = D3.irreps()
    n_irreps = len(irreps)

    # Compute template power spectrum
    template_power = np.zeros(n_irreps)
    for i, irrep in enumerate(irreps):
        fourier_coef = compute_group_fourier_coef(D3, template, irrep)
        template_power[i] = irrep.size * np.trace(fourier_coef.conj().T @ fourier_coef)
    template_power = template_power / group_order

    print(f"  Template power spectrum: {template_power}")
    print("  (These are dim^2 * diag_value^2 / |G| for each irrep)")

    # Sample checkpoints uniformly for evolution plot
    total_checkpoints = len(param_hist)
    if total_checkpoints <= num_checkpoints_to_sample:
        sampled_ckpt_indices = list(range(total_checkpoints))
    else:
        sampled_ckpt_indices = np.linspace(
            0, total_checkpoints - 1, num_checkpoints_to_sample, dtype=int
        ).tolist()

    # Get corresponding epoch numbers
    epoch_numbers = [param_save_indices[i] for i in sampled_ckpt_indices]

    # Compute model output power at each sampled checkpoint
    n_sampled = len(sampled_ckpt_indices)
    model_powers = np.zeros((n_sampled, n_irreps))

    X_subset = X_eval[:num_samples_for_power]

    for i, ckpt_idx in enumerate(sampled_ckpt_indices):
        model.load_state_dict(param_hist[ckpt_idx])
        model.eval()

        with torch.no_grad():
            outputs = model(X_subset)
            outputs_np = outputs.cpu().numpy()

        # Average power over all samples
        powers = np.zeros((len(outputs_np), n_irreps))
        for sample_i, output in enumerate(outputs_np):
            for irrep_i, irrep in enumerate(irreps):
                fourier_coef = compute_group_fourier_coef(D3, output, irrep)
                powers[sample_i, irrep_i] = irrep.size * np.trace(
                    fourier_coef.conj().T @ fourier_coef
                )
        powers = powers / group_order
        model_powers[i] = np.mean(powers, axis=0)

    # Create 3 subplots: linear, log-x, log-log
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    top_k = min(5, n_irreps)
    top_irrep_indices = np.argsort(template_power)[::-1][:top_k]

    colors_line = plt.cm.tab10(np.linspace(0, 1, top_k))

    # Filter out zero epochs for log scales
    valid_mask = np.array(epoch_numbers) > 0
    valid_epochs = np.array(epoch_numbers)[valid_mask]
    valid_model_powers = model_powers[valid_mask, :]

    # Plot 1: Linear scales
    ax = axes[0]
    for i, irrep_idx in enumerate(top_irrep_indices):
        power_values = model_powers[:, irrep_idx]
        ax.plot(
            epoch_numbers,
            power_values,
            "-",
            lw=2,
            color=colors_line[i],
            label=f"Irrep {irrep_idx} (dim={irreps[irrep_idx].size})",
        )
        ax.axhline(template_power[irrep_idx], linestyle="--", alpha=0.5, color=colors_line[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Power")
    ax.set_title("Linear Scales", fontsize=12)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 2: Log x-axis only
    ax = axes[1]
    for i, irrep_idx in enumerate(top_irrep_indices):
        power_values = valid_model_powers[:, irrep_idx]
        ax.plot(
            valid_epochs,
            power_values,
            "-",
            lw=2,
            color=colors_line[i],
            label=f"Irrep {irrep_idx} (dim={irreps[irrep_idx].size})",
        )
        ax.axhline(template_power[irrep_idx], linestyle="--", alpha=0.5, color=colors_line[i])
    ax.set_xscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Power")
    ax.set_title("Log X-axis", fontsize=12)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 3: Log-log scales
    ax = axes[2]
    for i, irrep_idx in enumerate(top_irrep_indices):
        power_values = valid_model_powers[:, irrep_idx]
        # Filter out zero powers for log scale
        power_mask = power_values > 0
        if np.any(power_mask):
            ax.plot(
                valid_epochs[power_mask],
                power_values[power_mask],
                "-",
                lw=2,
                color=colors_line[i],
                label=f"Irrep {irrep_idx} (dim={irreps[irrep_idx].size})",
            )
        if template_power[irrep_idx] > 0:
            ax.axhline(template_power[irrep_idx], linestyle="--", alpha=0.5, color=colors_line[i])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Power (log scale)")
    ax.set_title("Log-Log Scales", fontsize=12)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(
        f"D3 Power Evolution Over Training (k={k}, {optimizer}, init={init_scale:.0e})",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def produce_plots_D3(
    run_dir: Path,
    config: dict,
    model,
    param_hist,
    param_save_indices,
    train_loss_hist,
    template_D3: np.ndarray,
    device: str = "cpu",
):
    """
    Generate all analysis plots after training (D3 version).

    Args:
        run_dir: Directory to save plots
        config: Configuration dictionary (must have dimension='D3')
        model: Trained model (QuadraticRNN or SequentialMLP)
        param_hist: List of parameter snapshots
        param_save_indices: Indices where params were saved
        train_loss_hist: Training loss history
        template_D3: 1D template array of shape (group_order,) where group_order=6 for D3
        device: Device string ('cpu' or 'cuda')
    """
    print("\n=== Generating Analysis Plots (D3) ===")

    from escnn.group import DihedralGroup

    D3 = DihedralGroup(N=3)
    group_order = D3.order()  # = 6

    k = config["data"]["k"]
    batch_size = config["data"]["batch_size"]
    training_mode = config["training"]["mode"]

    # Total data space size for D3 with k compositions
    total_space_size = group_order**k

    # Calculate x-axis values
    if training_mode == "online":
        steps = np.arange(len(train_loss_hist))
        samples_seen = batch_size * steps
        fraction_of_space = samples_seen / total_space_size
        x_label = "Step"
        x_values = steps
    else:  # offline
        epochs = np.arange(len(train_loss_hist))
        samples_seen = config["data"]["num_samples"] * epochs
        fraction_of_space = samples_seen / total_space_size
        x_label = "Epoch"
        x_values = epochs

    # Save x-axis data
    samples_seen_path = run_dir / "samples_seen.npy"
    fraction_path = run_dir / "fraction_of_space_seen.npy"
    np.save(samples_seen_path, samples_seen)
    np.save(fraction_path, fraction_of_space)
    print(f"  ✓ Saved {samples_seen_path}")
    print(f"  ✓ Saved {fraction_path}")

    print(f"\nD3 group order: {group_order}")
    print(f"Sequence length k: {k}")
    print(f"Total data space: {total_space_size:,} sequences")
    if len(samples_seen) > 0:
        print(f"Samples seen: {samples_seen[-1]:,} ({fraction_of_space[-1] * 100:.4f}% of space)")

    ### ----- GENERATE EVALUATION DATA ----- ###
    print("\nGenerating evaluation data for visualization...")
    from src.datamodule import build_modular_addition_sequence_dataset_D3

    X_eval, Y_eval, _ = build_modular_addition_sequence_dataset_D3(
        template_D3,
        k,
        mode="sampled",
        num_samples=min(config["data"]["num_samples"], 1000),
        return_all_outputs=config["model"]["return_all_outputs"],
    )
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
    Y_eval_t = torch.tensor(Y_eval, dtype=torch.float32, device=device)
    print(f"  Generated {X_eval_t.shape[0]} samples for visualization")

    ### ----- COMPUTE CHECKPOINT INDICES ----- ###
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config["analysis"]["checkpoints"]
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]
    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")

    ### ----- PLOT TRAINING LOSS ----- ###
    print("\nPlotting training loss...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    scale_configs = [
        ("linear", "linear", "Linear Scale"),
        ("linear", "log", "Log Y"),
        ("log", "linear", "Log X"),
        ("log", "log", "Log-Log"),
    ]

    for ax, (xscale, yscale, title) in zip(axes.flat, scale_configs):
        ax.plot(x_values, train_loss_hist, lw=2, color="#1f77b4")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Training Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"D3 Group Composition (k={k})", fontsize=14)
    plt.tight_layout()
    training_loss_path = os.path.join(run_dir, "training_loss.pdf")
    plt.savefig(training_loss_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ Saved {training_loss_path}")

    ### ----- PLOT MODEL PREDICTIONS OVER TIME ----- ###
    print("\nPlotting model predictions over time...")
    plot_model_predictions_over_time_D3(
        model=model,
        param_hist=param_hist,
        X_eval=X_eval_t,
        Y_eval=Y_eval_t,
        group_order=group_order,
        checkpoint_indices=checkpoint_indices,
        save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
    )
    print(f"  ✓ Saved {os.path.join(run_dir, 'predictions_over_time.pdf')}")

    ### ----- PLOT POWER SPECTRUM OVER TIME ----- ###
    print("\nPlotting power spectrum over time...")
    optimizer = config["training"]["optimizer"]
    init_scale = config["model"]["init_scale"]
    plot_power_spectrum_over_time_D3(
        model=model,
        param_hist=param_hist,
        param_save_indices=param_save_indices,
        X_eval=X_eval_t,
        template=template_D3,
        D3=D3,
        k=k,
        optimizer=optimizer,
        init_scale=init_scale,
        save_path=os.path.join(run_dir, "power_spectrum_analysis.pdf"),
    )
    print(f"  ✓ Saved {os.path.join(run_dir, 'power_spectrum_analysis.pdf')}")

    print("\n✓ All D3 plots generated successfully!")


def train_single_run(config: dict, run_dir: Path = None) -> dict:
    """
    Train a model (QuadraticRNN or SequentialMLP) on modular addition for a single configuration.

    Args:
        config: Configuration dictionary. Must include 'model.model_type' to specify
                'QuadraticRNN' or 'SequentialMLP'.
        run_dir: Optional run directory. If None, will create a timestamped directory.

    Returns:
        dict: Training results including final losses and metadata.
    """
    # Setup run directory if not provided
    if run_dir is None:
        run_dir = setup_run_directory(base_dir="runs")
    print(f"Experiment directory: {run_dir}")

    # Set seed
    np.random.seed(config["data"]["seed"])
    torch.manual_seed(config["data"]["seed"])

    # Determine device
    device = config["device"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ### ----- GENERATE DATA ----- ###
    print("Generating data...")

    dimension = config["data"]["dimension"]
    template_type = config["data"]["template_type"]

    if dimension == 1:
        # 1D template generation
        p = config["data"]["p"]
        p_flat = p

        if template_type == "mnist":
            template_1d = mnist_template_1d(p, config["data"]["mnist_label"], root="data")
        elif template_type == "fourier":
            n_freqs = config["data"]["n_freqs"]
            template_1d = generate_fourier_template_1d(
                p, n_freqs=n_freqs, seed=config["data"]["seed"]
            )
        elif template_type == "gaussian":
            template_1d = generate_gaussian_template_1d(
                p, n_gaussians=3, seed=config["data"]["seed"]
            )
        elif template_type == "onehot":
            template_1d = generate_onehot_template_1d(p)
        else:
            raise ValueError(f"Unknown template_type: {template_type}")

        template_1d = template_1d - np.mean(template_1d)
        template = template_1d  # For consistency in code below

        # Visualize 1D template
        print("Visualizing template...")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(template_1d)
        ax.set_xlabel("Position")
        ax.set_ylabel("Value")
        ax.set_title("1D Template")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
        print("  ✓ Saved template")

    elif dimension == 2:
        # 2D template generation
        p1 = config["data"]["p1"]
        p2 = config["data"]["p2"]
        p_flat = p1 * p2

        if template_type == "mnist":
            template_2d = mnist_template_2d(p1, p2, config["data"]["mnist_label"], root="data")
        elif template_type == "fourier":
            n_freqs = config["data"]["n_freqs"]
            template_2d = generate_template_unique_freqs(
                p1, p2, n_freqs=n_freqs, seed=config["data"]["seed"]
            )
        else:
            raise ValueError(f"Unknown template_type for 2D: {template_type}")

        template_2d = template_2d - np.mean(template_2d)
        template = template_2d  # For consistency in code below

        # Visualize 2D template
        print("Visualizing template...")
        fig, ax = plot_2d_signal(template_2d, title="Template", cmap="gray")
        fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
        print("  ✓ Saved template")
    elif dimension == "D3":
        from escnn.group import DihedralGroup

        from group_agf.binary_action_learning.group_fourier_transform import (
            compute_group_inverse_fourier_transform,
        )

        D3 = DihedralGroup(N=3)  # D3 = dihedral group of order 6 (3 rotations * 2 for reflections)
        group_order = D3.order()  # = 6
        p_flat = group_order  # For D3, the "p" is the group order

        print(f"D3 group order: {group_order}")
        print(f"D3 irreps: {[irrep.size for irrep in D3.irreps()]} (dimensions)")

        if template_type == "onehot":
            # Generate one-hot template of length group_order
            # This creates a template with a spike at position 1
            template_d3 = np.zeros(group_order, dtype=np.float32)
            template_d3[1] = 10.0
            template_d3 = template_d3 - np.mean(template_d3)
            print("Template type: onehot")

        elif template_type == "custom_fourier":
            # Generate template from Fourier coefficients for each irrep
            # powers specifies the DESIRED POWER SPECTRUM values (not diagonal values)
            # We convert powers to Fourier coefficient diagonal values using:
            #   diag_value = sqrt(group_size * power / dim^2)
            # This is because: power = dim^2 * diag_value^2 / group_size
            powers = config["data"]["powers"]
            irreps = D3.irreps()
            irrep_dims = [ir.size for ir in irreps]

            assert len(powers) == len(irreps), (
                f"powers must have {len(irreps)} values (one per irrep), got {len(powers)}"
            )

            # Convert powers to Fourier coefficient diagonal values
            # (same formula as in binary_action_learning/main.py)
            fourier_coef_diag_values = [
                np.sqrt(group_order * p / dim**2) if p > 0 else 0.0
                for p, dim in zip(powers, irrep_dims)
            ]

            print("Template type: custom_fourier")
            print(f"Desired powers (per irrep): {powers}")
            print(f"Fourier coef diagonal values: {fourier_coef_diag_values}")

            # Build spectrum: list of diagonal matrices, one per irrep
            spectrum = []
            for i, irrep in enumerate(irreps):
                diag_val = fourier_coef_diag_values[i]
                diag_values = np.full(irrep.size, diag_val, dtype=float)
                mat = np.zeros((irrep.size, irrep.size), dtype=float)
                np.fill_diagonal(mat, diag_values)
                print(
                    f"  Irrep {i} (dim={irrep.size}): diag_value = {diag_val:.4f} -> power = {powers[i]}"
                )
                spectrum.append(mat)

            # Generate template via inverse group Fourier transform
            template_d3 = compute_group_inverse_fourier_transform(D3, spectrum)
            template_d3 = template_d3 - np.mean(template_d3)
            template_d3 = template_d3.astype(np.float32)
        else:
            raise ValueError(
                f"Unknown template_type for D3: {template_type}. Must be 'onehot' or 'custom_fourier'"
            )

        template = template_d3  # For consistency in code below
        print(f"Template shape: {template.shape}")

        # Visualize D3 template
        print("Visualizing template...")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(group_order), template_d3)
        ax.set_xlabel("Group element index")
        ax.set_ylabel("Value")
        title = f"D3 Template (order={group_order}, type={template_type})"
        if template_type == "custom_fourier":
            title += f"\npowers={powers}"
        ax.set_title(title)
        ax.set_xticks(range(group_order))
        fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print("  ✓ Saved template")
    else:
        raise ValueError(f"dimension must be 1 or 2, got {dimension}")

    ### ----- SETUP TRAINING ----- ###
    print("Setting up model and training...")

    # Flatten template for model (works for both 1D and 2D)
    template_torch = torch.tensor(template, device=device, dtype=torch.float32).flatten()

    # Determine which model to use
    model_type = config["model"]["model_type"]
    print(f"Using model type: {model_type}")

    if model_type == "QuadraticRNN":
        rnn_2d = QuadraticRNN(
            p=p_flat,
            d=config["model"]["hidden_dim"],
            template=template_torch,
            init_scale=config["model"]["init_scale"],
            return_all_outputs=config["model"]["return_all_outputs"],
            transform_type=config["model"]["transform_type"],
        ).to(device)
    elif model_type == "SequentialMLP":
        rnn_2d = SequentialMLP(
            p=p_flat,
            d=config["model"]["hidden_dim"],
            template=template_torch,
            k=config["data"]["k"],
            init_scale=config["model"]["init_scale"],
            return_all_outputs=config["model"]["return_all_outputs"],
        ).to(device)
    else:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be 'QuadraticRNN' or 'SequentialMLP'"
        )

    criterion = nn.MSELoss()

    # Optimizer selection with model-aware defaults
    optimizer_name = config["training"]["optimizer"]

    # Auto-select optimizer if not specified or if 'auto'
    if optimizer_name == "auto" or (optimizer_name not in ["adam", "hybrid", "per_neuron"]):
        if model_type == "SequentialMLP":
            optimizer_name = "per_neuron"
            print(f"Auto-selected optimizer: {optimizer_name} (recommended for SequentialMLP)")
        else:
            optimizer_name = "adam"
            print(f"Auto-selected optimizer: {optimizer_name}")
    else:
        print(f"Using optimizer: {optimizer_name}")

    if optimizer_name == "adam":
        optimizer = optim.Adam(
            rnn_2d.parameters(),
            lr=config["training"]["learning_rate"],
            betas=tuple(config["training"]["betas"]),
            weight_decay=config["training"]["weight_decay"],
        )
    elif optimizer_name == "hybrid":
        if model_type != "QuadraticRNN":
            raise ValueError(
                f"'hybrid' optimizer is only supported for QuadraticRNN, got {model_type}"
            )
        optimizer = HybridRNNOptimizer(
            rnn_2d,
            lr=1,
            scaling_factor=config["training"]["scaling_factor"],
            adam_lr=config["training"]["learning_rate"],
            adam_betas=tuple(config["training"]["betas"]),
            adam_eps=1e-8,
        )
    elif optimizer_name == "per_neuron":
        # Per-neuron scaled SGD (recommended for SequentialMLP)
        degree = config["training"]["degree"]
        lr = config["training"]["learning_rate"]

        # For SequentialMLP, use lr=1.0 by default if not specified
        if model_type == "SequentialMLP" and lr == 1.0e-3:
            print("  Note: Using lr=1.0 for per_neuron optimizer with SequentialMLP")
            lr = 1.0

        optimizer = PerNeuronScaledSGD(
            rnn_2d,
            lr=lr,
            degree=degree,  # Will auto-infer as k+1 for SequentialMLP (k = sequence length)
        )
        print(f"  Degree of homogeneity: {optimizer.param_groups[0]['degree']}")
    else:
        raise ValueError(
            f"Invalid optimizer: {optimizer_name}. Must be 'adam', 'hybrid', or 'per_neuron'"
        )

    ### ----- CREATE DATA LOADERS ----- ###
    training_mode = config["training"]["mode"]

    if training_mode == "online":
        print("Using ONLINE data generation...")

        if dimension == 1:
            from src.datamodule import OnlineModularAdditionDataset1D

            # Training dataset
            train_dataset = OnlineModularAdditionDataset1D(
                p=config["data"]["p"],
                template=template_1d,
                k=config["data"]["k"],
                batch_size=config["data"]["batch_size"],
                device=device,
                return_all_outputs=config["model"]["return_all_outputs"],
            )

            # Validation dataset
            val_dataset = OnlineModularAdditionDataset1D(
                p=config["data"]["p"],
                template=template_1d,
                k=config["data"]["k"],
                batch_size=config["data"]["batch_size"],
                device=device,
                return_all_outputs=config["model"]["return_all_outputs"],
            )
        elif dimension == 2:
            from src.datamodule import OnlineModularAdditionDataset2D

            # Training dataset
            train_dataset = OnlineModularAdditionDataset2D(
                p1=config["data"]["p1"],
                p2=config["data"]["p2"],
                template=template_2d,
                k=config["data"]["k"],
                batch_size=config["data"]["batch_size"],
                device=device,
                return_all_outputs=config["model"]["return_all_outputs"],
            )

            # Validation dataset
            val_dataset = OnlineModularAdditionDataset2D(
                p1=config["data"]["p1"],
                p2=config["data"]["p2"],
                template=template_2d,
                k=config["data"]["k"],
                batch_size=config["data"]["batch_size"],
                device=device,
                return_all_outputs=config["model"]["return_all_outputs"],
            )
        elif dimension == "D3":
            # Online training for D3 is not yet implemented
            raise NotImplementedError(
                "Online training mode is not yet implemented for D3. "
                "Please use training.mode='offline' in the config."
            )
        else:
            raise ValueError(f"dimension must be 1, 2, or 'D3', got {dimension}")

        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

        num_steps = config["training"]["num_steps"]
        print(f"  Training for {num_steps} steps")

    elif training_mode == "offline":
        print("Using OFFLINE pre-generated dataset...")
        from torch.utils.data import TensorDataset

        if dimension == 1:
            from src.datamodule import build_modular_addition_sequence_dataset_1d

            # Generate training dataset
            X_train, Y_train, _ = build_modular_addition_sequence_dataset_1d(
                config["data"]["p"],
                template_1d,
                config["data"]["k"],
                mode=config["data"]["mode"],
                num_samples=config["data"]["num_samples"],
                return_all_outputs=config["model"]["return_all_outputs"],
            )

            # Generate validation dataset
            val_samples = max(1000, config["data"]["num_samples"] // 10)
            X_val, Y_val, _ = build_modular_addition_sequence_dataset_1d(
                config["data"]["p"],
                template_1d,
                config["data"]["k"],
                mode="sampled",
                num_samples=val_samples,
                return_all_outputs=config["model"]["return_all_outputs"],
            )
        elif dimension == 2:
            from src.datamodule import build_modular_addition_sequence_dataset_2d

            # Generate training dataset
            X_train, Y_train, _ = build_modular_addition_sequence_dataset_2d(
                config["data"]["p1"],
                config["data"]["p2"],
                template_2d,
                config["data"]["k"],
                mode=config["data"]["mode"],
                num_samples=config["data"]["num_samples"],
                return_all_outputs=config["model"]["return_all_outputs"],
            )

            # Generate validation dataset
            val_samples = max(1000, config["data"]["num_samples"] // 10)
            X_val, Y_val, _ = build_modular_addition_sequence_dataset_2d(
                config["data"]["p1"],
                config["data"]["p2"],
                template_2d,
                config["data"]["k"],
                mode="sampled",
                num_samples=val_samples,
                return_all_outputs=config["model"]["return_all_outputs"],
            )
        elif dimension == "D3":
            from src.datamodule import build_modular_addition_sequence_dataset_D3

            # Generate training dataset
            X_train, Y_train, _ = build_modular_addition_sequence_dataset_D3(
                template_d3,
                config["data"]["k"],
                mode=config["data"]["mode"],
                num_samples=config["data"]["num_samples"],
                return_all_outputs=config["model"]["return_all_outputs"],
            )

            # Generate validation dataset
            val_samples = max(1000, config["data"]["num_samples"] // 10)
            X_val, Y_val, _ = build_modular_addition_sequence_dataset_D3(
                template_d3,
                config["data"]["k"],
                mode="sampled",
                num_samples=val_samples,
                return_all_outputs=config["model"]["return_all_outputs"],
            )
        else:
            raise ValueError(f"dimension must be 1 or 2, got {dimension}")

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)

        train_dataset = TensorDataset(X_train_t, Y_train_t)
        val_dataset = TensorDataset(X_val_t, Y_val_t)

        train_loader = DataLoader(
            train_dataset, batch_size=config["data"]["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

        epochs = config["training"]["epochs"]
        print(f"  Training for {epochs} epochs with {len(train_dataset)} samples")

    else:
        raise ValueError(f"Invalid training mode: {training_mode}. Must be 'online' or 'offline'")

    ### ----- TRAIN MODEL ----- ###
    print(f"Starting training in {training_mode} mode...")

    # Get optional early stopping threshold
    reduction_threshold = config["training"].get("reduction_threshold")
    if reduction_threshold is not None:
        print(f"Early stopping enabled at {reduction_threshold * 100:.1f}% reduction")

    start_time = time.time()

    if training_mode == "online":
        from src.train import train_online

        train_loss_hist, val_loss_hist, param_hist, param_save_indices, final_step = train_online(
            rnn_2d,
            train_loader,
            criterion,
            optimizer,
            num_steps=num_steps,
            verbose_interval=config["training"]["verbose_interval"],
            grad_clip=config["training"]["grad_clip"],
            eval_dataloader=val_loader,
            save_param_interval=config["training"]["save_param_interval"],
            reduction_threshold=reduction_threshold,
        )
    else:  # offline
        from src.train import train

        train_loss_hist, val_loss_hist, param_hist, param_save_indices, final_step = train(
            rnn_2d,
            train_loader,
            criterion,
            optimizer,
            epochs=epochs,
            verbose_interval=config["training"]["verbose_interval"],
            grad_clip=config["training"]["grad_clip"],
            eval_dataloader=val_loader,
            save_param_interval=config["training"]["save_param_interval"],
            reduction_threshold=reduction_threshold,
        )

    training_time = time.time() - start_time

    print("\nTraining complete!")
    print(f"  Final train loss: {train_loss_hist[-1]:.6f}")
    print(f"  Final val loss: {val_loss_hist[-1]:.6f}")
    print(f"  Training time: {training_time:.2f}s")
    if reduction_threshold is not None:
        max_steps_or_epochs = num_steps if training_mode == "online" else epochs
        stopped_early = final_step < max_steps_or_epochs
        status = "CONVERGED" if stopped_early else "DID NOT CONVERGE"
        print(f"  Status: {status} at step/epoch {final_step}")

    ### ----- SAVE RESULTS ----- ###
    metadata = save_results(
        run_dir,
        config,
        rnn_2d,
        train_loss_hist,
        val_loss_hist,
        param_hist,
        template,
        training_time,
        device,
    )

    ### ----- PRODUCE ALL PLOTS ----- ###
    if dimension == 2:
        # Only produce detailed plots for 2D (for now)
        produce_plots_2d(
            run_dir=run_dir,
            config=config,
            model=rnn_2d,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            train_loss_hist=train_loss_hist,
            template_2d=template_2d,
            training_mode=training_mode,
            device=device,
        )
    elif dimension == 1:
        # Produce detailed plots for 1D
        produce_plots_1d(
            run_dir=run_dir,
            config=config,
            model=rnn_2d,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            train_loss_hist=train_loss_hist,
            template_1d=template_1d,
            training_mode=training_mode,
            device=device,
        )
    elif dimension == "D3":
        # Produce basic plots for D3
        produce_plots_D3(
            run_dir=run_dir,
            config=config,
            model=rnn_2d,
            param_hist=param_hist,
            param_save_indices=param_save_indices,
            train_loss_hist=train_loss_hist,
            template_D3=template_d3,
            device=device,
        )
    else:
        raise ValueError(f"dimension must be 1, 2, or 'D3', got {dimension}")

    # Return results dictionary
    results = {
        "final_train_loss": float(train_loss_hist[-1]),
        "final_val_loss": float(val_loss_hist[-1]),
        "training_time": training_time,
        "metadata": metadata,
        "run_dir": str(run_dir),
        "final_step": final_step,
    }

    # Add early stopping info if enabled
    if reduction_threshold is not None:
        max_steps_or_epochs = num_steps if training_mode == "online" else epochs
        results["converged"] = final_step < max_steps_or_epochs

    return results


def main(config: dict):
    """
    Main entry point for single training run.

    Args:
        config: Configuration dictionary.
    """
    train_single_run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train QuadraticRNN or SequentialMLP on 2D modular addition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="gagf/rnns/config.yaml",
        help="Path to config YAML file (default: gagf/rnns/config.yaml)",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
