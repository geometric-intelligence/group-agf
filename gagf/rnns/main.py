import numpy as np
import torch
from gagf.rnns.datamodule import (
    mnist_template_1d,
    mnist_template_2d,
    generate_fourier_template_1d,
    generate_gaussian_template_1d,
    generate_onehot_template_1d,
    generate_template_unique_freqs,
)
from gagf.rnns.optimizers import HybridRNNOptimizer, PerNeuronScaledSGD

from torch.utils.data import DataLoader
from torch import nn, optim
from gagf.rnns.model import QuadraticRNN, SequentialMLP
import time
import yaml
import json
from pathlib import Path
from datetime import datetime
import argparse
import os

from gagf.rnns.utils import (
    plot_training_loss_with_theory, 
    plot_model_predictions_over_time, 
    plot_model_predictions_over_time_1d,
    plot_prediction_power_spectrum_over_time, 
    plot_prediction_power_spectrum_over_time_1d,
    plot_fourier_modes_reference,
    topk_template_freqs,
    topk_template_freqs_1d,
    plot_wout_neuron_specialization,
    plot_wout_neuron_specialization_1d,
    plot_wmix_frequency_structure,
    plot_2d_signal,
    compute_theoretical_final_loss_1d,
    compute_theoretical_final_loss_2d,
)

import matplotlib.pyplot as plt



def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
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
    device: str
) -> dict:
    """Save all experiment results."""
    print(f"Saving results to {run_dir}...")
    
    # Ensure checkpoints directory exists
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
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
        'final_train_loss': float(train_loss_hist[-1]),
        'final_val_loss': float(val_loss_hist[-1]),
        'training_time_seconds': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'device': device,
        'description': config.get('description', ''),
    }
    with open(run_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ All results saved")
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
    device: str
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
    dimension = config['data']['dimension']
    if dimension == 1:
        p_flat = config['data']['p']
    else:
        p_flat = config['data']['p1'] * config['data']['p2']
    
    k = config['data']['k']
    batch_size = config['data']['batch_size']
    total_space_size = p_flat ** k
    
    # Calculate different x-axis values for plotting
    if training_mode == 'online':
        steps = np.arange(len(train_loss_hist))
        samples_seen = batch_size * steps
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Step"
    else:  # offline
        epochs = np.arange(len(train_loss_hist))
        samples_seen = config['data']['num_samples'] * epochs
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Epoch"
    
    # Save x-axis data
    np.save(run_dir / "samples_seen.npy", samples_seen)
    np.save(run_dir / "fraction_of_space_seen.npy", fraction_of_space)
    
    print(f"Total data space: {total_space_size:,} sequences")
    print(f"Samples seen: {samples_seen[-1]:,} ({fraction_of_space[-1]*100:.4f}% of space)")
    
    ### ----- GENERATE EVALUATION DATA ----- ###
    print("Generating evaluation data for visualization...")
    from gagf.rnns.datamodule import build_modular_addition_sequence_dataset_2d
    X_seq_2d, Y_seq_2d, _ = build_modular_addition_sequence_dataset_2d(
        config['data']['p1'], 
        config['data']['p2'], 
        template_2d, 
        config['data']['k'], 
        mode="sampled", 
        num_samples=min(config['data']['num_samples'], 1000),
        return_all_outputs=config['model']['return_all_outputs'],
    )
    X_seq_2d_t = torch.tensor(X_seq_2d, dtype=torch.float32, device=device)
    Y_seq_2d_t = torch.tensor(Y_seq_2d, dtype=torch.float32, device=device)
    print(f"  Generated {X_seq_2d_t.shape[0]} samples for visualization")
    
    ### ----- COMPUTE CHECKPOINT INDICES ----- ###
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config['analysis']['checkpoints']
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]
    
    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")
    print(f"  Corresponding to step/epoch indices: {[param_save_indices[i] for i in checkpoint_indices]}")
    
    ### ----- PLOT TRAINING LOSS ----- ###
    print("\nPlotting training loss...")
    
    # Plot 1: Loss vs Steps/Epochs
    plot_training_loss_with_theory(
        loss_history=train_loss_hist,
        template_2d=template_2d, 
        p1=config['data']['p1'], 
        p2=config['data']['p2'],
        x_values=None,
        x_label=x_label_steps,
        save_path=os.path.join(run_dir, "training_loss_vs_steps.pdf"),
        show=False,
    )
    
    # Plot 2: Loss vs Samples Seen
    plot_training_loss_with_theory(
        loss_history=train_loss_hist,
        template_2d=template_2d, 
        p1=config['data']['p1'], 
        p2=config['data']['p2'],
        x_values=samples_seen,
        x_label="Samples Seen",
        save_path=os.path.join(run_dir, "training_loss_vs_samples.pdf"),
        show=False,
    )
    
    # Plot 3: Loss vs Fraction of Space
    plot_training_loss_with_theory(
        loss_history=train_loss_hist,
        template_2d=template_2d, 
        p1=config['data']['p1'], 
        p2=config['data']['p2'],
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
        config['data']['p1'],
        config['data']['p2'],
        steps=checkpoint_indices,
        save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
        show=False
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
    model_type = config['model']['model_type']
    if model_type == 'QuadraticRNN':
        print("Visualizing W_mix frequency structure...")
        plot_wmix_frequency_structure(
            param_hist,
            tracked_freqs,
            colors,
            config['data']['p1'],
            config['data']['p2'],
            steps=checkpoint_indices,
            within_group_order="phase",
            dead_l2_thresh=0.1,
            save_path=os.path.join(run_dir, "wmix_frequency_structure.pdf"),
            show=False
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
    device: str
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
    p = config['data']['p']
    k = config['data']['k']
    batch_size = config['data']['batch_size']
    total_space_size = p ** k
    
    # Calculate different x-axis values for plotting
    if training_mode == 'online':
        steps = np.arange(len(train_loss_hist))
        samples_seen = batch_size * steps
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Step"
    else:  # offline
        epochs = np.arange(len(train_loss_hist))
        samples_seen = config['data']['num_samples'] * epochs
        fraction_of_space = samples_seen / total_space_size
        x_label_steps = "Epoch"
    
    # Save x-axis data
    np.save(run_dir / "samples_seen.npy", samples_seen)
    np.save(run_dir / "fraction_of_space_seen.npy", fraction_of_space)
    
    print(f"Total data space: {total_space_size:,} sequences")
    print(f"Samples seen: {samples_seen[-1]:,} ({fraction_of_space[-1]*100:.4f}% of space)")
    
    ### ----- GENERATE EVALUATION DATA ----- ###
    print("Generating evaluation data for visualization...")
    from gagf.rnns.datamodule import build_modular_addition_sequence_dataset_1d
    X_seq_1d, Y_seq_1d, _ = build_modular_addition_sequence_dataset_1d(
        config['data']['p'], 
        template_1d, 
        config['data']['k'], 
        mode="sampled", 
        num_samples=min(config['data']['num_samples'], 1000),
        return_all_outputs=config['model']['return_all_outputs'],
    )
    X_seq_1d_t = torch.tensor(X_seq_1d, dtype=torch.float32, device=device)
    Y_seq_1d_t = torch.tensor(Y_seq_1d, dtype=torch.float32, device=device)
    print(f"  Generated {X_seq_1d_t.shape[0]} samples for visualization")
    
    ### ----- COMPUTE CHECKPOINT INDICES ----- ###
    total_checkpoints = len(param_hist)
    checkpoint_fractions = config['analysis']['checkpoints']
    checkpoint_indices = [int(f * (total_checkpoints - 1)) for f in checkpoint_fractions]
    
    print(f"Analysis checkpoints: {checkpoint_indices} (out of {total_checkpoints})")
    print(f"  Corresponding to step/epoch indices: {[param_save_indices[i] for i in checkpoint_indices]}")
    
    ### ----- PLOT TRAINING LOSS ----- ###
    print("\nPlotting training loss...")
    
    # Create a 2x2 subplot for different scale combinations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x_values = steps if training_mode == 'online' else epochs
    
    scale_configs = [
        ('linear', 'linear', 'Linear Scale'),
        ('linear', 'log', 'Log Y'),
        ('log', 'linear', 'Log X'),
        ('log', 'log', 'Log-Log'),
    ]
    
    for ax, (xscale, yscale, title) in zip(axes.flat, scale_configs):
        ax.plot(x_values, train_loss_hist, lw=2, color='#1f77b4')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(x_label_steps)
        ax.set_ylabel('Training Loss')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_loss.pdf"), bbox_inches='tight', dpi=150)
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
        show=False
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
        show=False
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


def produce_plots_D3(
    run_dir: Path,
    config: dict,
    model,
    param_hist,
    param_save_indices,
    train_loss_hist,
    template_D3: np.ndarray,
):
    """
    Generate all analysis plots after training (D3 version).
    
    Args:
        run_dir: Directory to save plots
        config: Configuration dictionary (must have dimension=3)
        model: Trained model (QuadraticRNN or SequentialMLP)
        param_hist: List of parameter snapshots
        param_save_indices: Indices where params were saved

        train_loss_hist: Training loss history
        template_D3: 3D template array (p, p, p)
    """
    print("\n=== Generating Analysis Plots (D3) ===")
    
    ### ----- COMPUTE X-AXIS VALUES ----- ###
    p = config['data']['p']
    k = config['data']['k']

    # TODO(Nina): Using code in binary_action_learning.py

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
    np.random.seed(config['data']['seed'])
    torch.manual_seed(config['data']['seed'])
    
    # Determine device
    device = config['device'] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ### ----- GENERATE DATA ----- ###
    print("Generating data...")
    
    dimension = config['data']['dimension'] # TODO(Nina): Here, it will be a string D3.
    template_type = config['data']['template_type'] # TODO(Nina): Here, use a one-hot template for D3, i.e., a one-hot vector of length D3.order() = 6.
    
    if dimension == 1:
        # 1D template generation
        p = config['data']['p']
        p_flat = p
        
        if template_type == 'mnist':
            template_1d = mnist_template_1d(p, config['data']['mnist_label'], root="data")
        elif template_type == 'fourier':
            n_freqs = config['data']["n_freqs"]
            template_1d = generate_fourier_template_1d(p, n_freqs=n_freqs, seed=config['data']['seed'])
        elif template_type == 'gaussian':
            template_1d = generate_gaussian_template_1d(p, n_gaussians=3, seed=config['data']['seed'])
        elif template_type == 'onehot':
            template_1d = generate_onehot_template_1d(p)
        else:
            raise ValueError(f"Unknown template_type: {template_type}")
        
        template_1d = template_1d - np.mean(template_1d)
        template = template_1d  # For consistency in code below
        
        # Visualize 1D template
        print("Visualizing template...")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(template_1d)
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.set_title('1D Template')
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved template")
        
    elif dimension == 2:
        # 2D template generation
        p1 = config['data']['p1']
        p2 = config['data']['p2']
        p_flat = p1 * p2
        
        if template_type == 'mnist':
            template_2d = mnist_template_2d(p1, p2, config['data']['mnist_label'], root="data")
        elif template_type == 'fourier':
            n_freqs = config['data']["n_freqs"]
            template_2d = generate_template_unique_freqs(p1, p2, n_freqs=n_freqs, seed=config['data']['seed'])
        else:
            raise ValueError(f"Unknown template_type for 2D: {template_type}")
        
        template_2d = template_2d - np.mean(template_2d)
        template = template_2d  # For consistency in code below
        
        # Visualize 2D template
        print("Visualizing template...")
        fig, ax = plot_2d_signal(template_2d, title="Template", cmap="gray")
        fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved template")
    elif dimension == 'D3':
        from escnn.group import DihedralGroup
        D3 = DihedralGroup(order=3)
        template_d3 = np.eye(D3.order())
        template_d3 = template_d3 - np.mean(template_d3)
        template = template_d3  # For consistency in code below
        
        # Visualize 3D template
        print("Visualizing template...")
        fig, ax = plt.plot(template_d3, title="Template", cmap="gray")
        fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved template")
    else:
        raise ValueError(f"dimension must be 1 or 2, got {dimension}")

    
    ### ----- SETUP TRAINING ----- ###
    print("Setting up model and training...")
    
    # Flatten template for model (works for both 1D and 2D)
    template_torch = torch.tensor(template, device=device, dtype=torch.float32).flatten()
    
    # Determine which model to use
    model_type = config['model']['model_type']
    print(f"Using model type: {model_type}")
    
    if model_type == 'QuadraticRNN':
        rnn_2d = QuadraticRNN(
            p=p_flat,
            d=config['model']['hidden_dim'],
            template=template_torch,
            init_scale=config['model']['init_scale'],
            return_all_outputs=config['model']['return_all_outputs'],
            transform_type=config['model']['transform_type'],
        ).to(device)
    elif model_type == 'SequentialMLP':
        rnn_2d = SequentialMLP( #TODO(NiT): Rename rnn_2d, in both if loops, to model or something similar.
            p=p_flat,
            d=config['model']['hidden_dim'],
            template=template_torch,
            k=config['data']['k'],
            init_scale=config['model']['init_scale'],
            return_all_outputs=config['model']['return_all_outputs'],
        ).to(device)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'QuadraticRNN' or 'SequentialMLP'")
    
    criterion = nn.MSELoss()

    # Optimizer selection with model-aware defaults
    optimizer_name = config['training']['optimizer']
    
    # Auto-select optimizer if not specified or if 'auto'
    if optimizer_name == 'auto' or (optimizer_name not in ['adam', 'hybrid', 'per_neuron']):
        if model_type == 'SequentialMLP':
            optimizer_name = 'per_neuron'
            print(f"Auto-selected optimizer: {optimizer_name} (recommended for SequentialMLP)")
        else:
            optimizer_name = 'adam'
            print(f"Auto-selected optimizer: {optimizer_name}")
    else:
        print(f"Using optimizer: {optimizer_name}")

    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            rnn_2d.parameters(), 
            lr=config['training']['learning_rate'], 
            betas=tuple(config['training']['betas']), 
            weight_decay=config['training']['weight_decay']
        )
    elif optimizer_name == 'hybrid':
        if model_type != 'QuadraticRNN':
            raise ValueError(f"'hybrid' optimizer is only supported for QuadraticRNN, got {model_type}")
        optimizer = HybridRNNOptimizer(
            rnn_2d,
            lr=1,
            scaling_factor=config['training']['scaling_factor'],
            adam_lr=config['training']['learning_rate'],
            adam_betas=tuple(config['training']['betas']),
            adam_eps=1e-8,
        )
    elif optimizer_name == 'per_neuron':
        # Per-neuron scaled SGD (recommended for SequentialMLP)
        degree = config['training']['degree']
        lr = config['training']['learning_rate']
        
        # For SequentialMLP, use lr=1.0 by default if not specified
        if model_type == 'SequentialMLP' and lr == 1.0e-3:
            print("  Note: Using lr=1.0 for per_neuron optimizer with SequentialMLP")
            lr = 1.0
        
        optimizer = PerNeuronScaledSGD(
            rnn_2d,
            lr=lr,
            degree=degree  # Will auto-infer as k+1 for SequentialMLP (k = sequence length)
        )
        print(f"  Degree of homogeneity: {optimizer.param_groups[0]['degree']}")
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_name}. Must be 'adam', 'hybrid', or 'per_neuron'")
    
    
    ### ----- CREATE DATA LOADERS ----- ###
    training_mode = config['training']['mode']
    
    if training_mode == 'online':
        print("Using ONLINE data generation...")
        
        if dimension == 1:
            from gagf.rnns.datamodule import OnlineModularAdditionDataset1D
            
            # Training dataset
            train_dataset = OnlineModularAdditionDataset1D( 
                p=config['data']['p'],
                template=template_1d,
                k=config['data']['k'],
                batch_size=config['data']['batch_size'],
                device=device,
                return_all_outputs=config['model']['return_all_outputs'],
            )
            
            # Validation dataset
            val_dataset = OnlineModularAdditionDataset1D(
                p=config['data']['p'],
                template=template_1d,
                k=config['data']['k'],
                batch_size=config['data']['batch_size'],
                device=device,
                return_all_outputs=config['model']['return_all_outputs'],
            )
        elif dimension == 2:
            from gagf.rnns.datamodule import OnlineModularAdditionDataset2D
            
            # Training dataset
            train_dataset = OnlineModularAdditionDataset2D(
                p1=config['data']['p1'],
                p2=config['data']['p2'],
                template=template_2d,
                k=config['data']['k'],
                batch_size=config['data']['batch_size'],
                device=device,
                return_all_outputs=config['model']['return_all_outputs'],
            )
            
            # Validation dataset
            val_dataset = OnlineModularAdditionDataset2D(
                p1=config['data']['p1'],
                p2=config['data']['p2'],
                template=template_2d,
                k=config['data']['k'],
                batch_size=config['data']['batch_size'],
                device=device,
                return_all_outputs=config['model']['return_all_outputs'],
            )
        else: # Note: nothing to add here yet for D3.
            raise ValueError(f"dimension must be 1 or 2, got {dimension}")
        
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)
        
        num_steps = config['training']['num_steps']
        print(f"  Training for {num_steps} steps")
        
    elif training_mode == 'offline':
        print("Using OFFLINE pre-generated dataset...")
        from torch.utils.data import TensorDataset
        
        if dimension == 1:
            from gagf.rnns.datamodule import build_modular_addition_sequence_dataset_1d
            
            # Generate training dataset
            X_train, Y_train, _ = build_modular_addition_sequence_dataset_1d(
                config['data']['p'], 
                template_1d, 
                config['data']['k'], 
                mode=config['data']['mode'], 
                num_samples=config['data']['num_samples'],
                return_all_outputs=config['model']['return_all_outputs'],
            )
            
            # Generate validation dataset
            val_samples = max(1000, config['data']['num_samples'] // 10)
            X_val, Y_val, _ = build_modular_addition_sequence_dataset_1d(
                config['data']['p'], 
                template_1d, 
                config['data']['k'], 
                mode='sampled',
                num_samples=val_samples,
                return_all_outputs=config['model']['return_all_outputs'],
            )
        elif dimension == 2:
            from gagf.rnns.datamodule import build_modular_addition_sequence_dataset_2d
            
            # Generate training dataset
            X_train, Y_train, _ = build_modular_addition_sequence_dataset_2d(
                config['data']['p1'], 
                config['data']['p2'], 
                template_2d, 
                config['data']['k'], 
                mode=config['data']['mode'], 
                num_samples=config['data']['num_samples'],
                return_all_outputs=config['model']['return_all_outputs'],
            )
            
            # Generate validation dataset
            val_samples = max(1000, config['data']['num_samples'] // 10)
            X_val, Y_val, _ = build_modular_addition_sequence_dataset_2d(
                config['data']['p1'], 
                config['data']['p2'], 
                template_2d, 
                config['data']['k'], 
                mode='sampled',
                num_samples=val_samples,
                return_all_outputs=config['model']['return_all_outputs'],
            )
        elif dimension == 'D3': # TODO(Nina): Add the option to ingest the D3 train and validation datasets.
            from gagf.rnns.datamodule import build_modular_addition_sequence_dataset_d3
            X_train, Y_train, _ = build_modular_addition_sequence_dataset_d3(
                config['data']['p'],
                template_d3,
                config['data']['k'],
                mode=config['data']['mode'],
                num_samples=config['data']['num_samples'],
                return_all_outputs=config['model']['return_all_outputs'],
            )
            X_val, Y_val, _ = build_modular_addition_sequence_dataset_d3(
                config['data']['p'],
                template_d3,
                config['data']['k'],
                mode='sampled',
                num_samples=val_samples,
                return_all_outputs=config['model']['return_all_outputs'],
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
            train_dataset, 
            batch_size=config['data']['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False
        )
        
        epochs = config['training']['epochs']
        print(f"  Training for {epochs} epochs with {len(train_dataset)} samples")
    
    else:
        raise ValueError(f"Invalid training mode: {training_mode}. Must be 'online' or 'offline'")
    
    ### ----- TRAIN MODEL ----- ###
    print(f"Starting training in {training_mode} mode...")
    
    # Get optional early stopping threshold
    reduction_threshold = config['training'].get('reduction_threshold')
    if reduction_threshold is not None:
        print(f"Early stopping enabled at {reduction_threshold*100:.1f}% reduction")
    
    start_time = time.time()
    
    if training_mode == 'online':
        from gagf.rnns.train import train_online
        train_loss_hist, val_loss_hist, param_hist, param_save_indices, final_step = train_online(
            rnn_2d,
            train_loader,
            criterion,
            optimizer,
            num_steps=num_steps,
            verbose_interval=config['training']['verbose_interval'],
            grad_clip=config['training']['grad_clip'],
            eval_dataloader=val_loader,
            save_param_interval=config['training']['save_param_interval'],
            reduction_threshold=reduction_threshold,
        )
    else:  # offline
        from gagf.rnns.train import train
        train_loss_hist, val_loss_hist, param_hist, param_save_indices, final_step = train(
            rnn_2d,
            train_loader,
            criterion,
            optimizer,
            epochs=epochs,
            verbose_interval=config['training']['verbose_interval'],
            grad_clip=config['training']['grad_clip'],
            eval_dataloader=val_loader,
            save_param_interval=config['training']['save_param_interval'],
            reduction_threshold=reduction_threshold,
        )
    
    training_time = time.time() - start_time

    print(f"\nTraining complete!")
    print(f"  Final train loss: {train_loss_hist[-1]:.6f}")
    print(f"  Final val loss: {val_loss_hist[-1]:.6f}")
    print(f"  Training time: {training_time:.2f}s")
    if reduction_threshold is not None:
        max_steps_or_epochs = num_steps if training_mode == 'online' else epochs
        stopped_early = final_step < max_steps_or_epochs
        status = "CONVERGED" if stopped_early else "DID NOT CONVERGE"
        print(f"  Status: {status} at step/epoch {final_step}")

    ### ----- SAVE RESULTS ----- ###
    metadata = save_results(
        run_dir, config, rnn_2d, 
        train_loss_hist, val_loss_hist, 
        param_hist,
        template, training_time, device
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
            device=device
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
            device=device
        )
    else: # TODO(Nina): Add the option to ingest the D3 train and validation datasets.
        raise ValueError(f"dimension must be 1 or 2, got {dimension}")
    
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
        max_steps_or_epochs = num_steps if training_mode == 'online' else epochs
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
    parser = argparse.ArgumentParser(description="Train QuadraticRNN or SequentialMLP on 2D modular addition")
    parser.add_argument(
        "--config", 
        type=str, 
        default="gagf/rnns/config.yaml",
        help="Path to config YAML file (default: gagf/rnns/config.yaml)"
    )

    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)