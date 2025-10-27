import numpy as np
import torch
from gagf.rnns.datamodule import (
    mnist_template_2d,
    generate_template_unique_freqs,
    build_modular_addition_sequence_dataset_2d,
)

from gagf.rnns.train import train
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from gagf.rnns.model import QuadraticRNN
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
    plot_prediction_power_spectrum_over_time, 
    plot_fourier_modes_reference,
    topk_template_freqs,
    plot_wout_neuron_specialization,
    plot_wmix_frequency_structure,
    plot_2d_signal,
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


def save_results(run_dir: Path, config: dict, model, loss_hist, acc_hist, param_hist, 
                 template: np.ndarray, training_time: float, device: str):
    """Save all experiment results."""
    print(f"Saving results to {run_dir}...")
    
    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save template
    np.save(run_dir / "template.npy", template)
    
    # Save training history
    np.save(run_dir / "loss_history.npy", np.array(loss_hist))
    np.save(run_dir / "acc_history.npy", np.array(acc_hist))
    torch.save(param_hist, run_dir / "param_history.pt")
    
    # Save final model
    torch.save(model.state_dict(), run_dir / "checkpoints" / "final_model.pt")
    
    # Save metadata
    metadata = {
        'final_loss': float(loss_hist[-1]),
        'final_accuracy': float(acc_hist[-1]),
        'training_time_seconds': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'device': device,
        'description': config.get('description', ''),
    }
    with open(run_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ All results saved")
    return metadata


def main(config: dict):
    """
    Train a QuadraticRNN on 2D modular addition.
    
    Args:
        config: Configuration dictionary.
    """
    # Setup run directory
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
    
    # template_2d = generate_template_unique_freqs(
    #     config['data']['p1'], 
    #     config['data']['p2'], 
    #     n_freqs=config['data']['n_freqs']
    # )
    template_2d = mnist_template_2d(
        config['data']['p1'],
        config['data']['p2'],
        config['data']['mnist_label'],
        root="data"
    )
    template_2d = template_2d - np.mean(template_2d)

    ### ----- VISUALIZE TEMPLATE ----- ###
    print("Visualizing template...")
    fig, ax = plot_2d_signal(
        template_2d,
        title="Template",
        cmap="gray"
    )
    fig.savefig(os.path.join(run_dir, "template.pdf"), bbox_inches="tight", dpi=150)
    print(f"  ✓ Saved template")
    
    p_flat = config['data']['p1'] * config['data']['p2']
    X_seq_2d, Y_seq_2d, sequence_xy = build_modular_addition_sequence_dataset_2d(
        config['data']['p1'], 
        config['data']['p2'], 
        template_2d, 
        config['data']['k'], 
        mode=config['data']['mode'], 
        num_samples=config['data']['num_samples']
    )
    
    X_seq_2d_t = torch.tensor(X_seq_2d, dtype=torch.float32, device=device)
    Y_seq_2d_t = torch.tensor(Y_seq_2d, dtype=torch.float32, device=device)
    
    print(f"  X: {X_seq_2d_t.shape}, Y: {Y_seq_2d_t.shape}")
    
    ### ----- SETUP TRAINING ----- ###
    print("Setting up model and training...")
    
    seq_dataset_2d = TensorDataset(X_seq_2d_t, Y_seq_2d_t)
    seq_loader_2d = DataLoader(
        seq_dataset_2d, 
        batch_size=config['data']['batch_size'], 
        shuffle=True, 
        drop_last=False
    )
    
    template_torch = torch.tensor(template_2d, device=device, dtype=torch.float32).flatten()
    rnn_2d = QuadraticRNN(
        p=p_flat,
        d=config['model']['hidden_dim'],
        template=template_torch,
        init_scale=config['model']['init_scale'],
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        rnn_2d.parameters(), 
        lr=config['training']['learning_rate'], 
        betas=tuple(config['training']['betas']), 
        weight_decay=config['training']['weight_decay']
    )
    
    ### ----- TRAIN MODEL ----- ###
    print(f"Training for {config['training']['epochs']} epochs...")
    start_time = time.time()
    
    loss_hist, acc_hist, param_hist = train(
        rnn_2d,
        seq_loader_2d,
        criterion,
        optimizer,
        epochs=config['training']['epochs'],
        verbose_interval=config['training']['verbose_interval'],
        grad_clip=config['training']['grad_clip'],
    )
    
    training_time = time.time() - start_time
    
    ### ----- SAVE RESULTS ----- ###
    metadata = save_results(
        run_dir, config, rnn_2d, loss_hist, acc_hist, param_hist,
        template_2d, training_time, device
    )

    ### ----- PLOT TRAINING LOSS ----- ###
    print("Plotting training loss...")


    plot_training_loss_with_theory(
        loss_history=loss_hist, 
        template_2d=template_2d, 
        p1=config['data']['p1'], 
        p2=config['data']['p2'], 
        save_path=os.path.join(run_dir, "training_loss.pdf"),
        show=False,
    )

    steps = config['analysis']['steps']

    ### ----- PLOT MODEL PREDICTIONS OVER TIME ----- ###
    print("Plotting model predictions over time...")
    plot_model_predictions_over_time(
        rnn_2d,
        param_hist,
        X_seq_2d_t,
        Y_seq_2d_t,
        config['data']['p1'],
        config['data']['p2'],
        steps=steps,
        save_path=os.path.join(run_dir, "predictions_over_time.pdf"),
        show=False
    )

    ### ----- PLOT POWER SPECTRUM ANALYSIS ----- ###
    print("Analyzing power spectrum of predictions over training...")
    plot_prediction_power_spectrum_over_time(
        rnn_2d,
        param_hist,
        X_seq_2d_t,
        Y_seq_2d_t,
        template_2d,
        config['data']['p1'],
        config['data']['p2'],
        loss_history=loss_hist,
        num_freqs_to_track=10,
        num_analysis_steps=100,
        num_samples=100,
        save_path=os.path.join(run_dir, "power_spectrum_analysis.pdf"),
        show=False
    )

    ### ----- PLOT FOURIER MODES REFERENCE ----- ###
    print("Creating Fourier modes reference...")
    tracked_freqs = topk_template_freqs(template_2d, K=10)
    # Use same colors as power spectrum plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_freqs)))

    plot_fourier_modes_reference(
        tracked_freqs,
        colors,
        config['data']['p1'],
        config['data']['p2'],
        save_path=os.path.join(run_dir, "fourier_modes_reference.pdf"),
        save_individual=True,
        individual_dir=os.path.join(run_dir, "fourier_modes"),
        show=False
    )

    ### ----- PLOT W_OUT NEURON SPECIALIZATION ----- ###
    print("Visualizing W_out neuron specialization...")

    plot_wout_neuron_specialization(
        param_hist,
        tracked_freqs,
        colors,
        config['data']['p1'],
        config['data']['p2'],
        steps=steps,
        dead_thresh_l2=0.25,
        save_dir=run_dir,
        show=False
    )

    ### ----- PLOT W_MIX FREQUENCY STRUCTURE ----- ###
    print("Visualizing W_mix frequency structure...")
    plot_wmix_frequency_structure(
        param_hist,
        tracked_freqs,
        colors,
        config['data']['p1'],
        config['data']['p2'],
        steps=steps,
        within_group_order="phase",
        dead_l2_thresh=0.1,
        save_path=os.path.join(run_dir, "wmix_frequency_structure.pdf"),
        show=False
    )
    
    
    print(f"Training complete!")
    print(f"  Final loss: {metadata['final_loss']:.6f}")
    print(f"  Final accuracy: {metadata['final_accuracy']:.2f}%")
    print(f"  Training time: {training_time:.2f}s")
    
    return rnn_2d, loss_hist, acc_hist, param_hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QuadraticRNN on 2D modular addition")
    parser.add_argument(
        "--config", 
        type=str, 
        default="gagf/rnns/config.yaml",
        help="Path to config YAML file (default: gagf/rnns/config.yaml)"
    )

    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)