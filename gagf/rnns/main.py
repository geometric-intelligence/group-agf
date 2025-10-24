import numpy as np
import torch
from gagf.rnns.datamodule import build_modular_addition_sequence_dataset_2d, generate_template_unique_freqs
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

from gagf.rnns.utils import plot_training_loss_with_theory


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
    
    template_2d = generate_template_unique_freqs(
        config['data']['p1'], 
        config['data']['p2'], 
        n_freqs=config['data']['n_freqs']
    )
    template_2d = template_2d - np.mean(template_2d)
    
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


    plot_training_loss_with_theory(loss_hist, template_2d, config['data']['p1'], config['data']['p2'], run_dir)
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