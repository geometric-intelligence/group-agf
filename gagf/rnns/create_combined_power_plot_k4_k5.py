#!/usr/bin/env python3
"""
Create a combined 2x3 plot showing power spectrum evolution for k=4 and k=5.
Each row corresponds to a k value, each column to a scale type (linear, log-x, log-log).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import os
from pathlib import Path
from escnn.group import DihedralGroup
from group_agf.binary_action_learning.group_fourier_transform import compute_group_fourier_coef
from gagf.rnns.model import SequentialMLP
from gagf.rnns.datamodule import build_modular_addition_sequence_dataset_D3

def load_run_data(run_dir):
    """Load all necessary data from a run directory."""
    run_dir = Path(run_dir)
    
    # Load config
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Load template
    template = np.load(run_dir / "template.npy")
    
    # Load parameter history
    param_hist = torch.load(run_dir / "param_history.pt", map_location='cpu')
    
    # Get k value
    k = config['data']['k']
    
    # Create model
    D3 = DihedralGroup(N=3)
    group_order = D3.order()
    template_t = torch.tensor(template, dtype=torch.float32)
    model = SequentialMLP(
        p=group_order,
        d=config['model']['hidden_dim'],
        template=template_t,
        k=k,
        init_scale=config['model']['init_scale'],
    )
    
    # Generate evaluation data
    X_eval, Y_eval, _ = build_modular_addition_sequence_dataset_D3(
        template, k, mode="sampled", num_samples=100, return_all_outputs=False
    )
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32)
    
    # Compute template power
    irreps = D3.irreps()
    n_irreps = len(irreps)
    template_power = np.zeros(n_irreps)
    for i, irrep in enumerate(irreps):
        fourier_coef = compute_group_fourier_coef(D3, template, irrep)
        template_power[i] = irrep.size * np.trace(fourier_coef.conj().T @ fourier_coef)
    template_power = template_power / group_order
    
    # Compute param_save_indices
    save_interval = config['training'].get('save_param_interval', 10) or 10
    param_save_indices = [i * save_interval for i in range(len(param_hist))]
    
    return {
        'config': config,
        'template': template,
        'param_hist': param_hist,
        'param_save_indices': param_save_indices,
        'model': model,
        'X_eval_t': X_eval_t,
        'D3': D3,
        'template_power': template_power,
        'k': k,
    }

def compute_power_evolution(run_data, num_checkpoints_to_sample=50, num_samples_for_power=100):
    """Compute power evolution over training."""
    model = run_data['model']
    param_hist = run_data['param_hist']
    param_save_indices = run_data['param_save_indices']
    X_eval_t = run_data['X_eval_t']
    D3 = run_data['D3']
    
    irreps = D3.irreps()
    n_irreps = len(irreps)
    
    # Sample checkpoints
    total_checkpoints = len(param_hist)
    if total_checkpoints <= num_checkpoints_to_sample:
        sampled_ckpt_indices = list(range(total_checkpoints))
    else:
        sampled_ckpt_indices = np.linspace(0, total_checkpoints - 1, num_checkpoints_to_sample, dtype=int).tolist()
    
    epoch_numbers = [param_save_indices[i] for i in sampled_ckpt_indices]
    
    # Compute model output power at each checkpoint
    model_powers = np.zeros((len(sampled_ckpt_indices), n_irreps))
    X_subset = X_eval_t[:num_samples_for_power]
    
    for i, ckpt_idx in enumerate(sampled_ckpt_indices):
        model.load_state_dict(param_hist[ckpt_idx])
        model.eval()
        
        with torch.no_grad():
            outputs = model(X_subset)
            outputs_np = outputs.cpu().numpy()
        
        powers = np.zeros((len(outputs_np), n_irreps))
        for sample_i, output in enumerate(outputs_np):
            for irrep_i, irrep in enumerate(irreps):
                fourier_coef = compute_group_fourier_coef(D3, output, irrep)
                powers[sample_i, irrep_i] = irrep.size * np.trace(fourier_coef.conj().T @ fourier_coef)
        powers = powers / D3.order()
        model_powers[i] = np.mean(powers, axis=0)
    
    return epoch_numbers, model_powers

def create_combined_plot(run_dirs_dict, save_path):
    """Create 2x3 combined plot."""
    # run_dirs_dict: {k: run_dir}
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.15, 1, 1], hspace=0.3, wspace=0.3)
    
    # Top row for common parameters
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    # Load first run to get common parameters
    first_run_dir = list(run_dirs_dict.values())[0]
    first_run_data = load_run_data(first_run_dir)
    common_config = first_run_data['config']
    
    # Extract common parameters
    hidden_dim = common_config['model']['hidden_dim']
    mode = common_config['data']['mode']
    optimizer = common_config['training']['optimizer']
    
    # Create title with common parameters
    title_text = f'D3 Power Spectrum Evolution Over Training\n'
    title_text += f'Common Parameters: hidden_dim={hidden_dim}, mode={mode}, optimizer={optimizer}'
    ax_title.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=14, fontweight='bold',
                  transform=ax_title.transAxes)
    
    # Create axes for plots with shared x-axes for each column
    axes = []
    # First create all axes in the first row (no sharing yet)
    for col in range(3):
        axes.append([fig.add_subplot(gs[1, col])])
    
    # Then create remaining rows sharing x-axis with first row in each column
    for row in range(1, 2):
        for col in range(3):
            axes[col].append(fig.add_subplot(gs[row+1, col], sharex=axes[col][0]))
    
    # Convert to row-major format for easier indexing
    axes = np.array([[axes[col][row] for col in range(3)] for row in range(2)])
    
    k_values = sorted(run_dirs_dict.keys())
    
    for row_idx, k in enumerate(k_values):
        run_dir = run_dirs_dict[k]
        print(f"Loading k={k} from {run_dir}...")
        run_data = load_run_data(run_dir)
        
        epoch_numbers, model_powers = compute_power_evolution(run_data)
        template_power = run_data['template_power']
        D3 = run_data['D3']
        irreps = D3.irreps()
        config = run_data['config']
        
        # Get row-specific parameters
        learning_rate = config['training']['learning_rate']
        init_scale = config['model']['init_scale']
        
        # Format init_scale nicely
        if init_scale >= 1e-3:
            init_scale_str = f"{init_scale:.0e}"
        elif init_scale >= 1e-6:
            init_scale_str = f"{init_scale:.1e}"
        else:
            init_scale_str = f"{init_scale:.2e}"
        
        # Format learning_rate nicely
        if learning_rate >= 1e-3:
            lr_str = f"{learning_rate:.0e}"
        elif learning_rate >= 1e-6:
            lr_str = f"{learning_rate:.1e}"
        else:
            lr_str = f"{learning_rate:.2e}"
        
        # Row label
        row_label = f'k={k}, lr={lr_str}, init_scale={init_scale_str}'
        
        # Get top irreps
        top_k_irreps = min(5, len(irreps))
        top_irrep_indices = np.argsort(template_power)[::-1][:top_k_irreps]
        colors_line = plt.cm.tab10(np.linspace(0, 1, top_k_irreps))
        
        # Filter for log scales
        valid_mask = np.array(epoch_numbers) > 0
        valid_epochs = np.array(epoch_numbers)[valid_mask]
        valid_model_powers = model_powers[valid_mask, :]
        
        # Column 1: Linear scales
        ax = axes[row_idx, 0]
        for i, irrep_idx in enumerate(top_irrep_indices):
            power_values = model_powers[:, irrep_idx]
            ax.plot(epoch_numbers, power_values, '-', lw=2, color=colors_line[i],
                    label=f'Irrep {irrep_idx} (dim={irreps[irrep_idx].size})')
            ax.axhline(template_power[irrep_idx], linestyle='--', alpha=0.5, color=colors_line[i])
        if row_idx == 1:  # Only bottom row shows xlabel
            ax.set_xlabel('Epoch')
        ax.set_ylabel('Power')
        if row_idx == 0:
            col_title = 'Linear Scales'
        else:
            col_title = ''
        ax.set_title(f'{col_title}\n{row_label}', fontsize=12 if row_idx == 0 else 10)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
        # Hide x-axis labels for non-bottom rows (they're shared)
        if row_idx < 1:
            ax.tick_params(labelbottom=False)
        
        # Column 2: Log x-axis
        ax = axes[row_idx, 1]
        for i, irrep_idx in enumerate(top_irrep_indices):
            power_values = valid_model_powers[:, irrep_idx]
            ax.plot(valid_epochs, power_values, '-', lw=2, color=colors_line[i],
                    label=f'Irrep {irrep_idx} (dim={irreps[irrep_idx].size})')
            ax.axhline(template_power[irrep_idx], linestyle='--', alpha=0.5, color=colors_line[i])
        ax.set_xscale('log')
        if row_idx == 1:  # Only bottom row shows xlabel
            ax.set_xlabel('Epoch (log scale)')
        ax.set_ylabel('Power')
        if row_idx == 0:
            col_title = 'Log X-axis'
        else:
            col_title = ''
        ax.set_title(f'{col_title}\n{row_label}', fontsize=12 if row_idx == 0 else 10)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
        # Hide x-axis labels for non-bottom rows (they're shared)
        if row_idx < 1:
            ax.tick_params(labelbottom=False)
        
        # Column 3: Log-log scales
        ax = axes[row_idx, 2]
        for i, irrep_idx in enumerate(top_irrep_indices):
            power_values = valid_model_powers[:, irrep_idx]
            power_mask = power_values > 0
            if np.any(power_mask):
                ax.plot(valid_epochs[power_mask], power_values[power_mask], '-', lw=2, color=colors_line[i],
                        label=f'Irrep {irrep_idx} (dim={irreps[irrep_idx].size})')
            if template_power[irrep_idx] > 0:
                ax.axhline(template_power[irrep_idx], linestyle='--', alpha=0.5, color=colors_line[i])
        ax.set_xscale('log')
        ax.set_yscale('log')
        if row_idx == 1:  # Only bottom row shows xlabel
            ax.set_xlabel('Epoch (log scale)')
        ax.set_ylabel('Power (log scale)')
        if row_idx == 0:
            col_title = 'Log-Log Scales'
        else:
            col_title = ''
        ax.set_title(f'{col_title}\n{row_label}', fontsize=12 if row_idx == 0 else 10)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
        # Hide x-axis labels for non-bottom rows (they're shared)
        if row_idx < 1:
            ax.tick_params(labelbottom=False)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n✓ Saved combined plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Map k values to run directories
    run_dirs_dict = {
        4: "runs/20260114_170639",
        5: "runs/20260114_170913",
    }
    
    # Verify all directories exist
    for k, run_dir in run_dirs_dict.items():
        if not os.path.exists(run_dir):
            print(f"Warning: {run_dir} does not exist for k={k}")
    
    save_path = "runs/combined_power_spectrum_k4_k5_2x3.pdf"
    create_combined_plot(run_dirs_dict, save_path)
