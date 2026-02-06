# group-agf

[![CI](https://github.com/geometric-intelligence/group-agf/actions/workflows/ci.yml/badge.svg)](https://github.com/geometric-intelligence/group-agf/actions/workflows/ci.yml)

**Group Alternating Gradient Flows** -- training neural networks to learn group composition on finite groups, with theoretical analysis via group Fourier transforms and power spectra.

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)
- `gfortran` (Linux only, required by some numerical dependencies)

### Setup

```bash
# Linux only: install gfortran
sudo apt install -y gfortran

# Create and activate the conda environment
conda env create -f conda.yaml
conda activate group-agf

# Install all Python dependencies (pinned versions from poetry.lock)
poetry install
```

## Repository Structure

```
group-agf/
├── src/                          # Main source code
│   ├── main.py                   # Training entry point (CLI)
│   ├── model.py                  # Neural network architectures
│   ├── optimizer.py              # Custom optimizers
│   ├── dataset.py                # Dataset generation and loading
│   ├── template.py               # Template construction functions
│   ├── fourier.py                # Group Fourier transforms
│   ├── power.py                  # Power spectrum computation
│   ├── viz.py                    # Plotting and visualization
│   ├── train.py                  # Training loops (offline and online)
│   ├── run_sweep.py              # Parameter sweep runner
│   ├── config.yaml               # Default configuration
│   ├── config_c10.yaml           # Cyclic group C10
│   ├── config_c4x4.yaml          # Product group C4 x C4
│   ├── config_d3.yaml            # Dihedral group D3
│   ├── config_octahedral.yaml    # Octahedral group
│   ├── config_a5.yaml            # Icosahedral group (A5)
│   └── sweep_configs/            # Sweep configuration examples
├── test/                         # Unit and integration tests
├── notebooks/                    # Jupyter notebooks for exploration
├── pyproject.toml                # Project metadata and dependencies
├── poetry.lock                   # Pinned dependency versions
├── conda.yaml                    # Conda environment specification
└── .pre-commit-config.yaml       # Pre-commit hooks
```

## Modules

### `model.py` -- Neural Network Architectures

Three architectures for learning group composition:

| Model | Description | Input |
|-------|-------------|-------|
| **TwoLayerNet** | Two-layer feedforward network with configurable nonlinearity (square, relu, tanh, gelu) | Flattened binary pair `(N, 2 * group_size)` |
| **QuadraticRNN** | Recurrent network: `h_t = (W_mix h_{t-1} + W_drive x_t)^2` | Sequence `(N, k, p)` |
| **SequentialMLP** | Feedforward MLP with k-th power activation, permutation-invariant for commutative groups | Sequence `(N, k, p)` |

### `optimizer.py` -- Custom Optimizers

| Optimizer | Description | Recommended for |
|-----------|-------------|-----------------|
| **PerNeuronScaledSGD** | SGD with per-neuron learning rate scaling exploiting model homogeneity | SequentialMLP, TwoLayerNet |
| **HybridRNNOptimizer** | Scaled SGD for MLP weights + Adam for recurrent weights | QuadraticRNN |
| Adam (PyTorch built-in) | Standard Adam | QuadraticRNN |

### `dataset.py` -- Data Generation

- **Online datasets**: `OnlineModularAdditionDataset1D`, `OnlineModularAdditionDataset2D` -- generate samples on-the-fly (GPU-accelerated)
- **Offline builders**: `build_modular_addition_sequence_dataset_1d`, `_2d`, `_D3`, `_generic`
- **Group datasets**: `cn_dataset`, `cnxcn_dataset`, `group_dataset` -- full group multiplication tables for TwoLayerNet

### `template.py` -- Template Construction

Functions for building target templates that define the composition task:

- **Group templates**: `one_hot`, `fixed_cn`, `fixed_cnxcn`, `fixed_group`
- **1D synthetic**: `fourier_1d`, `gaussian_1d`, `onehot_1d`
- **2D synthetic**: `gaussian_mixture_2d`, `unique_freqs_2d`, `fixed_2d`, `hexagon_tie_2d`, `ring_isotropic_2d`, `gaussian_2d`
- **MNIST-based**: `mnist`, `mnist_1d`, `mnist_2d`

### `fourier.py` -- Group Fourier Transforms

- `group_fourier(group, template)` -- compute Fourier coefficients using irreducible representations
- `group_fourier_inverse(group, fourier_coefs)` -- reconstruct template from Fourier coefficients

### `power.py` -- Power Spectrum Analysis

- `GroupPower` -- power spectrum of a template over any `escnn` group
- `CyclicPower` -- specialized for cyclic groups via FFT
- `model_power_over_time` -- track how the model's learned power spectrum evolves during training
- `theoretical_loss_levels_1d`, `_2d` -- predict staircase loss plateaus from template power

### `viz.py` -- Visualization

Plotting functions for training analysis: `plot_train_loss_with_theory`, `plot_predictions_1d`, `plot_predictions_2d`, `plot_predictions_group`, `plot_power_1d`, `plot_power_group`, `plot_wmix_structure`, `plot_irreps`, and more.

### `train.py` -- Training Loops

- `train(model, loader, criterion, optimizer, ...)` -- epoch-based offline training
- `train_online(model, loader, criterion, optimizer, ...)` -- step-based online training

## Supported Groups

The repository includes preconfigured experiments for five groups:

| Group | Config file | Order | Model |
|-------|-------------|-------|-------|
| Cyclic C10 | `src/config_c10.yaml` | 10 | QuadraticRNN |
| Product C4 x C4 | `src/config_c4x4.yaml` | 16 | QuadraticRNN |
| Dihedral D3 | `src/config_d3.yaml` | 6 | TwoLayerNet |
| Octahedral | `src/config_octahedral.yaml` | 24 | TwoLayerNet |
| Icosahedral (A5) | `src/config_a5.yaml` | 60 | TwoLayerNet |

## Usage

### Single Run

Train a model on a specific group:

```bash
python src/main.py --config src/config_d3.yaml
```

Results (loss curves, predictions, power spectra) are saved to a timestamped directory under `runs/`.

### Parameter Sweeps

Run experiments across multiple configurations and random seeds:

```bash
python src/run_sweep.py --sweep src/sweep_configs/example_sweep.yaml
```

Multi-GPU support:

```bash
# Auto-detect and use all available GPUs
python src/run_sweep.py --sweep src/sweep_configs/example_sweep.yaml --gpus auto

# Use specific GPUs
python src/run_sweep.py --sweep src/sweep_configs/example_sweep.yaml --gpus 0,1,2,3
```

Sweep results are saved to `sweeps/{sweep_name}_{timestamp}/` with per-seed results and aggregated summaries.

## Configuration

Key parameters in the YAML config files:

| Parameter | Options | Description |
|-----------|---------|-------------|
| `data.group_name` | `cn`, `cnxcn`, `dihedral`, `octahedral`, `A5` | Group to learn |
| `data.k` | integer | Number of elements to compose |
| `data.template_type` | `mnist`, `fourier`, `gaussian`, `onehot`, `custom_fourier` | Template generation method |
| `model.model_type` | `QuadraticRNN`, `SequentialMLP`, `TwoLayerNet` | Architecture |
| `model.hidden_dim` | integer | Hidden layer size |
| `model.init_scale` | float | Weight initialization scale |
| `training.optimizer` | `auto`, `adam`, `per_neuron`, `hybrid` | Optimizer (`auto` recommended) |
| `training.learning_rate` | float | Base learning rate |
| `training.mode` | `online`, `offline` | Training mode |
| `training.epochs` | integer | Number of epochs (offline mode) |

Example -- D3 with custom Fourier template:

```yaml
data:
  group_name: dihedral
  group_n: 3
  k: 2
  template_type: custom_fourier
  powers: [0.0, 30.0, 3000.0]

model:
  model_type: TwoLayerNet
  hidden_dim: 180
  init_scale: 0.001

training:
  optimizer: per_neuron
  learning_rate: 0.01
  mode: offline
  epochs: 2000
```

## Testing

Run the unit tests:

```bash
pytest test/ --ignore=test/test_notebooks.py -v
```

Run notebook smoke tests (requires `MAIN_TEST_MODE` flag for reduced parameters):

```bash
MAIN_TEST_MODE=1 pytest test/test_main.py -v
```

## Development

### Pre-commit Hooks

Install and run the pre-commit hooks (ruff linting and formatting, trailing whitespace, etc.):

```bash
pre-commit install
pre-commit run --all-files
```

### Linting

```bash
ruff check .
ruff format --check .
```
