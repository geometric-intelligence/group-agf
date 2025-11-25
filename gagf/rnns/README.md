# Group AGF in RNNs

## Models

This codebase supports two model architectures:

### QuadraticRNN
A recurrent neural network that processes sequences sequentially using recurrence:
- Uses temporal ordering of inputs
- Architecture: `h_t = (W_mix h_{t-1} + W_drive x_t)^2` (or multiplicative variant)

### SequentialMLP
A feedforward MLP that processes k inputs by concatenation:
- No temporal ordering (permutation-invariant for commutative groups)
- Architecture: `y = (concat(x_1, ..., x_k) @ W_in)^k @ W_out`
- Uses k-th power activation function (k=2 → quadratic, k=3 → cubic, etc.)

## Optimizers

The codebase supports three optimizer types:

### Adam (`'adam'`)
Standard Adam optimizer. Works with all models.
- Recommended for QuadraticRNN
- Learning rate: typically 1e-3 to 1e-4

### Per-Neuron Scaled SGD (`'per_neuron'`)
SGD with per-neuron learning rate scaling that exploits model homogeneity.
- **Recommended for SequentialMLP**
- Auto-infers degree of homogeneity from model (equals k+1 for SequentialMLP, where k is sequence length)
- Learning rate: typically 1.0 for SequentialMLP
- Exploits the property: scaling all parameters of neuron i by α scales output by α^degree

### Hybrid (`'hybrid'`)  
Combines per-neuron scaled SGD (for W_in, W_drive, W_out) with Adam (for W_mix).
- **Only for QuadraticRNN**
- Best for exploiting both MLP-like and recurrent structure

### Auto Selection (`'auto'`)
Automatically selects the recommended optimizer:
- SequentialMLP → `'per_neuron'`
- QuadraticRNN → `'adam'`

## Training

### Single Run

To train a model on modular addition tasks:
- **1D**: $(C_p)^k$ - Cyclic group of order $p$
- **2D**: $(C_{p1} \times C_{p2})^k$ - Product of two cyclic groups

**Steps:**

1. Edit `gagf/rnns/config.yaml` to specify your experiment.

   **Key configuration parameters:**
   
   | Parameter | Options | Description |
   |-----------|---------|-------------|
   | `data.dimension` | `1` or `2` | Use 1D cyclic group or 2D product group |
   | `data.p` | integer | Cyclic group dimension (1D only) |
   | `data.p1`, `data.p2` | integers | Product group dimensions (2D only) |
   | `data.k` | integer | Sequence length |
   | `data.template_type` | `'mnist'`, `'fourier'`, `'gaussian'` | How to generate template |
   | `model.model_type` | `'QuadraticRNN'`, `'SequentialMLP'` | Architecture choice |
   | `model.transform_type` | `'quadratic'`, `'multiplicative'` | Transform (QuadraticRNN only) |
   | `model.hidden_dim` | integer | Hidden layer size |
   | `training.optimizer` | `'auto'`, `'adam'`, `'per_neuron'`, `'hybrid'` | Optimizer (auto recommended) |
   | `training.learning_rate` | float | Base learning rate |

   **Example configurations:**
   
   ```yaml
   # 1D task with QuadraticRNN
   data:
     dimension: 1
     p: 100
     k: 3
     template_type: 'fourier'
   model:
     model_type: 'QuadraticRNN'
     hidden_dim: 200
   ```
   
   ```yaml
   # 2D task with SequentialMLP
   data:
     dimension: 2
     p1: 10
     p2: 10
     k: 3
     template_type: 'mnist'
   model:
     model_type: 'SequentialMLP'
     hidden_dim: 600  # SequentialMLP typically needs more capacity
   ```

2. Run the training script from the root directory:

```bash
python gagf/rnns/main.py --config gagf/rnns/config.yaml
```

This will train the model and save results to the `runs/` directory.

**Note on visualization:** Currently, detailed analysis plots (Fourier modes, neuron specialization, etc.) are only generated for 2D tasks. For 1D tasks, basic training loss plots are generated.

### Parameter Sweeps

To run parameter sweeps with multiple configurations and seeds:

1. Create or modify a sweep configuration file in `gagf/rnns/sweeps/`. The sweep config should specify:
   - `base_config`: Path to the base configuration file
   - `n_seeds`: Number of random seeds per experiment
   - `experiments`: List of experiments with parameter overrides

Example sweep config (`gagf/rnns/sweeps/example_sweep.yaml`):
```yaml
base_config: "gagf/rnns/config.yaml"
n_seeds: 3

experiments:
  - name: "hidden_dim_32"
    overrides:
      model:
        hidden_dim: 32
  
  - name: "hidden_dim_64"
    overrides:
      model:
        hidden_dim: 64
```

2. Run the sweep from the root directory:

```bash
python gagf/rnns/run_sweep.py --sweep gagf/rnns/sweeps/example_sweep.yaml
```

This will:
- Run each experiment configuration with multiple seeds
- Save results to `sweeps/{sweep_name}_{timestamp}/`
- Generate summary statistics across seeds
- Create experiment and sweep-level summary files

**Multi-GPU Usage:**

To leverage multiple GPUs, specify the GPU ID when launching sweeps:

```bash
# Terminal 1 - Run sweep on GPU 0
python gagf/rnns/run_sweep.py --sweep gagf/rnns/sweeps/sweep1.yaml --gpu 0

# Terminal 2 - Run sweep on GPU 1 simultaneously
python gagf/rnns/run_sweep.py --sweep gagf/rnns/sweeps/sweep2.yaml --gpu 1
```

The `--gpu` flag overrides the device setting in the config file.

### Sweep Results Structure

```
sweeps/
└── example_sweep_20231112_143022/
    ├── sweep_metadata.yaml       # Overall sweep information
    ├── sweep_summary.yaml        # Aggregated results
    ├── configs/                  # Saved configs for each experiment
    ├── hidden_dim_32/
    │   ├── experiment_summary.yaml
    │   ├── seed_0/               # Results for seed 0
    │   ├── seed_1/               # Results for seed 1
    │   └── seed_2/               # Results for seed 2
    └── hidden_dim_64/
        ├── experiment_summary.yaml
        ├── seed_0/
        ├── seed_1/
        └── seed_2/
```

See `gagf/rnns/sweeps/` for more example sweep configurations.