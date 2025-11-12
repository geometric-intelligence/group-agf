# Group AGF in RNNs


## Training

### Single Run

To train a Quadratic RNN on the sequential 2D modular addition task, $(C_n \times C_n \to C_n)^k$, where $k$ is the sequence length, you should:

1. Modify the config file `gagf/rnns/config.yaml`.

2. Run the `main.py` script from the root directory:

```bash
python gagf/rnns/main.py --config gagf/rnns/config.yaml
```

This will train the model and save the results to the `runs` directory.

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