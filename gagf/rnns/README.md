# Group AGF in RNNs


## Training

To train a Quadratic RNN on the sequential 2D modular addition task, $(C_n \times C_n \to C_n)^k$, where $k$ is the sequence length, you should:

1. Modify the config file `gagf/rnns/config.yaml`.

2. Run the `main.py` script from the root directory:

```
python gagf/rnns/main.py
```

This will train the model and save the results to the `runs` directory.