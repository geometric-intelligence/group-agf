import numpy as np
import torch

import group_agf.binary_action_learning.templates as templates

def load_dataset(config):
    """Load dataset based on configuration."""

    if config["group_name"] == "znz_znz":
        # template = mnist_template(config["image_length"], digit=config["mnist_digit"])
        template = templates.fixed_znz_znz_template(
            config["image_length"], config["fourier_coef_diag_values"]
        )
        X, Y = modular_addition_dataset_2d(template)

    else:
        template = templates.fixed_group_template(
            config["group"], config["fourier_coef_diag_values"]
        )
        X, Y = group_dataset(config["group"], template)

        print(f"dataset_fraction: {config['dataset_fraction']}")
        
    if config["dataset_fraction"] != 1.0:
        assert 0 < config["dataset_fraction"] <= 1.0, "fraction must be in (0, 1]"
        # Sample a subset of the dataset according to the specified fraction
        N = X.shape[0]
        n_sample = int(np.ceil(N * config["dataset_fraction"]))
        rng = np.random.default_rng(config["seed"])
        indices = rng.choice(
            N, size=n_sample, replace=False
        )  # indices of the sampled subset
        X = X[indices]
        Y = Y[indices]

    return X, Y, template


def group_dataset(group, template):
    """Generate a dataset of group elements acting on the template.

    Using the regular representation.

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    template : np.ndarray, shape=[group.order()]
        The template to generate the dataset from.

    Returns
    -------
    X : np.ndarray, shape=[group.order()**2, 2, group.order()]
        The dataset of group elements acting on the template.
    Y : np.ndarray, shape=[group.order()**2, group.order()]
        The dataset of group elements acting on the template.
    """

    # Initialize data arrays
    group_order = group.order()
    assert (
        len(template) == group_order
    ), "template must have the same length as the group order"
    n_samples = group_order**2
    X = np.zeros((n_samples, 2, group_order))
    Y = np.zeros((n_samples, group_order))
    regular_rep = group.representations["regular"]

    idx = 0
    for g1 in group.elements:
        for g2 in group.elements:
            g1_rep = regular_rep(g1)
            g2_rep = regular_rep(g2)
            g12_rep = g1_rep @ g2_rep

            X[idx, 0, :] = g1_rep @ template
            X[idx, 1, :] = g2_rep @ template
            Y[idx, :] = g12_rep @ template
            idx += 1

    return X, Y


def modular_addition_dataset_2d(template):
    """Generate a dataset for the 2D modular addition operation.

    General idea: We are generating a dataset where each sample consists of
    two inputs (a*template and b*template) and an output (a*b)*template,
    where $a, b \in Z/pZ x Z/pZ$. The template is a flattened 2D array
    representing the modular addition operation in a 2D space.

    Each element $X_i$ will contain the template with a different $a_i$, $b_i$, and
    in fact $X$ contains the template at all possible $a$, $b$ shifts.
    The output $Y_i$ will contain the template shifted by $a_i*b_i$.
    * refers to the composition of two group actions (but by an abuse of notation,
    also refers to group action on the template.)

    Parameters
    ----------
    template : np.ndarray
        A flattened 2D square image of shape (image_length*image_length,).
    
    Returns
    -------
    X : np.ndarray
        Input data of shape (p^4, 2, p*p).
        2 inputs (a and b), each with shape (p*p,).
         is the total number of combinations of shifted a's and b's.
    Y : np.ndarray
        Output data of shape (p^4, p*p), where each sample is the result of the modular addition.
    """
    image_length = int(np.sqrt(len(template)))
    # Initialize data arrays
    X = np.zeros((image_length**4, 2, image_length * image_length))
    Y = np.zeros((image_length**4, image_length * image_length))  
    translations = np.zeros((image_length**4, 3, 2), dtype=int)

    # Generate the dataset
    idx = 0
    template_2d = template.reshape((image_length, image_length))
    for a_x in range(image_length):
        for a_y in range(image_length):
            for b_x in range(image_length):
                for b_y in range(image_length):
                    q_x = (a_x + b_x) % image_length
                    q_y = (a_y + b_y) % image_length
                    X[idx, 0, :] = np.roll(
                        np.roll(template_2d, a_x, axis=0), a_y, axis=1
                    ).flatten()
                    X[idx, 1, :] = np.roll(
                        np.roll(template_2d, b_x, axis=0), b_y, axis=1
                    ).flatten()
                    Y[idx, :] = np.roll(
                        np.roll(template_2d, q_x, axis=0), q_y, axis=1
                    ).flatten()
                    translations[idx, 0, :] = (a_x, a_y)
                    translations[idx, 1, :] = (b_x, b_y)
                    translations[idx, 2, :] = (q_x, q_y)
                    idx += 1

    return X, Y


def move_dataset_to_device_and_flatten(X, Y, device=None):
    """Move dataset tensors to available or specified device.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (num_samples, 2, p*p)
    Y : np.ndarray
        Target data of shape (num_samples, p*p)
    device : torch.device, optional
        Device to move tensors to. If None, automatically choose GPU if available.

    Returns
    -------
    X : torch.Tensor
        Input data tensor on specified device, flattened to (num_samples, 2*p*p)
    Y : torch.Tensor
        Target data tensor on specified device, flattened to (num_samples, p*p)
    """
    # Reshape X to (num_samples, 2*num_data_features), where num_data_features is inferred from len(X[0][0])
    num_data_features = len(X[0][0])
    X_flat = X.reshape(X.shape[0], 2 * num_data_features)
    Y_flat = Y.reshape(Y.shape[0], num_data_features)
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_flat, dtype=torch.float32)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available. Using CUDA.")
        else:
            device = torch.device("cpu")
            print("GPU is not available. Using CPU.")

    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    return X_tensor, Y_tensor, device
