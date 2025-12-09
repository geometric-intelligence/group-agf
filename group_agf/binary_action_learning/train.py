import os
import pickle
import torch


def test_accuracy(model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(inputs.shape[0], -1)  # Flatten input for FC layers
            outputs = model(inputs)
            _, predicted = torch.max(
                outputs, 1
            )  # Get the index of the largest value (class)
            _, true_labels = torch.max(
                labels, 1
            )  # Get the true class from the one-hot encoding
            correct += (predicted == true_labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy


def get_model_save_path(config, checkpoint_epoch):
    """Generate a unique model save path based on the config parameters."""
    if config["group_name"] == "cnxcn":
        model_save_path = (
            f"{config['model_save_dir']}model_"
            f"group_name{config['group_name']}_"
            f"group_size{config['group_size']}_"
            # f"digit{config['mnist_digit']}_"
            f"frac{config['dataset_fraction']}_"
            f"group_size{config['group_size']}_"
            f"init{config['init_scale']}_"
            f"lr{config['lr']}_"
            f"mom{config['mom']}_"
            f"bs{config['batch_size']}_"
            f"checkpoint_epoch{checkpoint_epoch}_"
            f"seed{config['seed']}.pt"
        )
    else:
        model_save_path = (
            f"{config['model_save_dir']}model_"
            f"group_name{config['group_name']}_"
            f"group_size{config['group_size']}_"
            f"init{config['init_scale']}_"
            f"lr{config['lr']}_"
            f"mom{config['mom']}_"
            f"bs{config['batch_size']}_"
            f"epochs{checkpoint_epoch}_"
            f"seed{config['seed']}.pt"
        )

    return model_save_path


def save_param_history(param_history_path, param_history):
    """Save param_history separately for analysis (can be very large)."""
    torch.save({"param_history": param_history}, param_history_path)
    print(
        f"Parameter history saved to {param_history_path} "
        f"(size: {os.path.getsize(param_history_path) / (1024**3):.2f} GB)"
    )

def load_param_history(param_history_path):
    """Load param_history separately for analysis (can be very large)."""
    param_history = torch.load(param_history_path)["param_history"]
    return param_history


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    loss_history,
    accuracy_history,
    param_history=None,
    save_param_history=True,
):
    # Get optimizer state dict and remove model reference from param_groups if present
    # (model reference can't be pickled and will be restored from the model parameter during load)
    optimizer_state = optimizer.state_dict()
    if "param_groups" in optimizer_state:
        # Create a copy without the model reference
        param_groups_clean = []
        for group in optimizer_state["param_groups"]:
            group_clean = {k: v for k, v in group.items() if k != "model"}
            param_groups_clean.append(group_clean)
        optimizer_state_clean = {
            "state": optimizer_state["state"],
            "param_groups": param_groups_clean,
        }
    else:
        optimizer_state_clean = optimizer_state

    # Build checkpoint dict - only include param_history if requested (saves disk space)
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state_clean,
        "epoch": epoch,
        "loss_history": loss_history,
        "accuracy_history": accuracy_history,
    }
    if save_param_history and param_history is not None:
        checkpoint_dict["param_history"] = param_history

    try:
        torch.save(checkpoint_dict, checkpoint_path)
        print(
            f"Training history saved to {checkpoint_path}. You can reload it later with torch.load({checkpoint_path}, map_location='cpu')."
        )
    except RuntimeError as e:
        if "No space left on device" in str(e) or "unexpected pos" in str(e):
            print(f"ERROR: Failed to save checkpoint due to disk space issues: {e}")
            print(f"Checkpoint path: {checkpoint_path}")
            print(
                "Consider cleaning up old checkpoints or using a different save location."
            )
            raise
        else:
            raise


def load_checkpoint(checkpoint_path, model, optimizer=None, map_location="cpu"):
    # Try loading with torch.load first (for .pt files or new .pkl files saved with torch.save)
    # If that fails, try pickle.load for backward compatibility with old .pkl files
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as e:
        # Fallback to pickle for old checkpoints
        print(
            f"Warning: torch.load failed, trying pickle.load for backward compatibility: {e}"
        )

        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        # For PerNeuronScaledSGD, we need to restore the model reference in param_groups
        optimizer_state = checkpoint["optimizer_state_dict"]
        # Restore model reference in param_groups if it was removed during save
        if (
            "param_groups" in optimizer_state
            and len(optimizer_state["param_groups"]) > 0
        ):
            # Check if optimizer expects a model reference (e.g., PerNeuronScaledSGD)
            if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0:
                if "model" in optimizer.param_groups[0]:
                    # Restore model reference before loading state dict
                    for group in optimizer_state["param_groups"]:
                        group["model"] = model
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            print(f"Warning: Could not fully load optimizer state: {e}")
            print("Optimizer will continue with current configuration.")
    print(
        f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', -1)})"
    )
    return checkpoint


def train(
    config,
    model,
    dataloader,
    criterion,
    optimizer,
):
    """Train the model with checkpointing and resume capability.

    Parameters:
    ----------
    config : dict
        Configuration dictionary with training parameters.
    model : torch.nn.Module
        The neural network model to train.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing the training data.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    Returns:
    -------
    loss_history : list
        List of loss values for each epoch.
    accuracy_history : list
        List of accuracy values for each epoch.
    param_history : list
        List of model parameters for each epoch.
    """

    model.train()
    start_epoch = 0
    loss_history = []
    accuracy_history = []
    param_history = []

    if (
        config["resume_from_checkpoint"]
        and config["checkpoint_path"] is not None
        and os.path.isfile(config["checkpoint_path"])
    ):
        print(f"Resuming from checkpoint at {config['checkpoint_path']}.")
        checkpoint = load_checkpoint(config["checkpoint_path"], model, optimizer)
        param_history = load_param_history(config["checkpoint_path"].replace(".pt", "_param_history.pt"))
        start_epoch = checkpoint.get("epoch", 0) + 1
        loss_history = checkpoint.get("loss_history", [])
        accuracy_history = checkpoint.get("accuracy_history", [])
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print(f"Starting training from scratch (no checkpoint to resume). Checkpoint path was: {config['checkpoint_path']}")

    for epoch in range(start_epoch, config["epochs"]):
        running_loss = 0.0
        for (inputs, labels) in dataloader:
            inputs = inputs.view(inputs.shape[0], -1)  # Flatten input for FC layers

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Append the average loss for the epoch to loss_history
        avg_loss = running_loss / len(dataloader)
        # Detect NaN loss early in training and raise an error
        if torch.isnan(torch.tensor(avg_loss)):
            if epoch < 0.75 * config["epochs"]:
                raise RuntimeError(
                    f"NaN loss encountered at epoch {epoch+1} (avg_loss={avg_loss})."
                )
        loss_history.append(avg_loss)

        # Append accuracy
        model.eval()
        accuracy = test_accuracy(model, dataloader)
        accuracy_history.append(accuracy)
        model.train()

        # Save current model parameters
        current_params = {
            "U": model.U.detach().cpu().clone(),
            "V": model.V.detach().cpu().clone(),
            "W": model.W.detach().cpu().clone(),
        }
        param_history.append(current_params)

        # Print verbose information every `verbose_interval` epochs
        if (epoch + 1) % config["verbose_interval"] == 0:
            print(
                f"Epoch {epoch+1}/{config["epochs"]}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )

        # Save checkpoint if at checkpoint interval or at the end of the training
        if (epoch + 1) % config["checkpoint_interval"] == 0 or (epoch + 1) == config[
            "epochs"
        ]:
            checkpoint_path = get_model_save_path(
                config, checkpoint_epoch=(epoch + 1)
            )
            # Only save param_history in the final checkpoint to save disk space
            # (param_history can be very large as it stores all parameters for every epoch)
            is_final_checkpoint = (epoch + 1) == config["epochs"]
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                loss_history,
                accuracy_history,
                param_history,
                save_param_history=is_final_checkpoint,
            )
            
            # Save param_history separately only at the end of training (it can be very large)
            if (epoch + 1) == config["epochs"]:
                param_history_path = checkpoint_path.replace(".pt", "_param_history.pt")
                save_param_history(param_history_path, param_history)

    return (
        loss_history,
        accuracy_history,
        param_history,
    )  # Return loss history for plotting
