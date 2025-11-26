import torch
import pickle
import os


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
    if config["group_name"] == "znz_znz":
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
            # f"freq{config['frequencies_to_learn']}_"
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
            # f"run_start{config['run_start_time']}.pkl"
        )

    return model_save_path


def save_checkpoint(checkpoint_path, model, optimizer, epoch, loss_history, accuracy_history, param_history):
    # Get optimizer state dict and remove model reference from param_groups if present
    # (model reference can't be pickled and will be restored from the model parameter during load)
    optimizer_state = optimizer.state_dict()
    if 'param_groups' in optimizer_state:
        # Create a copy without the model reference
        param_groups_clean = []
        for group in optimizer_state['param_groups']:
            group_clean = {k: v for k, v in group.items() if k != 'model'}
            param_groups_clean.append(group_clean)
        optimizer_state_clean = {
            'state': optimizer_state['state'],
            'param_groups': param_groups_clean
        }
    else:
        optimizer_state_clean = optimizer_state
    
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state_clean,
            "epoch": epoch,
            "loss_history": loss_history,
            "accuracy_history": accuracy_history,
            "param_history": param_history,
        },
        checkpoint_path,
    )
    print(
        f"Training history saved to {checkpoint_path}. You can reload it later with torch.load({checkpoint_path}, map_location='cpu')."
    )

def load_checkpoint(checkpoint_path, model, optimizer=None, map_location="cpu"):
    # Try loading with torch.load first (for .pt files or new .pkl files saved with torch.save)
    # If that fails, try pickle.load for backward compatibility with old .pkl files
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        # Fallback to pickle for old checkpoints
        print(f"Warning: torch.load failed, trying pickle.load for backward compatibility: {e}")
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        # For PerNeuronScaledSGD, we need to restore the model reference in param_groups
        optimizer_state = checkpoint['optimizer_state_dict']
        # Restore model reference in param_groups if it was removed during save
        if 'param_groups' in optimizer_state and len(optimizer_state['param_groups']) > 0:
            # Check if optimizer expects a model reference (e.g., PerNeuronScaledSGD)
            if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                if 'model' in optimizer.param_groups[0]:
                    # Restore model reference before loading state dict
                    for group in optimizer_state['param_groups']:
                        group['model'] = model
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            print(f"Warning: Could not fully load optimizer state: {e}")
            print("Optimizer will continue with current configuration.")
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', -1)})")
    return checkpoint


def train(
    config,
    model,
    dataloader,
    criterion,
    optimizer,
):
    model.train()  # Set the model to training mode
    start_epoch = 0
    loss_history = []
    accuracy_history = []
    param_history = []

    # Resume from checkpoint if requested
    if config["resume_from_checkpoint"] and config["checkpoint_path"] is not None and os.path.isfile(config["checkpoint_path"]):
        print(f"Resuming from checkpoint at {config['checkpoint_path']}.")
        checkpoint = load_checkpoint(config["checkpoint_path"], model, optimizer)
        start_epoch = checkpoint.get("epoch", 0) + 1
        loss_history = checkpoint.get("loss_history", [])
        accuracy_history = checkpoint.get("accuracy_history", [])
        param_history = checkpoint.get("param_history", [])
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting training from scratch (no checkpoint to resume).")

    for epoch in range(start_epoch, config["epochs"]):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(inputs.shape[0], -1)  # Flatten input for FC layers

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Append the average loss for the epoch to loss_history
        avg_loss = running_loss / len(dataloader)
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
        if ((epoch + 1) % config["checkpoint_interval"] == 0 or (epoch + 1) == config["epochs"]):
            checkpoint_path = get_model_save_path(config, checkpoint_epoch=(epoch + 1))
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                loss_history,
                accuracy_history,
                param_history,
            )

    return (
        loss_history,
        accuracy_history,
        param_history,
    )  # Return loss history for plotting


