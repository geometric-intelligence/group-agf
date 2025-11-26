import torch
import pickle


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

# TODO: Make loss plot every loss_log_epoch to be able to monitor.
# TODO: Add checkpointing every checkpoint_interval to be able to resume training.
def train(
    model,
    dataloader,
    criterion,
    optimizer,
    epochs=100,
    verbose_interval=1,
    model_save_path=None,
):
    model.train()  # Set the model to training mode
    loss_history = []  # List to store loss values
    accuracy_history = []
    param_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(inputs.shape[0], -1)  # Flatten input for FC layers

            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        # Append the average loss for the epoch to loss_history
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)

        # Append the accuracy
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
        if (epoch + 1) % verbose_interval == 0:
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )

    # Save the model if a path is provided
    if model_save_path is not None:
        with open(model_save_path, "wb") as f:
            pickle.dump(
                {
                    "loss_history": loss_history,
                    "accuracy_history": accuracy_history,
                    "param_history": param_history,
                },
                f,
            )

        print(
            f"Training history saved to {model_save_path}. You can reload it later with pickle.load(open({model_save_path}, 'rb'))."
        )

    return (
        loss_history,
        accuracy_history,
        param_history,
    )  # Return loss history for plotting
