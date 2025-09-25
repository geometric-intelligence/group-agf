import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

def test_accuracy(model, dataloader):
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(inputs.shape[0], -1)  # Flatten input for FC layers
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the largest value (class)
            _, true_labels = torch.max(labels, 1)  # Get the true class from the one-hot encoding
            correct += (predicted == true_labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    return accuracy

def train(model, dataloader, criterion, optimizer, epochs=100, verbose_interval=1, neurons_to_plot=[0,1,2, 3,4,5]):
    model.train()  # Set the model to training mode
    loss_history = []  # List to store loss values
    accuracy_history = []
    param_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Check if labels are all zeros or if labels are not being set correctly
            if torch.all(labels == 0):
                print(f"Warning: All labels are zero in batch {batch_idx} of epoch {epoch}")
            if torch.isnan(inputs).any() or torch.isnan(labels).any():
                print(f"NaN detected in inputs or labels in batch {batch_idx} of epoch {epoch}")

            inputs = inputs.view(inputs.shape[0], -1)  # Flatten input for FC layers

            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Forward pass

            # Check if outputs are all zeros or constant
            if torch.all(outputs == 0):
                print(f"Warning: Model outputs are all zero in batch {batch_idx} of epoch {epoch}")

            loss = criterion(outputs, labels)  # Compute loss

            # Check if loss is zero from the start
            if epoch == 0 and batch_idx == 0 and loss.item() == 0.0:
                print("Warning: Loss is zero at the very first batch. Check if criterion and labels are set up correctly.")

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
            "W": model.W.detach().cpu().clone()
        }
        param_history.append(current_params)

        # Print verbose information every `verbose_interval` epochs
        if (epoch + 1) % verbose_interval == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Plot results every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
            model.eval()
            with torch.no_grad():
                # Get one random example from the dataset
                idx = np.random.randint(len(dataloader.dataset))
                x, y = dataloader.dataset[idx]
                # If x is not already a tensor, convert it
                if not torch.is_tensor(x):
                    x = torch.tensor(x, dtype=torch.float32)
                if not torch.is_tensor(y):
                    y = torch.tensor(y, dtype=torch.float32)
                # Move to device if needed
                x_input = x.view(1, -1)
                if hasattr(model, 'device'):
                    x_input = x_input.to(model.device)
                elif next(model.parameters()).is_cuda:
                    x_input = x_input.cuda()
                output = model(x_input)
                output_np = output.cpu().numpy().squeeze()
                target_np = y.cpu().numpy().squeeze() if hasattr(y, 'cpu') else y.numpy().squeeze()

                # Try to infer image size if possible
                # Ensure x, output, and target are on CPU and numpy arrays for plotting
                if torch.is_tensor(x):
                    x_np = x.detach().cpu().numpy()
                else:
                    x_np = np.array(x)
                if torch.is_tensor(output):
                    output_np = output.detach().cpu().numpy().squeeze()
                # output_np already defined above, but ensure it's on CPU and numpy
                if torch.is_tensor(y):
                    target_np = y.detach().cpu().numpy().squeeze()
                else:
                    target_np = np.array(y).squeeze()

                # Infer image size
                image_size = int(np.sqrt(x_np.shape[-1] // 2)) if x_np.shape[-1] % 2 == 0 else int(np.sqrt(x_np.shape[-1]))

                fig, axs = plt.subplots(1, 4, figsize=(15, 3), sharey=True)
                axs[0].imshow(x_np[:image_size*image_size].reshape(image_size, image_size))
                axs[0].set_title('Input 1')
                axs[1].imshow(x_np[image_size*image_size:].reshape(image_size, image_size))
                axs[1].set_title('Input 2')
                axs[2].imshow(output_np.reshape(image_size, image_size))
                axs[2].set_title('Output')
                axs[3].imshow(target_np.reshape(image_size, image_size))
                axs[3].set_title('Target')
                plt.tight_layout()
                plt.savefig(f"figs/prediction_fig_epoch_{epoch+1}.png", bbox_inches='tight')
                plt.close(fig)

                plot_neuron_weights(model, neurons_to_plot, p=image_size, save_path=f"figs/weights_epoch_{epoch+1}.png", show=False)

    
             

    return loss_history, accuracy_history, param_history # Return loss history for plotting