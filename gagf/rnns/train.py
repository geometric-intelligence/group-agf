"""Training function for Quadratic RNN with sequential inputs."""

from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from typing import Optional

# TODO: make sure this is correct? Does it vary with batch size and other hyperparameters? If so, why?
def test_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            outputs = model(X_batch)  # (B, p)
            pred = outputs.argmax(dim=1)
            true = Y_batch.argmax(dim=1)
            correct += (pred == true).sum().item()
            total += Y_batch.size(0)
    return 100.0 * correct / total


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int = 2000,
    verbose_interval: int = 100,
    grad_clip: Optional[float] = None,
) -> tuple[list[float], list[float], list[dict[str, torch.Tensor]]]:
    """
    Train a Quadratic RNN with sequential inputs.
    
    Args:
        model: nn.Module, the model to train
        dataloader: DataLoader, the data loader
        criterion: nn.Module, the loss function
        optimizer: optim.Optimizer, the optimizer
        epochs: int, the number of epochs to train for
        verbose_interval: int, the interval at which to print verbose information
        grad_clip: Optional[float], if not None, the gradient clip value
    Returns:
        tuple[list[float], list[float], list[dict[str, torch.Tensor]]]: the loss history, accuracy history, and parameter history
    """
    loss_history, acc_history, param_history = [], [], []

    # --- BEFORE TRAINING (epoch 0) ---
    model.eval()
    with torch.no_grad():
        # avg loss over data
        total = 0.0
        for Xb, Yb in dataloader:
            out = model(Xb)
            total += criterion(out, Yb).item()
        loss0 = total / len(dataloader)
        acc0 = test_accuracy(model, dataloader)
        snap0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    loss_history.append(loss0)
    acc_history.append(acc0)
    param_history.append(snap0)

    # --- TRAINING LOOP (epochs 1..epochs) ---
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item()

        avg_loss = running / len(dataloader)
        loss_history.append(avg_loss)

        model.eval()
        acc = test_accuracy(model, dataloader)
        acc_history.append(acc)

        with torch.no_grad():
            snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        param_history.append(snap)

        if epoch % verbose_interval == 0:
            print(f"[RNN] Epoch {epoch}/{epochs} | loss {avg_loss:.6f} | acc {acc:.2f}%")

    return loss_history, acc_history, param_history