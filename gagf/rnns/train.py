"""Training function for Quadratic RNN with sequential inputs."""

from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from typing import Optional


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int = 2000,
    verbose_interval: int = 100,
    grad_clip: Optional[float] = None,
    eval_dataloader: Optional[DataLoader] = None,
    save_param_interval: Optional[int] = None,  # NEW parameter
) -> tuple[list[float], list[float], list[float], list[dict[str, torch.Tensor]], list[int]]:
    """
    Train a Quadratic RNN with sequential inputs (offline/epoch-based).
    
    Returns:
        tuple: train_loss_history, val_loss_history,
               parameter_history, param_save_epochs (indices where params were saved)
    """
    train_loss_history, val_loss_history, param_history = [], [], []
    param_save_epochs = []

    # --- BEFORE TRAINING (epoch 0) ---
    model.eval()
    with torch.no_grad():
        if eval_dataloader is not None:
            X_eval, Y_eval = next(iter(eval_dataloader))
            out = model(X_eval)
            val_loss0 = criterion(out, Y_eval).item()
        else:
            X_eval, Y_eval = next(iter(dataloader))
            out = model(X_eval)
            val_loss0 = criterion(out, Y_eval).item()

        snap0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    train_loss_history.append(val_loss0)
    val_loss_history.append(val_loss0)
    param_history.append(snap0)
    param_save_epochs.append(0)

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
        train_loss_history.append(avg_loss)

        model.eval()
        if eval_dataloader is not None:
            X_eval, Y_eval = next(iter(eval_dataloader))
            out = model(X_eval)
            val_loss = criterion(out, Y_eval).item()
        else:
            X_eval, Y_eval = next(iter(dataloader))
            out = model(X_eval)
            val_loss = criterion(out, Y_eval).item()

        val_loss_history.append(val_loss)

        # Only save parameters at intervals or at the end
        should_save = (save_param_interval is None or 
                      epoch % save_param_interval == 0 or 
                      epoch == epochs)
        
        if should_save:
            with torch.no_grad():
                snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            param_history.append(snap)
            param_save_epochs.append(epoch)

        if epoch % verbose_interval == 0:
            print(f"[RNN] Epoch {epoch}/{epochs} | train_loss {avg_loss:.6f} | val_loss {val_loss:.6f}")

    return train_loss_history, val_loss_history, param_history, param_save_epochs

def train_online(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_steps: int = 10000,
    verbose_interval: int = 100,
    grad_clip: Optional[float] = None,
    eval_dataloader: Optional[DataLoader] = None,
    save_param_interval: Optional[int] = None,  # NEW parameter
) -> tuple[list[float], list[float], list[float], list[dict[str, torch.Tensor]], list[int]]:
    """
    Train with online data generation (step-based instead of epoch-based).
    
    Args:
        ...
        save_param_interval: If provided, save params every N steps. 
                            If None, save at every step (memory intensive!)
    
    Returns:
        tuple: train_loss_history, val_loss_history,
               parameter_history, param_save_steps (indices where params were saved)
    """
    train_loss_history, val_loss_history, param_history = [], [], []
    param_save_steps = []  # Track which steps have saved params
    
    # Initial evaluation (step 0)
    model.eval()
    with torch.no_grad():
        if eval_dataloader is not None:
            X_eval, Y_eval = next(iter(eval_dataloader))
            out = model(X_eval)
            val_loss0 = criterion(out, Y_eval).item()
        else:
            X_batch, Y_batch = next(iter(dataloader))
            out = model(X_batch)
            val_loss0 = criterion(out, Y_batch).item()
        
        snap0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    
    train_loss_history.append(val_loss0)
    val_loss_history.append(val_loss0)
    param_history.append(snap0)
    param_save_steps.append(0)
    
    # Training loop
    model.train()
    data_iter = iter(dataloader)
    
    for step in range(1, num_steps + 1):
        # Get fresh batch
        X_batch, Y_batch = next(data_iter)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Record training loss
        train_loss_history.append(loss.item())
        
        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            if eval_dataloader is not None:
                X_eval, Y_eval = next(iter(eval_dataloader))
                out = model(X_eval)
                val_loss = criterion(out, Y_eval).item()
            else:
                X_eval, Y_eval = next(data_iter)
                out = model(X_eval)
                val_loss = criterion(out, Y_eval).item()
            
            val_loss_history.append(val_loss)
            
            # Only save parameters at specified intervals or at the end
            should_save = (save_param_interval is None or 
                          step % save_param_interval == 0 or 
                          step == num_steps)
            
            if should_save:
                snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                param_history.append(snap)
                param_save_steps.append(step)
        
        model.train()
        
        if step % verbose_interval == 0:
            print(f"[RNN] Step {step}/{num_steps} | train_loss {loss.item():.6f} | val_loss {val_loss:.6f}")
    
    return train_loss_history, val_loss_history, param_history, param_save_steps