# Includes training functions needed for TreeLearn

import time
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
from collections import defaultdict
#from fastprogress.fastprogress import master_bar, progress_bar

from .TreeLearn import TreeLearn, LOSS_MULTIPLIER_SEMANTIC
from Modules.Utils import cuda_cast, EarlyStopper


def train(model, train_loader, optimizer, scheduler, scaler, epoch):
    """
    Perform one epoch of training.
    
    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for training.
        loss_fn: Loss function.
    
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    start = time.time()
    losses_dict = defaultdict(list)

    for batch in tqdm(train_loader, desc="Training", leave=False):

        scheduler.step(epoch)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):

            # forward
            loss, loss_dict = model(batch, return_loss=True)
            for key, value in loss_dict.items():
                losses_dict[key].append(value.detach().cpu().item())

        # backward
        scaler.scale(loss).backward()
        # if config.grad_norm_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), True, norm_type=2)
        scaler.step(optimizer)
        scaler.update()

        mb.child.comment = f'Training'

    loss_off = np.mean(loss_dict['offset_loss'])
    loss_sem = np.mean(loss_dict['semantic_loss'])
    loss_total = loss_off + LOSS_MULTIPLIER_SEMANTIC * loss_sem

    epoch_time = time.time() - start
    lr = optimizer.param_groups[0]['lr']

    return loss_total, loss_off, loss_sem


def validate(model, val_loader, epoch):
    """
    Perform validation and compute average loss.
    
    Args:
        model: The model to validate.
        val_loader: DataLoader for validation data.
        loss_fn: Loss function.
    
    Returns:
        float: Average validation loss.
    """
    model.eval()
    start = time.time()
    losses_dict = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):

            with torch.cuda.amp.autocast(enabled=True):

                # forward
                loss, loss_dict = model(batch, return_loss=True)
                for key, value in loss_dict.items():
                    losses_dict[key].append(value.detach().cpu().item())

            mb.child.comment = f'Validating'

    loss_off = np.mean(loss_dict['offset_loss'])
    loss_sem = np.mean(loss_dict['semantic_loss'])
    loss_total = loss_off + LOSS_MULTIPLIER_SEMANTIC * loss_sem

    return loss_total, loss_off, loss_sem


def run_training(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    scheduler=None,
    early_stopper=None,
):
    """
    Train a model with optional learning rate scheduling and early stopping.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for training.
        loss_fn: Loss function.
        epochs: Number of epochs to train.
        scheduler: Optional learning rate scheduler.
        early_stopper: Optional EarlyStopping instance.
    """
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Training phase
        train_loss_total, train_loss_off, train_loss_sem = train(model, train_loader, optimizer, scheduler, scaler, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss_total:.4f}")

        # Validation phase
        val_loss_total, val_loss_off, val_loss_sem = validate(model, val_loader, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss_total:.4f}")

        # Step scheduler if provided
        # if scheduler:
        #     scheduler.step(val_loss)

        # Early stopping check
        if early_stopper:
            early_stopper(val_loss_total)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
