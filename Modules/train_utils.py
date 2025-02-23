# Includes training functions needed for TreeLearn

import time
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
from collections import defaultdict
from fastprogress.fastprogress import master_bar, progress_bar
import fastprogress
import logging

from Modules.Utils import cuda_cast, EarlyStopper


def train(model, train_loader, optimizer, scheduler, scaler, epoch, mb, raster_hierarchical):
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

    pb = progress_bar(train_loader, parent=mb)  # Progress bar for training
    pb.comment = f"Training"  # Comment for the progress bar

    i = 0
    for batch in pb:

        scheduler.step(epoch)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=True):

            # forward
            if raster_hierarchical:
                loss, loss_dict = model.forward_hierarchical(batch, return_loss=True)
            else:
                loss, loss_dict = model(batch, return_loss=True)
            for key, value in loss_dict.items():
                losses_dict[key].append(value.detach().cpu().item())

        # backward
        scaler.scale(loss*50).backward()
        # if config.grad_norm_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), True, norm_type=2)
        scaler.step(optimizer)
        scaler.update()

        i+=1
        if i % 1 == 0:  # Adjust frequency based on your observation
            torch.cuda.empty_cache()
            i=0
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))
            # print(f"Active memory: {torch.cuda.memory_allocated()/1e6:.2f} MB")
            # print(f"Reserved memory: {torch.cuda.memory_reserved()/1e6:.2f} MB")

    loss_off = np.mean(losses_dict['offset_loss'])
    loss_sem = np.mean(losses_dict['semantic_loss'])
    loss_total = loss_off + loss_sem

    epoch_time = time.time() - start
    lr = optimizer.param_groups[0]['lr']

    return loss_total, loss_off, loss_sem


def validate(model, val_loader, epoch, mb, raster_hierarchical):
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

    pb = progress_bar(val_loader, parent=mb)  # Progress bar for validation
    pb.comment = f"Validation"

    with torch.no_grad():
        i=0
        for batch in pb:

            with torch.amp.autocast('cuda', enabled=True):

                # forward
                if raster_hierarchical:
                    loss, loss_dict = model.forward_hierarchical(batch, return_loss=True)
                else:
                    loss, loss_dict = model(batch, return_loss=True)
                for key, value in loss_dict.items():
                    losses_dict[key].append(value.detach().cpu().item())

            i+=1
            if i % 1 == 0:  # Adjust frequency based on your observation
                torch.cuda.empty_cache()
                i=0

    loss_off = np.mean(losses_dict['offset_loss'])
    loss_sem = np.mean(losses_dict['semantic_loss'])
    loss_total = loss_off + loss_sem

    return loss_total, loss_off, loss_sem


def run_training(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    scheduler=None,
    early_stopper=None,
    verbose=False,
    raster_hierarchical=False
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
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    mb = master_bar(range(epochs))  # Master bar for epochs

    torch.autograd.set_detect_anomaly(True)

    for epoch in mb:
        mb.main_bar.comment = f"Epoch {epoch + 1}/{epochs}"
        # Training phase
        train_loss_total, train_loss_off, train_loss_sem = train(model, train_loader, optimizer, scheduler, scaler, epoch, mb, raster_hierarchical)

        # Validation phase
        val_loss_total, val_loss_off, val_loss_sem = validate(model, val_loader, epoch, mb, raster_hierarchical)

        # Log losses
        log_message = (
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Total Loss: {train_loss_total:.4f}, Val Total Loss: {val_loss_total:.4f}, "
            f"Train Offset Loss: {train_loss_off:.4f}, Val Offset Loss: {val_loss_off:.4f}, "
            f"Train Semantic Loss: {train_loss_sem:.4f}, Val Semantic Loss: {val_loss_sem:.4f}"
        )
        logging.info(log_message)

        if verbose:
            mb.write(f"Epoch {epoch+1}/{epochs}, Tr Total: {train_loss_total:.4f}, Val Total: {val_loss_total:.4f}, Tr Off: {train_loss_off:.4f}, Val Off: {val_loss_off:.4f}, Tr Sem: {train_loss_sem:.4f}, Val Sem: {val_loss_sem:.4f}")

        # Step scheduler if provided
        # if scheduler:
        #     scheduler.step(val_loss)

        # Early stopping check
        if early_stopper:
            early_stopper(model, train_loss_total, val_loss_total)
            if early_stopper.early_stop:
                best_train_loss, best_val_loss = early_stopper.get_scores()
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                logging.info(f"Best scores:\ttrain: {best_train_loss:.4f}, val: {best_val_loss:.4f}")
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    print(f"Best scores:\ttrain: {best_train_loss:.4f}, val: {best_val_loss:.4f}")
                break
