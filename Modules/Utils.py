import numpy as np
import pandas as pd
import torch
import functools

#################### EARLY STOPPER ##################

class EarlyStopper:
    def __init__(self, patience=5, verbose=False, model_save_path=None):
        """
        Early stopping utility to stop training if validation loss doesn't improve.
        Saves the model if validation loss improves.
        
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            verbose (bool): If True, prints messages when validation loss doesn't improve.
            model_save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.model_save_path = model_save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, model, val_loss):
        """
        Check whether validation loss has improved and handle early stopping.

        Args:
            model: The model to save if validation loss improves.
            val_loss (float): The current validation loss.
        """
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement is seen
            if self.model_save_path:
                self.save_model(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_save_path)

#################### GET DEVICE #####################

def get_device(GPU=True):
    if torch.cuda.is_available() and GPU:
        device = torch.device('cuda')  
        print("Using cuda device")
        # Get the current CUDA device
        device_id = torch.cuda.current_device()
        # Print device properties
        device_name = torch.cuda.get_device_name(device_id)
        print(f"Using CUDA Device: {device_name}")
    else:
        device = torch.device('cpu')
        print("Using cpu")
    return device

#################### CUDA CAST ######################

def cuda_cast(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if isinstance(x, torch.Tensor):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper