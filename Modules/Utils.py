import numpy as np
import pandas as pd
import torch
import functools
from scipy.optimize import curve_fit

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
        self.train_loss = None
        self.early_stop = False

    def __call__(self, model, train_loss, val_loss):
        """
        Check whether validation loss has improved and handle early stopping.

        Args:
            model: The model to save if validation loss improves.
            val_loss (float): The current validation loss.
        """
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.train_loss = train_loss
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

    def get_scores(self):
        return self.train_loss, self.best_loss
    
#################### Power Law fit ##################

def power_law(x, a, b):
    """Power-law function y = a * x^b."""
    return a * np.power(x, b)

def fit_power_law(x, y):
    """
    Fits a power law y = a * x^b to the given data in log-log space.

    Parameters:
        x (numpy.ndarray): Original distances.
        y (numpy.ndarray): Transformed distances.

    Returns:
        x_fit (numpy.ndarray): X values for fitted line.
        y_fit (numpy.ndarray): Corresponding Y values from fitted model.
        a (float): Estimated coefficient.
        b (float): Estimated exponent.
        a_err (float): Standard error of 'a'.
        b_err (float): Standard error of 'b'.
    """
    epsilon = 1e-8  # For numerical stability

    # Clamp values so that none are below epsilon
    x_clipped = np.clip(x, epsilon, None)
    y_clipped = np.clip(y, epsilon, None)
    
    log_x = np.log(x_clipped)
    log_y = np.log(y_clipped)

    # Fit the power-law model in log-log space
    popt, pcov = curve_fit(lambda log_x, log_a, b: log_a + b * log_x, log_x, log_y)
    log_a, b = popt
    a = np.exp(log_a)

    # Compute standard errors
    perr = np.sqrt(np.diag(pcov))
    a_err = a * perr[0]  # Convert log error to standard scale
    b_err = perr[1]

    # Generate fitted values with x values evenly spaced in log-space:
    x_fit = np.logspace(-5, np.log10(x_clipped.max()), 100)
    y_fit = power_law(x_fit, a, b)

    return x_fit, y_fit, a, b, a_err, b_err

    # epsilon = 1e-4  # For numerical stability

    # # Clamp values so that none are below epsilon
    # # x_clipped = np.clip(x, epsilon, None)
    # # y_clipped = np.clip(y, epsilon, None)
    # x_clipped = x
    # y_clipped = y
    
    # # Fit the power-law model in linear space using non-linear least squares.
    # # p0 provides initial guesses for [a, b]. Adjust these if needed.
    # popt, pcov = curve_fit(power_law, x_clipped, y_clipped, p0=[1.0, 1.0])
    # a, b = popt

    # # Compute standard errors from the covariance matrix.
    # perr = np.sqrt(np.diag(pcov))
    # a_err = perr[0]
    # b_err = perr[1]

    # # Generate fitted values with x values evenly spaced in linear space.
    # x_fit = np.logspace(-4, np.log10(x_clipped.max()), 100)
    # y_fit = power_law(x_fit, a, b)

    # return x_fit, y_fit, a, b, a_err, b_err

def generate_log_bins(min_val, max_val):
    bins = []
    order_min = int(np.floor(np.log10(min_val)))
    order_max = int(np.ceil(np.log10(max_val)))
    for order in range(order_min, order_max + 1):
        for m in range(1, 10):
            value = m * 10**order
            if min_val <= value <= max_val:
                bins.append(value)
    bins = np.array(sorted(bins))
    # Make sure the bins cover the full range:
    if bins[0] > min_val:
        bins = np.insert(bins, 0, min_val)
    if bins[-1] < max_val:
        bins = np.append(bins, max_val)
    return bins

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