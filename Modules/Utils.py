import numpy as np
import pandas as pd
import torch
import functools

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