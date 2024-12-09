import numpy as np
import pandas as pd
import torch

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