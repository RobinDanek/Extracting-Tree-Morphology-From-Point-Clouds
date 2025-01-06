import torch
import numpy as np
import os
from Modules.TreeLearn.TreeLearn import TreeLearn
from Modules.TreeLearn.train import run_training
from Modules.DataLoading.TreeSet import TreeSet, get_dataloader
from Modules.Utils import EarlyStopper
from timm.scheduler.cosine_lr import CosineLRScheduler

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

###### Define parameters #######

batch_size = 2
epochs = 10

# trainloader
train_root = os.path.join( 'data', 'labeled', 'trainset' )
trainset = TreeSet( data_root=train_root, training=True )
train_loader = get_dataloader( trainset, batch_size, num_workers=0, training=True )
# valloader
val_root = os.path.join( 'data', 'labeled', 'testset' )
valset = TreeSet( data_root=val_root, training=False )
val_loader = get_dataloader( valset, batch_size, num_workers=0, training=False )

# model
model = TreeLearn( dim_feat=0, use_coords=True, use_feats=False )

# scheduler and optimizer
optimizer = torch.optim.AdamW( model.parameters(), lr = 0.002, weight_decay= 0.001  )
scheduler = CosineLRScheduler(optimizer, t_initial=1300, lr_min=0.0001, cycle_decay=1, warmup_lr_init=0.00001, warmup_t=50, cycle_limit=1, t_in_epochs=True)

# early stopper
early_stopper = EarlyStopper( verbose=True )


# Start training
if __name__ == "__main__":
    run_training( model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                epochs=epochs, scheduler=scheduler, early_stopper=early_stopper )