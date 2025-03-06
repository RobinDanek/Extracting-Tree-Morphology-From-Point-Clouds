import torch
import torch.nn as nn

from Modules.PointNet2.PointNet2 import PointNet2
from Modules.PointTransformerV3.PointTransformerV3 import PointTransformerWithHeads
from Modules.TreeLearn.TreeLearn import TreeLearn
from Modules.DataLoading.TreeSet import *
from Modules.DataLoading.RasterizedTreeSet import *

import pandas as pd
import numpy as np
import json

def load_model( 
    model_type, offset_model_dir, noise_model_dir, num_blocks=3, use_feats=True, use_coords=True, voxel_size=0.02, pointnet2_depth=4,
    ):
    """
        This function loads a model by returning a dict of the models.
        The dict includes the 4 offset and the 4 noise predicting models.
    """
    if model_type="treelearn":
        model_dict = load_TreeLearn(offset_model_dir, noise_model_dir, num_blocks, use_feats, use_coords, voxel_size)
    elif model_type="pointnet2":
        model_dict = load_pointnet2(offset_model_dir, noise_model_dir, use_feats, use_coords, pointnet2_depth)
    elif model_type="pointtransformerv3":
        model_dict = load_pointtransformerv3( offset_model_dir, noise_model_dir, use_feats, use_coords, voxel_size )

    return model_dict


def load_treelearn(offset_model_dir, noise_model_dir, num_blocks, use_feats, use_coords, voxel_size):
    model_dict = {}

    offset_model_paths = [os.path.join( offset_model_dir, f ) for f in os.listdir( offset_model_dir ) if f.endswith('.pt')]
    noise_model_paths = [os.path.join( noise_model_dir, f ) for f in os.listdir( noise_model_dir ) if f.endswith('.pt') ]

    for model_path in offset_model_paths:
        model = TreeLearn( num_blocks=num_blocks, use_feats=use_feats, use_coords=use_coords, voxel_size=voxel_size )
        model.load_state_dict(torch.load( model_path, weights_only=True ))

        plot = os.path.basename(model_path).splitext('_')[-1].splitext('.')[0]

        model_dict[f"O_{plot}"] = model

    for model_path in noise_model_dir:
        model = TreeLearn( num_blocks=num_blocks, use_feats=use_feats, use_coords=use_coords, voxel_size=voxel_size )
        model.load_state_dict(torch.load( model_path, weights_only=True ))

        plot = os.path.basename(model_path).splitext('_')[-1].splitext('.')[0]

        model_dict[f"N_{plot}"] = model

    return model_dict

def load_pointnet2(offset_model_dir, noise_model_dir, use_feats, use_coords, pointnet2_depth):
    model_dict = {}

    offset_model_paths = [os.path.join( offset_model_dir, f ) for f in os.listdir( offset_model_dir ) if f.endswith('.pt')]
    noise_model_paths = [os.path.join( noise_model_dir, f ) for f in os.listdir( noise_model_dir ) if f.endswith('.pt') ]

    for model_path in offset_model_paths:
        model = PointNet2( use_feats=use_feats, use_coords=use_coords, depth=pointnet2_depth )
        model.load_state_dict(torch.load( model_path, weights_only=True ))

        plot = os.path.basename(model_path).splitext('_')[-1].splitext('.')[0]

        model_dict[f"O_{plot}"] = model

    for model_path in noise_model_dir:
        model = PointNet2( use_feats=use_feats, use_coords=use_coords, depth=pointnet2_depth )
        model.load_state_dict(torch.load( model_path, weights_only=True ))

        plot = os.path.basename(model_path).splitext('_')[-1].splitext('.')[0]

        model_dict[f"N_{plot}"] = model

    return model_dict

def load_pointtransformerv3(offset_model_dir, noise_model_dir, use_feats, use_coords, voxel_size):
    model_dict = {}

    offset_model_paths = [os.path.join( offset_model_dir, f ) for f in os.listdir( offset_model_dir ) if f.endswith('.pt')]
    noise_model_paths = [os.path.join( noise_model_dir, f ) for f in os.listdir( noise_model_dir ) if f.endswith('.pt') ]

    for model_path in offset_model_paths:
        model = PointTransformerV3( use_feats=use_feats, use_coords=use_coords, voxel_size=voxel_size )
        model.load_state_dict(torch.load( model_path, weights_only=True ))

        plot = os.path.basename(model_path).splitext('_')[-1].splitext('.')[0]

        model_dict[f"O_{plot}"] = model

    for model_path in noise_model_dir:
        model = PointTransformerV3( use_feats=use_feats, use_coords=use_coords, voxel_size=voxel_size )
        model.load_state_dict(torch.load( model_path, weights_only=True ))

        plot = os.path.basename(model_path).splitext('_')[-1].splitext('.')[0]

        model_dict[f"N_{plot}"] = model

    return model_dict
