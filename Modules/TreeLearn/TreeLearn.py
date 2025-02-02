# Definition of the adapted TreeLearn Model

import functools
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.modules import SparseModule

from Modules.Utils import cuda_cast
from .blocks import MLP, ResidualBlock, UBlock
from Modules.Loss import point_wise_loss

# LOSS_MULTIPLIER_SEMANTIC = 50 # multiply semantic loss for similar magnitude with offset loss
LOSS_MULTIPLIER_SEMANTIC = 0
N_POINTS = None # only calculate loss for specified number of randomly sampled points; use all points if set to None

####################### TREELEARN #############################

class TreeLearn(nn.Module):

    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 kernel_size=3,
                 dim_coord=3,
                 dim_feat=1,
                 fixed_modules=[],
                 use_feats=True,
                 use_coords=False,
                 spatial_shape=None,
                 max_num_points_per_voxel=10,
                 voxel_size=0.1,
                 loss_multiplier_semantic=1,
                 loss_multiplier_offset=1,
                 **kwargs):

        super().__init__()
        self.voxel_size = voxel_size
        self.fixed_modules = fixed_modules
        self.use_feats = use_feats
        self.use_coords = use_coords
        self.spatial_shape = spatial_shape
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.loss_multiplier_semantic = loss_multiplier_semantic
        self.loss_multiplier_offset = loss_multiplier_offset

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        
        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                dim_coord + dim_feat, channels, kernel_size=kernel_size, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, ResidualBlock, kernel_size, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
        
        # head
        self.semantic_linear = MLP(channels, 2, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)
        self.init_weights()

        # weight init
        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()


    # manually set batchnorms in fixed modules to eval mode
    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()


    def forward(self, batch, return_loss):
        backbone_output, v2p_map = self.forward_backbone(
            coords=batch["coords"],
            feats=batch["feats"],
            batch_ids=batch["batch_ids"],
            batch_size=batch["batch_size"]
        )
        
        noise_backbone_output, noise_v2p_map = None, None
        if "noise_coords" in batch and batch["noise_coords"] is not None:
            noise_backbone_output, noise_v2p_map = self.forward_backbone(
                coords=batch["noise_coords"],
                feats=batch["noise_feats"],
                batch_ids=batch["noise_batch_ids"],
                batch_size=batch["batch_size"]
            )
        
        output = self.forward_head(backbone_output, v2p_map, noise_backbone_output, noise_v2p_map)
        
        if return_loss:
            output = self.get_loss(model_output=output, **batch)
        
        return output

    @cuda_cast
    def forward_backbone(self, coords, feats, batch_ids, batch_size, **kwargs):
        voxel_feats, voxel_coords, v2p_map, spatial_shape = voxelize(torch.hstack([coords, feats]), batch_ids, batch_size, self.voxel_size, self.use_coords, self.use_feats, max_num_points_per_voxel=self.max_num_points_per_voxel)
        #print(f"######### SPATIAL SHAPE: {spatial_shape} ###############")
        if self.spatial_shape is not None:
            spatial_shape = torch.tensor(self.spatial_shape, device=voxel_coords.device)
        #print("Generating input")
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        #print(f"input tensor: {input}")
        #print("Generating output: Input conv")
        output = self.input_conv(input)
        #print("Generating output: UNet")
        output = self.unet(output)
        #print("Generating output: output layer")
        output = self.output_layer(output)
        return output, v2p_map
    

    def forward_head(self, backbone_output, v2p_map, noise_backbone_output, noise_v2p_map):
        output = dict()
        backbone_feats = backbone_output.features[v2p_map]
        output['backbone_feats'] = backbone_feats
        
        if noise_backbone_output is not None:
            noise_backbone_feats = noise_backbone_output.features[noise_v2p_map]
            output['semantic_prediction_logits'] = self.semantic_linear(noise_backbone_feats)
        else:
            output['semantic_prediction_logits'] = self.semantic_linear(backbone_feats)
        
        output['offset_predictions'] = self.offset_linear(backbone_feats)
        return output


    def get_loss(self, model_output, semantic_labels, offset_labels, masks_off, **kwargs):
        loss_dict = dict()
        semantic_loss, offset_loss = point_wise_loss(model_output['semantic_prediction_logits'].float(), model_output['offset_predictions'][masks_off].float(), 
                                                            semantic_labels, offset_labels[masks_off], n_points=N_POINTS)
        loss_dict['semantic_loss'] = semantic_loss * self.loss_multiplier_semantic
        loss_dict['offset_loss'] = offset_loss * self.loss_multiplier_offset

        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict




def voxelize(feats, batch_ids, batch_size, voxel_size, use_coords, use_feats, max_num_points_per_voxel, epsilon=1):
    """
    Voxelize point clouds with batch IDs, tailored for TreeSet-style datasets.

    Args:
        feats (torch.Tensor): Input features (coordinates and optional features), shape (N, D).
        batch_ids (torch.Tensor): Batch IDs for each point, shape (N,).
        batch_size (int): Number of unique batches in the data.
        voxel_size (float): Size of the voxel grid cells.
        use_coords (bool): Whether to include coordinates in the output features.
        use_feats (bool): Whether to include additional features in the output features.
        max_num_points_per_voxel (int): Maximum number of points per voxel.
        epsilon (float): Small value added to the max range for stability.

    Returns:
        voxel_feats (torch.Tensor): Voxelized features, shape (M, D).
        voxel_coords (torch.Tensor): Voxel coordinates, shape (M, 4) (batch ID + Z, Y, X).
        v2p_maps (torch.Tensor): Voxel-to-point mappings, shape (M, max_num_points_per_voxel).
        spatial_shape (torch.Tensor): Shape of the voxel grid in (Z, Y, X) dimensions.
    """
    # Initialize outputs
    voxel_coords, voxel_feats, v2p_maps = [], [], []
    total_len_voxels = 0

    # Process each batch
    for i in range(batch_size):
        # Extract features for the current batch
        feats_one_element = feats[batch_ids == i]

        # Calculate min and max ranges for voxelization
        min_range = torch.min(feats_one_element[:, :3], dim=0).values
        max_range = torch.max(feats_one_element[:, :3], dim=0).values + epsilon

        # Create the voxelizer
        voxelizer = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, voxel_size],
            coors_range_xyz=min_range.tolist() + max_range.tolist(),
            num_point_features=feats.shape[1],
            max_num_voxels=feats.size(0),
            max_num_points_per_voxel=max_num_points_per_voxel,
            device=feats.device,
        )

        # Generate voxel data
        voxel_feat, voxel_coord, _, v2p_map = voxelizer.generate_voxel_with_id(feats_one_element)
        if voxel_coord.size(0) == 0:
            #print("SKIPPED EMPTY BATCH")
            continue  # Skip this batch if no valid voxels were generated
        #print(f"Batch {i}: v2p_map size={v2p_map.size()}")
        assert torch.sum(v2p_map == -1) == 0, f"Invalid entries in v2p_map for batch {i}"

        # Adjust voxel coordinates and add batch ID
        voxel_coord[:, [0, 2]] = voxel_coord[:, [2, 0]]  # Swap X and Z axes
        voxel_coord = torch.cat((torch.ones((len(voxel_coord), 1), device=feats.device) * i, voxel_coord), dim=1)

        # Compute mean features for each voxel
        zero_rows = torch.sum(voxel_feat == 0, dim=2) == voxel_feat.shape[2]
        voxel_feat[zero_rows] = float("nan")
        voxel_feat = torch.nanmean(voxel_feat, dim=1)

        # Apply feature selection
        if not use_coords:
            voxel_feat[:, :3] = torch.ones_like(voxel_feat[:, :3])
        if not use_feats:
            voxel_feat[:, 3:] = torch.ones_like(voxel_feat[:, 3:])
        voxel_feat = torch.hstack([voxel_feat[:, 3:], voxel_feat[:, :3]])

        # Collect results
        voxel_coords.append(voxel_coord)
        voxel_feats.append(voxel_feat)
        v2p_maps.append(v2p_map + total_len_voxels)
        total_len_voxels += len(voxel_coord)

        if torch.isnan(voxel_feat).any():
            print("Warning: NaNs detected in voxel_feats!")
        if torch.isnan(voxel_coord).any():
            print("Warning: NaNs detected in voxel_feats!")

        #print(f"batch {i} coord first column: {voxel_coord[0]}\n\tfirst feat column: {voxel_feat[0]}")

    # Concatenate results
    voxel_coords = torch.cat(voxel_coords, dim=0)
    voxel_feats = torch.cat(voxel_feats, dim=0)
    #print(f"Voxel feats size: {voxel_feats.shape}")
    v2p_maps = torch.cat(v2p_maps, dim=0)
    spatial_shape = voxel_coords.max(dim=0).values + 1
    #print(voxel_coords.max(dim=0))

    return voxel_feats, voxel_coords, v2p_maps, spatial_shape[1:]

