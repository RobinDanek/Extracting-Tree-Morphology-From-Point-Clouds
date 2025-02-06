import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import *

class MLP(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)



class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channels, mlp_channels):
        """
        npoint: Number of sampled points
        radius: Radius of ball query
        nsample: Number of neighbors to sample
        in_channels: Input feature channels
        mlp_channels: List of MLP output sizes
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_channels[0]),
            nn.ReLU(),
            nn.Linear(mlp_channels[0], mlp_channels[1]),
            nn.ReLU(),
            nn.Linear(mlp_channels[1], mlp_channels[2])
        )

    def forward(self, xyz, features):
        """
        xyz: [B, N, 3] - Input points
        features: [B, N, C] or None - Features (None if using only xyz)
        
        Returns:
        - new_xyz: [B, npoint, 3] - Sampled points
        - new_features: [B, npoint, C_out]
        """
        B, N, _ = xyz.shape

        # Step 1: Farthest Point Sampling (FPS)
        fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
        new_xyz = gather_operation(xyz.transpose(1, 2), fps_idx).transpose(1, 2)  # [B, npoint, 3]

        # Step 2: Ball Query to find neighbors
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # [B, npoint, nsample]

        # Step 3: Grouping features
        if features is None:
            features = torch.zeros(B, N, 1, device=xyz.device)  # Placeholder features if None
        grouped_features = grouping_operation(features.transpose(1, 2), idx)  # [B, C, npoint, nsample]

        # Step 4: Apply MLP on grouped features
        grouped_features = self.mlp(grouped_features)  # [B, C_out, npoint, nsample]
        new_features = F.max_pool2d(grouped_features, kernel_size=[1, self.nsample]).squeeze(-1)  # [B, C_out, npoint]

        return new_xyz, new_features.transpose(1, 2)  # [B, npoint, C_out]



class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_channels[0]),
            nn.ReLU(),
            nn.Linear(mlp_channels[0], mlp_channels[1]),
            nn.ReLU(),
            nn.Linear(mlp_channels[1], mlp_channels[2])
        )

    def forward(self, xyz1, xyz2, features1, features2):
        """
        xyz1: [B, N, 3] - Original resolution points
        xyz2: [B, npoint, 3] - Downsampled points
        features1: [B, N, C1] or None - Features at original resolution
        features2: [B, npoint, C2] - Features from SA layer

        Returns:
        - new_features: [B, N, C_out] - Upsampled features
        """
        dist, idx = three_nn(xyz1, xyz2)  # Find 3 nearest neighbors
        dist = torch.clamp(dist, min=1e-10)  # Avoid division by zero
        norm = torch.sum(1.0 / dist, dim=-1, keepdim=True)
        weight = (1.0 / dist) / norm  # Compute interpolation weights

        interpolated_features = three_interpolate(features2.transpose(1, 2), idx, weight)
        interpolated_features = interpolated_features.transpose(1, 2)  # [B, N, C2]

        if features1 is not None:
            new_features = torch.cat([features1, interpolated_features], dim=-1)
        else:
            new_features = interpolated_features  # Use only interpolated features

        return self.mlp(new_features)




