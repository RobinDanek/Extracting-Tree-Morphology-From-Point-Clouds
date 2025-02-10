import torch
import torch.nn as nn
import functools
# from torch_points3d.applications.pointnet2 import PointNet2
from .blocks import *
from Modules.Utils import cuda_cast
from Modules.Loss import point_wise_loss

class PointNet2(nn.Module):
    def __init__(self, 
                input_nc=3,
                loss_multiplier_semantic=1,
                loss_multiplier_offset=1,
                dim_feat=4,
                use_coords=True,
                use_features=False,
                **kwargs
                ):
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.loss_multiplier_semantic = loss_multiplier_semantic
        self.loss_multiplier_offset = loss_multiplier_offset

        self.use_coords = use_coords
        self.use_features = use_features

        # Set Abstraction Layers
        input_dim = 0
        if use_coords:
            input_dim += 3
        if use_features:
            input_dim += dim_feat
        self.sa1 = PointNetSetAbstraction(512, 2.0, 32, input_dim, [64, 64, 128], group_all=True)
        self.sa2 = PointNetSetAbstraction(128, 2.0, 64, 128 + 3, [128, 128, 256], group_all=True)
        self.sa3 = PointNetSetAbstraction(32, 2.0, 128, 256 + 3, [256, 512, 1024], group_all=True)

        # Feature Propagation Layers
        self.fp3 = PointNetFeaturePropagation(1024 + 256, [512, 512, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 128, [256, 256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + 4, [128, 128, 128])

        # Output MLP for per-point offset prediction and noise classification
        self.semantic_linear = MLP(128, 2, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(128, 3, norm_fn=norm_fn, num_layers=2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()

    def forward(self, batch, return_loss):
        output = dict()

        # Backbone pass
        output['backbone_feats'] = self.forward_backbone(
            coords=batch["coords"],
            feats=batch["feats"],
            batch_ids=batch["batch_ids"],
            batch_size=batch["batch_size"]
        )
        
        # Head pass
        output['semantic_prediction_logits'] = self.semantic_linear(output['backbone_feats'])
        output['offset_predictions'] = self.offset_linear(output['backbone_feats'])

        if return_loss:
            output = self.get_loss(model_output=output, **batch)
        
        return output

    @cuda_cast
    def forward_backbone(self, coords, feats, batch_ids, batch_size, **kwargs):
        # Handle optional features before passing to the network
        feats = feats if self.use_features else None

        l1_xyz, l1_features = self.sa1(coords, feats)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)

        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        backbone_features = self.fp1(coords, l1_xyz, feats, l1_features)

        return backbone_features

    def get_loss(self, model_output, semantic_labels, offset_labels, masks_off, masks_pad, **kwargs):
        loss_dict = dict()
        semantic_loss, offset_loss = point_wise_loss(model_output['semantic_prediction_logits'][masks_pad].float(), model_output['offset_predictions'][masks_pad][masks_off].float(), 
                                                            semantic_labels, offset_labels[masks_off])
        loss_dict['semantic_loss'] = semantic_loss * self.loss_multiplier_semantic
        loss_dict['offset_loss'] = offset_loss * self.loss_multiplier_offset

        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict