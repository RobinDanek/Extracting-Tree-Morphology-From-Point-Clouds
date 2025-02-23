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
                half_size=False,
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

        if half_size:
            self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_dim, [16, 16, 32], False)
            self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 32 + 3, [32, 32, 64], False)
            self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 64 + 3, [64, 64, 128], False)
            self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 128 + 3, [128, 128, 256], False)
            self.fp4 = PointNetFeaturePropagation(384, [128, 128])
            self.fp3 = PointNetFeaturePropagation(192, [128, 128])
            self.fp2 = PointNetFeaturePropagation(160, [128, 64])
            self.fp1 = PointNetFeaturePropagation(64, [64, 64, 128])
        else:
            self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_dim, [32, 32, 64], False)
            self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
            self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
            self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
            self.fp4 = PointNetFeaturePropagation(768, [256, 256])
            self.fp3 = PointNetFeaturePropagation(384, [256, 256])
            self.fp2 = PointNetFeaturePropagation(320, [256, 128])
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # Output MLP for per-point offset prediction and noise classification
        self.semantic_linear = ConvHead(128, 2, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = ConvHead(128, 3, norm_fn=norm_fn, num_layers=2)

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
        )
        
        # Head pass
        output['semantic_prediction_logits'] = self.semantic_linear(output['backbone_feats'])
        output['offset_predictions'] = self.offset_linear(output['backbone_feats'])

        if return_loss:
            output = self.get_loss(model_output=output, **batch)
        
        return output

    @cuda_cast
    def forward_backbone(self, coords, feats, **kwargs):
        # Handle optional features before passing to the network
        if self.use_features:
            l0_points = feats
        else:
            l0_points = None

        l0_xyz = coords

        with torch.amp.autocast('cuda', enabled=False):
            l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        

            l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
            l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        # l1_xyz, l1_features = self.sa1(coords, feats)
        # l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        # l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)

        # l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        # l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        # backbone_features = self.fp1(coords, l1_xyz, feats, l1_features)

        backbone_features = l0_points

        return backbone_features

    def get_loss(self, model_output, semantic_labels, offset_labels, masks_off, masks_pad, **kwargs):
        loss_dict = dict()

        # Permute to [B, max_points, C] and then flatten to [B * max_points, C]
        sem_logits_flat = model_output['semantic_prediction_logits'].permute(0, 2, 1).reshape(-1, 2)
        off_preds_flat   = model_output['offset_predictions'].permute(0, 2, 1).reshape(-1, 3)

        # Flatten the padding mask from [B, max_points] to [B * max_points]
        mask_flat = masks_pad.reshape(-1)

        # Index the flattened predictions with the valid mask.
        # This yields tensors of shape [N_valid, C].
        sem_logits_valid = sem_logits_flat[mask_flat]
        off_preds_valid  = off_preds_flat[mask_flat]

        # Now apply offset mask
        off_preds_valid = off_preds_valid[masks_off]

        semantic_loss, offset_loss = point_wise_loss(sem_logits_valid.float(),
                                              off_preds_valid.float(),
                                              semantic_labels,
                                              offset_labels)

        loss_dict['semantic_loss'] = semantic_loss * self.loss_multiplier_semantic
        loss_dict['offset_loss'] = offset_loss * self.loss_multiplier_offset

        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict
    

    def forward_hierarchical(self, batch, return_loss):
        output = dict()

        # First create arrays to store the per point predictions
        avg_off_predictions = torch.zeros((batch["cloud_length"], 3), dtype=torch.float, device="cuda")
        avg_sem_predictions = torch.zeros((batch["cloud_length"], 2), dtype=torch.float, device="cuda")
        count_predictions_sem = torch.zeros((batch["cloud_length"], 1), dtype=torch.float, device="cuda")
        count_predictions_off = torch.zeros((batch["cloud_length"], 1), dtype=torch.float, device="cuda")


        # Now make predictions per mini batch
        for mini_batch in batch["mini_batches"]:

            point_ids = mini_batch["point_ids"]
            mask_off = mini_batch["masks_off"]
            masks_pad = mini_batch["masks_pad"]
            offset_labels = batch["offset_labels"]
            semantic_labels = batch["semantic_labels"]

            # Backbone pass
            backbone_feats = self.forward_backbone(
                coords=mini_batch["coords"],
                feats=mini_batch["feats"],
            )

            # Head pass
            with torch.amp.autocast('cuda', enabled=False):
                sem_pred_logits = self.semantic_linear(backbone_feats)
                off_preds = self.offset_linear(backbone_feats)

            # Permute to [B, max_points, C] then flatten to [B * max_points, C]
            sem_logits_flat = sem_pred_logits.permute(0, 2, 1).reshape(-1, 2)
            off_preds_flat   = off_preds.permute(0, 2, 1).reshape(-1, 3)

            # Flatten the padding mask (which marks valid points within the mini-batch)
            mask_flat = masks_pad.reshape(-1)
            sem_logits_valid = sem_logits_flat[mask_flat]  # shape: [N_valid, 2]
            off_preds_valid  = off_preds_flat[mask_flat]     # shape: [N_valid, 3]

            # Ffurther apply the offset mask
            off_preds_valid = off_preds_valid[mask_off]
            point_ids_off = point_ids[mask_off]

            # Now, scatter the predictions into the final prediction arrays using the point_ids.
            # The point_ids tensor maps each valid mini-batch prediction to its original cloud index.
            avg_sem_predictions[point_ids] += sem_logits_valid
            avg_off_predictions[point_ids_off] += off_preds_valid

            # Increment the count for each index
            count_predictions_sem[point_ids] += 1
            count_predictions_off[point_ids_off] += 1

        # Avoid division by zero: only average for points that were predicted at least once.
        nonzero_mask_sem = count_predictions_sem.squeeze(1) > 0
        nonzero_mask_off = count_predictions_off.squeeze(1) > 0
        avg_sem_predictions[nonzero_mask_sem] /= count_predictions_sem[nonzero_mask_sem]
        avg_off_predictions[nonzero_mask_off] /= count_predictions_off[nonzero_mask_off]

        # output["semantic_prediction_logits"] = avg_sem_predictions
        # output["offset_predictions"] = avg_off_predictions

        if return_loss:
        # Compute the loss using the averaged predictions
            output = self.get_loss_hierarchical(avg_sem_predictions, avg_off_predictions, semantic_labels.squeeze(), offset_labels)

        return output

    def get_loss_hierarchical(self, sem_logits, off_preds, semantic_labels, offset_labels, **kwargs):

        loss_dict = dict()

        semantic_loss, offset_loss = point_wise_loss(sem_logits.float(),
                                              off_preds.float(),
                                              semantic_labels,
                                              offset_labels)

        loss_dict['semantic_loss'] = semantic_loss * self.loss_multiplier_semantic
        loss_dict['offset_loss'] = offset_loss * self.loss_multiplier_offset

        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict