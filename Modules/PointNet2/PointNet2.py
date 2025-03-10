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
                use_features=True,
                depth=4,
                **kwargs
                ):
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.loss_multiplier_semantic = loss_multiplier_semantic
        self.loss_multiplier_offset = loss_multiplier_offset

        self.use_coords = use_coords
        self.use_features = use_features
        self.depth = depth

        # Set Abstraction Layers
        input_dim = 0
        if use_coords:
            input_dim += 3
        if use_features:
            input_dim += dim_feat

        if depth == 4:
            # Original 4-layer configuration
            self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_dim, [32, 32, 64], False)
            self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
            self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
            self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
            
            self.fp4 = PointNetFeaturePropagation(768, [256, 256])
            self.fp3 = PointNetFeaturePropagation(384, [256, 256])
            self.fp2 = PointNetFeaturePropagation(320, [256, 128])
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        elif depth == 5:
            # Original 4-layer configuration
            self.sa1 = PointNetSetAbstraction(100, 0.1, 32, input_dim, [32, 32, 64], False)
            self.sa2 = PointNetSetAbstraction(50, 0.2, 32, 64 + 3, [64, 64, 128], False)
            self.sa3 = PointNetSetAbstraction(20, 0.4, 32, 128 + 3, [128, 128, 256], False)
            self.sa4 = PointNetSetAbstraction(8, 0.8, 32, 256 + 3, [256, 256, 512], False)
            
            self.fp4 = PointNetFeaturePropagation(768, [256, 256])
            self.fp3 = PointNetFeaturePropagation(384, [256, 256])
            self.fp2 = PointNetFeaturePropagation(320, [256, 128])
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        elif depth == 6:
            # Original 4-layer configuration
            self.sa1 = PointNetSetAbstraction(50, 0.1, 32, input_dim, [32, 32, 64], False)
            self.sa2 = PointNetSetAbstraction(20, 0.2, 32, 64 + 3, [64, 64, 128], False)
            self.sa3 = PointNetSetAbstraction(10, 0.4, 32, 128 + 3, [128, 128, 256], False)
            self.sa4 = PointNetSetAbstraction(4, 0.8, 32, 256 + 3, [256, 256, 512], False)
            
            self.fp4 = PointNetFeaturePropagation(768, [256, 256])
            self.fp3 = PointNetFeaturePropagation(384, [256, 256])
            self.fp2 = PointNetFeaturePropagation(320, [256, 128])
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        elif depth == 3:
            # Three-layer configuration with radii: 0.1, 0.3, 0.6
            self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_dim, [32, 32, 64], False)
            self.sa2 = PointNetSetAbstraction(256, 0.3, 32, 64 + 3, [64, 64, 128], False)
            self.sa3 = PointNetSetAbstraction(64, 0.6, 32, 128 + 3, [128, 128, 256], False)
            
            # Feature Propagation: note the concatenation of skip connections from SA layers
            self.fp3 = PointNetFeaturePropagation(128 + 256, [256, 256])
            self.fp2 = PointNetFeaturePropagation(64 + 256, [256, 128])
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
            
        elif depth == 2:
            # Two-layer configuration with radii: 0.1 and 0.3
            self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, input_dim, [32, 32, 64], False)
            self.sa2 = PointNetSetAbstraction(256, 0.3, 32, 64 + 3, [64, 64, 128], False)
            
            self.fp2 = PointNetFeaturePropagation(64 + 128, [128, 128, 128])
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
            
        else:
            raise ValueError("Unsupported depth value. Please use depth=2, 3, or 4.")

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
            if self.depth == 4 or self.depth == 5 or self.depth == 6:
                l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
                l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
                l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
                l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

                l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
                l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
                l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
                l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
                
            elif self.depth == 3:
                l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
                l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
                l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

                l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
                l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
                l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
                
            elif self.depth == 2:
                l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
                l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

                l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
                l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
            else:
                raise ValueError("Unsupported depth value. Please use depth=2, 3, or 4.")

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
    

    def forward_hierarchical_streaming(self, batch, return_loss, scaler=None):
        """
        Processes a tree (composed of multiple mini‐batches) in a streaming manner.
        
        When return_loss is True, computes a loss for each mini‐batch (by masking the global labels with
        the mini‐batch’s point IDs) and backpropagates it immediately, thereby releasing its computation graph.
        In all cases, it aggregates predictions over the whole tree so that inference can be performed.
        
        Returns:
            If return_loss is True: a tuple (avg_loss, output) where output is a dict with aggregated predictions.
            Otherwise, returns the output dict with aggregated predictions only.
        """
        # Preallocate global tensors for prediction aggregation.
        cloud_length = batch["cloud_length"]
        device = "cuda"  # adjust if using a different device

        avg_off_predictions = torch.zeros((cloud_length, 3), dtype=torch.float, device=device)
        avg_sem_predictions = torch.zeros((cloud_length, 2), dtype=torch.float, device=device)
        count_predictions_sem = torch.zeros((cloud_length, 1), dtype=torch.float, device=device)
        count_predictions_off = torch.zeros((cloud_length, 1), dtype=torch.float, device=device)

        total_loss = 0.0
        loss_dict = {"offset_loss": 0, "semantic_loss": 0}

        num_minibatches = 0


        # Iterate over mini‐batches using the streaming generator.
        for mini_batch in batch["mini_batches"]:
            # Retrieve necessary tensors.
            point_ids = mini_batch["point_ids"]      # 1D tensor indexing the global cloud for semantic predictions.
            mask_off  = mini_batch["masks_off"]        # Boolean mask for offset predictions.
            masks_pad = mini_batch["masks_pad"]         # Boolean mask indicating real (non-padded) points.

            # Backbone pass.
            backbone_feats = self.forward_backbone(
                coords=mini_batch["coords"],
                feats=mini_batch["feats"],
            )

            # Head pass (with autocast disabled).
            with torch.amp.autocast('cuda', enabled=False):
                sem_pred_logits = self.semantic_linear(backbone_feats)
                off_preds = self.offset_linear(backbone_feats)

            # Reshape predictions.
            # Expected shapes:
            #   sem_pred_logits: [B, C, max_points] and off_preds: [B, 3, max_points]
            sem_logits_flat = sem_pred_logits.permute(0, 2, 1).reshape(-1, 2)
            off_preds_flat   = off_preds.permute(0, 2, 1).reshape(-1, 3)

            # Use the padding mask to select valid predictions.
            mask_flat = masks_pad.reshape(-1)
            sem_logits_valid = sem_logits_flat[mask_flat]  # [N_valid, 2]
            off_preds_valid  = off_preds_flat[mask_flat]     # [N_valid, 3]

            # Further apply the offset mask.
            off_preds_valid = off_preds_valid[mask_off]
            point_ids_off = point_ids[mask_off]

            # Scatter the mini‐batch predictions into the global aggregates.
            # Here we assume mini_batch["point_ids"] has length equal to sem_logits_valid.
            avg_sem_predictions[point_ids] += sem_logits_valid.detach()
            avg_off_predictions[point_ids_off] += off_preds_valid.detach()

            count_predictions_sem[point_ids] += 1
            count_predictions_off[point_ids_off] += 1

            # If training (loss should be computed), calculate the loss for this mini‐batch.
            if return_loss:
                # For semantic predictions, mask the global semantic labels with the mini‐batch indices.
                mini_sem_labels = batch["semantic_labels"].squeeze()[point_ids.cpu()].to(device)
                mini_off_labels = batch["offset_labels"][point_ids_off.cpu()].to(device)

                mini_loss, mini_loss_dict = self.get_loss_hierarchical(
                    model_output={
                        "semantic_prediction_logits": sem_logits_valid,
                        "offset_predictions": off_preds_valid
                    },
                    semantic_labels=mini_sem_labels,
                    offset_labels=mini_off_labels,
                    n_points=None  # Use all points in the mini‐batch.
                )

                # Backpropagate immediately to release the mini‐batch graph.
                if scaler:
                    scaler.scale(mini_loss*50).backward()

                loss_dict["offset_loss"] += mini_loss_dict["offset_loss"]
                loss_dict["semantic_loss"] += mini_loss_dict["semantic_loss"]

                total_loss += mini_loss.item()
                num_minibatches += 1

            # Delete local variables from this iteration to help free memory.
            del backbone_feats, sem_pred_logits, off_preds, sem_logits_flat, off_preds_flat, mask_flat, sem_logits_valid, off_preds_valid
            del mini_batch  # Remove reference to the mini-batch dictionary
            # Optionally, call: torch.cuda.empty_cache()

        # After processing all mini‐batches, average the aggregated predictions.
        nonzero_mask_sem = count_predictions_sem.squeeze(1) > 0
        nonzero_mask_off = count_predictions_off.squeeze(1) > 0
        avg_sem_predictions[nonzero_mask_sem] /= count_predictions_sem[nonzero_mask_sem]
        avg_off_predictions[nonzero_mask_off] /= count_predictions_off[nonzero_mask_off]

        output = {
            "semantic_prediction_logits": avg_sem_predictions,
            "offset_predictions": avg_off_predictions
        }

        if return_loss:
            avg_loss = total_loss / num_minibatches if num_minibatches > 0 else 0.0
            loss_dict["offset_loss"] /= num_minibatches if num_minibatches > 0 else 0.0
            loss_dict["semantic_loss"] /= num_minibatches if num_minibatches > 0 else 0.0

            return avg_loss, loss_dict
        else:
            return output

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

        output["semantic_prediction_logits"] = avg_sem_predictions
        output["offset_predictions"] = avg_off_predictions

        if return_loss:
        # Compute the loss using the averaged predictions
            output = self.get_loss_hierarchical(output, semantic_labels.squeeze(), offset_labels)

        return output

    def get_loss_hierarchical(self, model_output, semantic_labels, offset_labels, **kwargs):

        loss_dict = dict()

        semantic_loss, offset_loss = point_wise_loss(model_output['semantic_prediction_logits'].float(),
                                              model_output['offset_predictions'].float(),
                                              semantic_labels,
                                              offset_labels)

        loss_dict['semantic_loss'] = semantic_loss * self.loss_multiplier_semantic
        loss_dict['offset_loss'] = offset_loss * self.loss_multiplier_offset

        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict