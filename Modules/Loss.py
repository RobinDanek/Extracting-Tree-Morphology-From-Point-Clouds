from .Utils import cuda_cast
import torch
import torch.nn.functional as F
import numpy as np 

@cuda_cast
def point_wise_loss(semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, n_points=None):

    if n_points is not None and len(offset_predictions) >= n_points:
        permuted_indices_sem = torch.randperm(len(semantic_prediction_logits))
        permuted_indices_off = torch.randperm(len(offset_predictions))
        ind_sem = permuted_indices_sem[:n_points]
        ind_off = permuted_indices_off[:n_points]
    else:
        ind_sem = torch.arange(len(semantic_prediction_logits))
        ind_off = torch.arange(len(offset_predictions))

    if len(semantic_prediction_logits) == 0:
        semantic_loss = 0 * semantic_labels.sum()
    else:
        # semantic_loss
        semantic_loss = F.cross_entropy(
            semantic_prediction_logits[ind_sem], semantic_labels[ind_sem], reduction='sum') / len(semantic_prediction_logits[ind_sem])
        
    if len(offset_predictions) == 0:
        offset_loss = 0 * offset_predictions.sum()
    else:
        # offset loss
        offset_losses = (offset_predictions[ind_off] - offset_labels[ind_off]).pow(2).sum(1).sqrt()
        # Clamp for stability
        eps = 1e-8
        offset_losses = torch.sqrt(torch.clamp(offset_losses, min=eps))

        offset_loss = offset_losses.mean()

    return semantic_loss, offset_loss
