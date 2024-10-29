import torch
from torch import nn

from nnunetv2.utilities.helpers import softmax_helper_dim1

from nnunetv2.training.Insta_Match_Losses.helpers import get_connected_components, get_regions
from nnunetv2.training.Insta_Match_Losses.helpers import dice, tversky
from nnunetv2.training.Insta_Match_Losses.helpers import compute_loss #blob loss
from nnunetv2.training.Insta_Match_Losses.overlap import MemoryEfficientSoftDiceLoss, MemoryEfficientTverskyLoss

class RegionDiceLoss(nn.Module):
    def __init__(self):
        super(RegionDiceLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, y):

        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label[i], _ = get_connected_components(y[i])
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)

        # Ensure inputs are proper probabilities
        x = torch.sigmoid(x) if x.dim() == 5 else x

        batch_size = x.size(0)
        losses = []

        for b in range(batch_size):
            x_volume = x[b].squeeze()  # (D, H, W)
            y_volume = multi_label[b].squeeze()

            region_map, num_features = get_regions(y_volume)

            if num_features == 0:
                # Handle cases with no regions
                losses.append(torch.tensor(1.0, device=self.device))
                continue

            region_dice_scores = []
            for region_label in range(1, num_features + 1):
                region_mask = (region_map == region_label)
                x_region = x_volume[region_mask]
                y_region = y_volume[region_mask]

                dice_score = dice(x_region, y_region)
                region_dice_scores.append(dice_score)

            # Calculate mean Dice score for this volume
            mean_dice = torch.mean(torch.stack(region_dice_scores))
            losses.append(1 - mean_dice)  # Convert to loss

        # Return mean loss across batch
        loss = torch.mean(torch.stack(losses))
        return loss

class RegionTverskyLoss(nn.Module):
    def __init__(self):
        super(RegionTverskyLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, y):

        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label[i], _ = get_connected_components(y[i])
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)

        # Ensure inputs are proper probabilities
        x = torch.sigmoid(x) if x.dim() == 5 else x

        batch_size = x.size(0)
        losses = []

        for b in range(batch_size):
            x_volume = x[b].squeeze()  # (D, H, W)
            y_volume = multi_label[b].squeeze()

            region_map, num_features = get_regions(y_volume)

            if num_features == 0:
                # Handle cases with no regions
                losses.append(torch.tensor(1.0, device=self.device))
                continue

            region_tversky_scores = []
            for region_label in range(1, num_features + 1):
                region_mask = (region_map == region_label)
                x_region = x_volume[region_mask]
                y_region = y_volume[region_mask]

                tversky_score = tversky(x_region, y_region)
                region_tversky_scores.append(tversky_score)

            # Calculate mean tversky score for this volume
            mean_tversky = torch.mean(torch.stack(region_tversky_scores))
            losses.append(1 - mean_tversky)  # Convert to loss

        # Return mean loss across batch
        loss = torch.mean(torch.stack(losses))
        return loss

class ClusterDiceLoss(nn.Module):
    def __init__(self):
        super(ClusterDiceLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, pred, target):
        if not pred.shape == target.shape:
            raise ValueError(f"Pred shape {pred.shape} must match target shape {target.shape}")

        batch_size = pred.shape[0]
        batch_losses = []

        for b in range(batch_size):
            pred_sample = pred[b].squeeze()
            target_sample = target[b].squeeze()

            # Ensure inputs require gradients
            pred_sample = pred_sample.requires_grad_(True)

            overlay = pred_sample + target_sample
            overlay = (overlay > 0).float()

            # Get regions without tracking gradients
            region_map, num_connected_components = get_regions(overlay)
            dice_scores = []

            for region_label in range(1, num_connected_components + 1):
                region_mask = (region_map == region_label)
                pred_region = pred_sample[region_mask]
                target_region = target_sample[region_mask]

                # Compute Dice score maintaining gradients
                intersection = (pred_region * target_region).sum()
                union = pred_region.sum() + target_region.sum()
                dice_score = (2. * intersection + 1e-6) / (union + 1e-6)  # Adding small epsilon for numerical stability
                dice_scores.append(dice_score)

            if dice_scores:
                sample_loss = 1.0 - torch.mean(torch.stack(dice_scores))
            else:
                # Handle edge case where there are no regions
                sample_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
            
            batch_losses.append(sample_loss)

        loss = torch.mean(torch.stack(batch_losses))
        return loss

class ClusterTverskyLoss(nn.Module):
    def __init__(self):
        super(ClusterTverskyLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, pred, target):
        if not pred.shape == target.shape:
            raise ValueError(f"Pred shape {pred.shape} must match target shape {target.shape}")

        batch_size = pred.shape[0]
        batch_losses = []

        for b in range(batch_size):
            pred_sample = pred[b].squeeze()
            target_sample = target[b].squeeze()

            # Ensure inputs require gradients
            pred_sample = pred_sample.requires_grad_(True)

            overlay = pred_sample + target_sample
            overlay = (overlay > 0).float()

            # Get regions without tracking gradients
            region_map, num_connected_components = get_regions(overlay)
            tversky_scores = []

            for region_label in range(1, num_connected_components + 1):
                region_mask = (region_map == region_label)
                pred_region = pred_sample[region_mask]
                target_region = target_sample[region_mask]

                # Compute Tversky score maintaining gradients
                intersection = (pred_region * target_region).sum()
                fp = (1 - target_region).sum()
                fn = (1 - pred_region).sum()
                tversky_score = (intersection + 1e-6) / (intersection + fp + fn + 1e-6)  # Adding small epsilon for numerical stability
                tversky_scores.append(tversky_score)

            if tversky_scores:
                sample_loss = 1.0 - torch.mean(torch.stack(tversky_scores))
            else:
                # Handle edge case where there are no regions
                sample_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
            
            batch_losses.append(sample_loss)

        loss = torch.mean(torch.stack(batch_losses))
        return loss

class blobDiceLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(blobDiceLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y):

        dice = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)
        
        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)

        for i in range(x.shape[0]):
            multi_label[i], _ = get_connected_components(y[i])
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"dice": {"name": "dice", "loss": dice, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"dice": {"name": "dice", "loss": dice, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

class blobTverskyLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(blobTverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y):

        tversky = MemoryEfficientTverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)
        
        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)

        for i in range(x.shape[0]):
            multi_label[i], _ = get_connected_components(y[i])
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"tversky": {"name": "tversky", "loss": tversky, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"tversky": {"name": "tversky", "loss": tversky, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss