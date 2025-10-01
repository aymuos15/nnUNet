"""
Blob-wise Dice Loss for nnU-Net

Based on: https://github.com/neuronflow/blob_loss
Paper: "A multi-hypothesis approach to color constancy" (adapted for segmentation)

Computes Dice loss per connected component (blob) in the ground truth, then averages.
This encourages the network to segment each individual instance accurately.
"""

import torch
from torch import nn
from typing import Optional, Tuple

# Optional GPU accelerated connected components
try:
    import cupy as cp  # type: ignore
    from cucim.skimage import measure as cucim_measure  # type: ignore
    _HAS_GPU_CC = True
except Exception:
    _HAS_GPU_CC = False

try:
    from scipy.ndimage import label as scipy_label  # type: ignore
except Exception:
    scipy_label = None


def _connected_components(binary_mask: torch.Tensor, connectivity: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """Return connected component labels for a (D,H,W) binary tensor.

    Tries GPU (CuPy+cuCIM) if available and tensor is on CUDA; falls back to scipy on CPU.
    Returns (labels_tensor, num_labels). Labels start at 1. Background is 0.
    """
    assert binary_mask.ndim == 3, "binary_mask must be 3D (D,H,W)"
    is_cuda = binary_mask.is_cuda

    if is_cuda and _HAS_GPU_CC:
        try:
            mask_cu = cp.asarray(binary_mask.detach().bool().cpu().numpy())
            labeled_img, num_features = cucim_measure.label(mask_cu, connectivity=connectivity, return_num=True)
            labeled_torch = torch.as_tensor(labeled_img.get(), device=binary_mask.device, dtype=torch.int32)
            return labeled_torch, int(num_features)
        except Exception:
            pass  # fallback to CPU

    # CPU fallback
    mask_np = binary_mask.detach().cpu().numpy().astype(bool)
    if scipy_label is None:
        raise RuntimeError("scipy.ndimage.label not available for connected components")
    labeled_np, num = scipy_label(mask_np)
    labeled_torch = torch.from_numpy(labeled_np).to(binary_mask.device, non_blocking=True)
    return labeled_torch, int(num)


def _dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft dice between pred and target (already masked to a blob)."""
    intersect = torch.sum(pred * target)
    denom = pred.sum() + target.sum()
    return (2. * intersect + eps) / (denom + eps)


class BlobDiceLoss(nn.Module):
    """
    Blob-wise Dice Loss for instance-aware segmentation.

    For each foreground class, computes connected components in GT and calculates
    Dice per blob. Final loss = -mean(blob dice scores).

    This encourages accurate per-instance segmentation rather than just foreground/background.

    Args:
        apply_nonlin: Optional nonlinearity (e.g., softmax, sigmoid)
        include_background: Whether to compute loss for background class
        smooth: Smoothing factor for dice computation
        batch_dice: If True, aggregate blob scores across batch before computing mean (more stable)
    """

    def __init__(
        self,
        apply_nonlin=None,
        include_background: bool = False,
        smooth: float = 1e-6,
        batch_dice: bool = False
    ):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.include_background = include_background
        self.smooth = smooth
        self.batch_dice = batch_dice

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Network output [B, C, D, H, W] (raw logits or probabilities)
            y: Ground truth labels [B, 1, D, H, W] or [B, D, H, W] (integer labels)

        Returns:
            Scalar loss value
        """
        spatial_dims = x.shape[2:]
        b, c = x.shape[:2]

        # Apply nonlinearity if specified
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Convert labels to one-hot if needed
        if y.ndim == x.ndim - 1:
            # [B, D, H, W] -> [B, 1, D, H, W]
            y_ = y.long().view(b, 1, *spatial_dims)
        elif y.ndim == x.ndim and y.shape[1] == 1:
            y_ = y.long()
        else:
            y_ = y.long()

        # Create one-hot encoding
        if y_.shape != x.shape:
            y_onehot = torch.zeros_like(x, dtype=torch.bool)
            y_onehot.scatter_(1, y_, True)
        else:
            y_onehot = y_.bool()

        # Determine which classes to process
        class_indices = list(range(c)) if self.include_background else list(range(1, c))

        if self.batch_dice:
            # Batch mode: aggregate all blob scores across entire batch
            all_blob_scores = []

            for bidx in range(b):
                for cls in class_indices:
                    gt_class = y_onehot[bidx, cls]  # [D, H, W]

                    # Skip if no foreground pixels
                    if not gt_class.any():
                        continue

                    # Find connected components (blobs) in ground truth
                    labels, num_blobs = _connected_components(gt_class)

                    if num_blobs == 0:
                        continue

                    pred_class = x[bidx, cls]  # [D, H, W]

                    # Compute dice per blob
                    for blob_id in range(1, num_blobs + 1):
                        blob_mask = (labels == blob_id)

                        # Extract predictions and targets for this blob
                        pred_blob = pred_class[blob_mask]
                        gt_blob = gt_class[blob_mask].float()

                        if pred_blob.numel() == 0:
                            continue

                        # Compute dice for this blob
                        dice = _dice_score(pred_blob, gt_blob, eps=self.smooth)
                        all_blob_scores.append(dice)

            # Mean across all blobs in batch, return negative
            if all_blob_scores:
                return -torch.mean(torch.stack(all_blob_scores))
            else:
                return x[0, 0].sum() * 0.0

        else:
            # Per-sample mode: compute dice per sample, then average
            batch_losses = []

            for bidx in range(b):
                sample_blob_scores = []

                for cls in class_indices:
                    gt_class = y_onehot[bidx, cls]  # [D, H, W]

                    # Skip if no foreground pixels
                    if not gt_class.any():
                        continue

                    # Find connected components (blobs) in ground truth
                    labels, num_blobs = _connected_components(gt_class)

                    if num_blobs == 0:
                        continue

                    pred_class = x[bidx, cls]  # [D, H, W]

                    # Compute dice per blob
                    for blob_id in range(1, num_blobs + 1):
                        blob_mask = (labels == blob_id)

                        # Extract predictions and targets for this blob
                        pred_blob = pred_class[blob_mask]
                        gt_blob = gt_class[blob_mask].float()

                        if pred_blob.numel() == 0:
                            continue

                        # Compute dice for this blob
                        dice = _dice_score(pred_blob, gt_blob, eps=self.smooth)
                        sample_blob_scores.append(dice)

                # Average dice across all blobs in this sample
                if sample_blob_scores:
                    sample_dice = torch.mean(torch.stack(sample_blob_scores))
                    batch_losses.append(-sample_dice)  # Return negative dice
                else:
                    # No foreground blobs - zero loss for this sample
                    batch_losses.append(x[bidx, 0].sum() * 0.0)

            # Average loss across batch
            return torch.mean(torch.stack(batch_losses))


class BlobDiceCELoss(nn.Module):
    """
    Combined Blob Dice + Cross-Entropy Loss.

    Blob Dice encourages per-instance accuracy, while CE provides stable gradients.

    Args:
        blob_weight: Weight for blob dice term (default: 1.0)
        ce_weight: Weight for cross-entropy term (default: 1.0)
        apply_nonlin: Nonlinearity for dice (CE handles its own softmax)
        include_background: Whether to include background in dice
        batch_dice: If True, aggregate blob scores across batch before computing mean (more stable)
    """

    def __init__(
        self,
        blob_weight: float = 1.0,
        ce_weight: float = 1.0,
        apply_nonlin=None,
        include_background: bool = False,
        batch_dice: bool = False
    ):
        super().__init__()
        self.blob_weight = blob_weight
        self.ce_weight = ce_weight

        self.blob_dice = BlobDiceLoss(
            apply_nonlin=apply_nonlin,
            include_background=include_background,
            batch_dice=batch_dice
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Network output [B, C, D, H, W] (raw logits)
            y: Ground truth [B, 1, D, H, W] or [B, D, H, W] (integer labels)
        """
        # Blob dice loss
        dice_loss = self.blob_dice(x, y)

        # Cross-entropy loss (expects [B, C, D, H, W] and [B, D, H, W])
        if y.ndim == x.ndim and y.shape[1] == 1:
            y_ce = y.squeeze(1)
        else:
            y_ce = y
        ce_loss = self.ce(x, y_ce.long())

        # Weighted combination
        return self.blob_weight * dice_loss + self.ce_weight * ce_loss
