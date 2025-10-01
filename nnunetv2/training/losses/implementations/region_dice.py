import torch
from torch import nn
from typing import Optional, Tuple

# Optional GPU accelerated connected components via CuPy + cuCIM
try:
    import cupy as cp  # type: ignore
    from cucim.skimage import measure as cucim_measure  # type: ignore
    _HAS_GPU_CC = True
except Exception:  # pragma: no cover - fallback path
    _HAS_GPU_CC = False

try:
    from scipy.ndimage import label as scipy_label  # type: ignore
except Exception:  # pragma: no cover
    scipy_label = None


def _connected_components(binary_mask: torch.Tensor, connectivity: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """Return connected component labels for a (D,H,W) binary tensor.
    Tries GPU (CuPy+cuCIM) if available and the tensor is on CUDA; falls back to scipy on CPU.
    Returns (labels_tensor, num_labels). Labels start at 1. Background is 0.
    """
    assert binary_mask.ndim == 3, "binary_mask must be 3D (D,H,W)"
    is_cuda = binary_mask.is_cuda
    if is_cuda and _HAS_GPU_CC:
        try:
            mask_cu = cp.asarray(binary_mask.detach().bool().cpu().numpy())  # move through host (cuCIM needs CuPy array)
            labeled_img, num_features = cucim_measure.label(mask_cu, connectivity=connectivity, return_num=True)
            labeled_torch = torch.as_tensor(labeled_img.get(), device=binary_mask.device, dtype=torch.int32)
            return labeled_torch, int(num_features)
        except Exception:
            pass  # fallback
    # CPU fallback
    mask_np = binary_mask.detach().cpu().numpy().astype(bool)
    if scipy_label is None:
        # Minimal manual implementation (single pass) if scipy not available
        # But scipy is a dependency of nnU-Net, so raise if missing
        raise RuntimeError("scipy.ndimage.label not available for connected components")
    labeled_np, num = scipy_label(mask_np)
    labeled_torch = torch.from_numpy(labeled_np).to(binary_mask.device, non_blocking=True)
    return labeled_torch, int(num)


def _dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft dice between two 1D tensors (already masked to a region).
    pred and target expected float.
    """
    intersect = torch.sum(pred * target)
    denom = pred.sum() + target.sum()
    return (2. * intersect + eps) / (denom + eps)


class RegionDiceLoss(nn.Module):
    """Region-based Dice Loss supporting both multiclass (softmax) and region-based (multi-label) nnU-Net modes.

    Modes:
        has_regions = False (standard multiclass): network outputs C channels (background + (C-1) foreground). We may
            exclude background via include_background.
        has_regions = True: network outputs one channel per foreground region (no explicit background). Target may have
            an additional ignore channel (if has_ignore=True) located at y[:, -1]. Those voxels are excluded.

    For each applicable channel, we find connected components in the GT channel (excluding ignore voxels) and compute
    a Dice score per component. Per-channel Dice = mean over its components. Per-sample Dice = mean over channels that
    had at least one component. Final loss is mean over batch of (1 - per-sample Dice).

    Empty samples (no channels with components) yield zero loss (no gradient for that sample).
    """
    def __init__(self, apply_nonlin=None, include_background: bool = False, has_regions: bool = False, has_ignore: bool = False):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.include_background = include_background
        self.has_regions = has_regions
        self.has_ignore = has_ignore

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        spatial_dims = x.shape[2:]
        b, c = x.shape[:2]

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if self.has_regions:
            if y.ndim != x.ndim:
                raise RuntimeError("RegionDiceLoss (has_regions=True) expects target one-hot with same ndim as x")
            if self.has_ignore and y.shape[1] == c + 1:
                ignore_channel = y[:, -1].bool()
                y_work = y[:, :-1]
            else:
                ignore_channel = None
                y_work = y
            if y_work.shape[1] != c:
                raise RuntimeError(f"RegionDiceLoss: expected {c} region channels, got {y_work.shape[1]}")
            y_onehot = y_work.bool()
        else:
            # Multiclass label map -> convert if needed
            if y.ndim == x.ndim - 1:
                y_ = y.long().view(b, 1, *spatial_dims)
            elif y.ndim == x.ndim and y.shape[1] == 1:
                y_ = y.long()
            else:
                y_ = y.long()
            if y_.shape != x.shape:
                y_onehot = torch.zeros_like(x, dtype=torch.bool)
                y_onehot.scatter_(1, y_, True)
            else:
                y_onehot = y_.bool()
            ignore_channel = None

        if self.has_regions:
            class_indices = list(range(c))
        else:
            class_indices = list(range(c)) if self.include_background else list(range(1, c))

        losses = []
        for bidx in range(b):
            per_sample_scores = []
            ignore_mask = ignore_channel[bidx] if (ignore_channel is not None) else None
            for cls in class_indices:
                gt_class = y_onehot[bidx, cls]
                if ignore_mask is not None:
                    gt_class = gt_class & (~ignore_mask)
                if not gt_class.any():
                    continue
                labels, num = _connected_components(gt_class)
                if num == 0:
                    continue
                pred_class = x[bidx, cls]
                region_scores = []
                for region_label in range(1, num + 1):
                    region_mask = (labels == region_label)
                    if ignore_mask is not None:
                        region_mask = region_mask & (~ignore_mask)
                    pred_vals = pred_class[region_mask]
                    if pred_vals.numel() == 0:
                        continue
                    gt_vals = gt_class[region_mask].float()
                    score = _dice_score(pred_vals, gt_vals)
                    region_scores.append(score)
                if region_scores:
                    per_sample_scores.append(torch.mean(torch.stack(region_scores)))
            if per_sample_scores:
                losses.append(1 - torch.mean(torch.stack(per_sample_scores)))
            else:
                losses.append(x[bidx, 0].sum() * 0)
        return torch.mean(torch.stack(losses))
