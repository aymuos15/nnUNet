"""
Dice metric computation for nnU-Net.

This module provides functions for computing true positives (TP), false positives (FP),
false negatives (FN), and true negatives (TN) from network outputs and ground truth labels.
These metrics are used during training validation and can be used during inference.
"""

import torch


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    Compute TP, FP, FN, TN from network output and ground truth.

    Args:
        net_output: Network output, must be (b, c, x, y(, z))
        gt: Ground truth label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z)))
            or one hot encoding (b, c, x, y(, z))
        axes: Axes to sum over. Can be (, ) = no summation. If None, defaults to spatial axes.
        mask: Mask must be 1 for valid pixels and 0 for invalid pixels.
              If provided, must have shape (b, 1, x, y(, z))
        square: If True then fp, tp and fn will be squared before summation

    Returns:
        Tuple of (tp, fp, fn, tn) as torch tensors
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn
