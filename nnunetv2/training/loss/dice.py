from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn

import cc3d

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc

def dice(pred, gt):
    intersection = torch.sum(pred * gt)
    sum_pred = torch.sum(pred)
    sum_gt = torch.sum(gt)
    return 2.0 * intersection / (sum_pred + sum_gt)

def instance_scores(net_output, gt):
    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    for batch_idx in range(y_onehot.shape[0]):
        for channel_idx in range(y_onehot.shape[1]):
            lbl = y_onehot[batch_idx, channel_idx]
            lbl = lbl.cpu().numpy()
            components = cc3d.connected_components(lbl, connectivity=26)
            y = torch.tensor(components).to(y_onehot.device)
            y_onehot[batch_idx, channel_idx] = y
    
    for batch_idx in range(net_output.shape[0]):
        for channel_idx in range(net_output.shape[1]):
            pred = net_output[batch_idx, channel_idx]
            pred = pred.cpu().numpy()
            components = cc3d.connected_components(pred, connectivity=26)
            o = torch.tensor(components).to(net_output.device)
            net_output[batch_idx, channel_idx] = o
    
    total_dice_scores = torch.tensor([]).to(net_output.device)
    total_counts = torch.tensor([]).to(net_output.device)

    for batch_idx in range(y_onehot.shape[0]):
        for channel_idx in range(y_onehot.shape[1]):

            pred_cc_volume = net_output[batch_idx, channel_idx]
            gt_cc_volume = y_onehot[batch_idx, channel_idx]

            num_lesions = torch.unique(pred_cc_volume[pred_cc_volume != 0]).size(0)

            lesion_dice_scores = 0
            tp = torch.tensor([]).to(pred_cc_volume.device)

            for gtcomp in range(1, num_lesions + 1):
                
                gt_tmp = (gt_cc_volume == gtcomp)
                intersecting_cc = torch.unique(pred_cc_volume[gt_tmp])
                intersecting_cc = intersecting_cc[intersecting_cc != 0]

                if len(intersecting_cc) > 0:
                    pred_tmp = torch.zeros_like(pred_cc_volume, dtype=torch.bool)
                    pred_tmp[torch.isin(pred_cc_volume, intersecting_cc)] = True
                    dice_score = dice(pred_tmp, gt_tmp)
                    lesion_dice_scores += dice_score
                    tp = torch.cat([tp, intersecting_cc])
                else:
                    pass

            mask = (pred_cc_volume != 0) & (~torch.isin(pred_cc_volume, tp))
            fp = torch.unique(pred_cc_volume[mask], sorted=True).to(pred_cc_volume.device)
            fp = fp[fp != 0]

            if num_lesions + len(fp) > 0:
                volume_dice_score = lesion_dice_scores / (num_lesions + len(fp))
            else:
                # Handle the case where the denominator is zero, e.g., set volume_dice_score to 0 or handle it appropriately for your use case
                volume_dice_score = 0  # or any other appropriate value

            count = num_lesions - len(tp)

            volume_dice_score = torch.tensor([volume_dice_score]).to(net_output.device)
            count = torch.tensor([count]).to(net_output.device)

            total_dice_scores = torch.cat([total_dice_scores, volume_dice_score])
            total_counts = torch.cat([total_counts, count])

    total_dice_scores = total_dice_scores.mean()
    total_counts = total_counts.mean()

    return total_dice_scores, total_counts

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """

    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here

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
