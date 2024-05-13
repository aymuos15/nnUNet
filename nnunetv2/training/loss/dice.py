from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn

import cc3d
import numpy as np


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

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
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
    print()
    print('This is within the function')
    print()
    
    if axes is None:
        axes = tuple(range(2, net_output.ndim))
        print('This is axes')
        print(axes)
        print()

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    print('This is net_output')
    print(net_output.shape)
    print(torch.unique(net_output))
    print()

    print('This is y_onehot')
    print(y_onehot.shape)
    print(torch.unique(y_onehot))
    print()

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

def dice(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    intersection = np.logical_and(pred, gt)

    union = pred.sum() + gt.sum()

    if union == 0:
        return 1.0

    return 2. * intersection.sum() / union

def instance_scores(net_output, gt):

    #! No need to one hot the GT?
    
    gt = gt.squeeze(1)
    gt = gt.cpu().numpy()
    for batch_idx in range(gt.shape[0]): #0 is batch
            gt[batch_idx] = cc3d.connected_components(gt[batch_idx], connectivity=26)
    gt_label_cc = torch.tensor(gt)

    net_output = net_output.cpu().numpy()
    for i in range(net_output.shape[0]):
        for j in range(net_output.shape[1]):
            net_output[i][j] = cc3d.connected_components(net_output[i][j], connectivity=26)
    pred_label_cc = torch.tensor(net_output)

    num_lesions = torch.unique(gt_label_cc)
    num_lesions = len(num_lesions[num_lesions != 0])

    for batch_idx in range(gt_label_cc.shape[0]):
        for channel_idx in range(gt_label_cc.shape[1]):
            for volume_idx in range(gt_label_cc.shape[2]):

                tp = []
                # lesion_dice_scores = []

                for lesion_idx in range(num_lesions):
                    lesion_idx += 1

                    ## Extracting current lesion
                    gt_tmp = np.zeros_like(gt_label_cc)
                    gt_tmp[gt_label_cc == lesion_idx] = 1
                    
                    ## Extracting Predicted true positive lesions
                    pred_tmp = np.copy(pred_label_cc)
                    pred_tmp = pred_tmp*gt_tmp

                    intersecting_cc = np.unique(pred_tmp) 
                    intersecting_cc = intersecting_cc[intersecting_cc != 0] 

                    for cc in intersecting_cc:
                        tp.append(cc)

                #     ## Isolating Predited Lesions to calculate Metrics
                #     pred_tmp = np.copy(pred_label_cc)
                #     pred_tmp[np.isin(pred_tmp,intersecting_cc,invert=True)] = 0
                #     pred_tmp[np.isin(pred_tmp,intersecting_cc)] = 1

                #     dice_score = dice(pred_tmp, gt_tmp)
                #     # pprint(f'Lesion {lesion_idx} in Volume {volume_idx}, channel {channel_idx}, batch {batch_idx}: Dice = {dice_score}')
                #     lesion_dice_scores.append(dice_score)

                # fp = np.unique(pred_label_cc[np.isin(pred_label_cc,tp+[0],invert=True)])
                # lesion_dice = np.sum(lesion_dice_scores)/(len(lesion_dice_scores) + len(fp))

    print('This is the COUNT SCORE:', (num_lesions - len(tp)))
    # print('This is Lesion Dice:', lesion_dice)

    # return lesion_dice, (num_lesions - len(tp))
    return (num_lesions - len(tp))

if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
