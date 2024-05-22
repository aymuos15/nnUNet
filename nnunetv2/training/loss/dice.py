from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn

import cc3d

import time

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

def dice(pred, gt):
    # Convert inputs to PyTorch tensors
    # pred = torch.as_tensor(pred, dtype=torch.bool)
    # gt = torch.as_tensor(gt, dtype=torch.bool)

    intersection = torch.sum(pred * gt)
    sum_pred = torch.sum(pred)
    sum_gt = torch.sum(gt)
    return 2.0 * intersection / (sum_pred + sum_gt)

def instance_scores(net_output, gt):
    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)

    ###########################################################################################
    # clone_start = time.time()

    pred_clone = net_output.clone()
    pred_label = pred_clone.cpu().numpy()
    for i in range(pred_label.shape[0]):
        for j in range(pred_label.shape[1]):
            # pred_label[i][j] = cc3d.connected_components(pred_label[i][j], connectivity=26)
            components = cc3d.connected_components(pred_label[i][j], connectivity=26)
            pred_label[i][j] = components
            print('No of lesions (pred):', torch.unique(torch.tensor(pred_label[i][j])))
    pred_label_cc = torch.tensor(pred_label)
    pred_label_cc = pred_label_cc.to(net_output.device)
    print('Total number of lesions (pred):', torch.unique(pred_label_cc))

    gt_clone = gt.clone()
    gt_label = gt_clone.cpu().numpy()
    for i in range(gt_label.shape[0]):
        for j in range(gt_label.shape[1]):
            # gt_label[i][j] = cc3d.connected_components(gt_label[i][j], connectivity=26)
            components = cc3d.connected_components(gt_label[i][j], connectivity=26)
            gt_label[i][j] = components
            print('No. of lesions (gt):', torch.unique(torch.tensor(gt_label[i][j])))
    gt_label_cc = torch.tensor(gt_label)
    gt_label_cc = gt_label_cc.to(net_output.device)
    print('Total number of lesions (gt):', torch.unique(gt_label_cc))

    num_gt_lesions = torch.unique(gt_label_cc[gt_label_cc != 0]).size(0)
    print('Number of lesions:', num_gt_lesions)

    total_dice_scores = torch.tensor([]).to(net_output.device)
    total_counts = torch.tensor([]).to(net_output.device)

    # clone_end = time.time()
    # print(f"Time taken for cloning: {clone_end - clone_start}")
    # print()
    #? It does not take much time at all. (0.0001s)
    ###########################################################################################
    

    #####################################################################################################################
    # dice_calculation_prep_start = time.time()

    for batch_idx in range(gt_label_cc.shape[0]):
        for channel_idx in range(gt_label_cc.shape[1]):
            for volume_idx in range(gt_label_cc.shape[2]):
                # print('For Volume:', volume_idx)
                # print()

                pred_cc_volume = pred_label_cc[batch_idx, channel_idx, volume_idx]
                gt_cc_volume = gt_label_cc[batch_idx, channel_idx, volume_idx]

                lesion_dice_scores = 0
                tp = torch.tensor([]).to(pred_cc_volume.device)
                # fn = torch.tensor([]).to(pred_cc_volume.device)

                # dice_calculation_prep_end = time.time()
                # print(f"Time taken for dice calculation prep: {dice_calculation_prep_end - dice_calculation_prep_start}")
                # print()

                #! Is this the bottleneck? 0.02s every iteration.
    #####################################################################################################################

    #####################################################################################################################
                # loop_start = time.time()

                for gtcomp in range(1, num_gt_lesions + 1):
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
                        # fn = torch.cat([fn, torch.tensor([gtcomp])])
                        pass
                
                # loop_end = time.time()
                # print(f"Time taken for loop: {loop_end - loop_start}")
                #? It does not take much time at all.
    #####################################################################################################################

    #####################################################################################################################
                # final_creation_start = time.time()

                mask = (pred_cc_volume != 0) & (~torch.isin(pred_cc_volume, tp)).to(pred_cc_volume.device)
                # mask = (pred_cc_volume != 0) & (~torch.isin(pred_cc_volume, tp))
                fp = torch.unique(pred_cc_volume[mask], sorted=True)
                fp = fp[fp != 0]

                volume_dice_score = lesion_dice_scores / (num_gt_lesions + len(fp))
                # print('Dice Score:', volume_dice_score)
                count = num_gt_lesions - len(tp)
                # print('Count:', count)

                volume_dice_score = torch.tensor([volume_dice_score]).to(net_output.device)
                count = torch.tensor([count]).to(net_output.device)

                total_dice_scores = torch.cat([total_dice_scores, volume_dice_score])
                total_counts = torch.cat([total_counts, count])

                # final_creation_end = time.time()
                # print(f"Time taken for final creation: {final_creation_end - final_creation_start}")
                # print()
                #? It does not take much time at all.
    #####################################################################################################################

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
            # if this is the case then gt is probably already a one hot encoding
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


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
