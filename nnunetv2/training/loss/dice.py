from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
import numpy as np
from scipy.optimize import linear_sum_assignment

import cc3d

#########################################################################################################
import cupy as cp
from cucim.skimage import measure as cucim_measure

#! GPU connected components
def get_connected_components(img, connectivity=None):

    img_cupy = cp.asarray(img)
    labeled_img, num_features = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
    labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)

    return labeled_img_torch, num_features

#! Dice
# Define the optimized function
def dice(im1, im2):
    
    # Ensure the tensors are on the same device
    im1 = im1.to(im2.device)
    im2 = im2.to(im1.device)

    # Compute Dice coefficient using optimized operations
    intersection = torch.sum(im1 * im2)
    im1_sum = torch.sum(im1)
    im2_sum = torch.sum(im2)
    
    dice_score = (2. * intersection) / (im1_sum + im2_sum)

    return dice_score
#########################################################################################################


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

def cluster_scores(net_output, gt):
    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)
        
    overlay = net_output + y_onehot
    overlay[overlay > 0] = 1

    labelled_overlay = torch.zeros_like(overlay)

    overlay = overlay.to(net_output.device)
    labelled_overlay = labelled_overlay.to(net_output.device)

    overlay = overlay.detach().cpu().numpy()
    for batch in range(net_output.shape[0]):
        for channel in range(net_output.shape[1]):
            components = cc3d.connected_components(overlay[batch, channel], connectivity=26)
            components = components.astype(np.uint8)
            labelled_overlay[batch, channel] = torch.tensor(components, device=net_output.device)

    num_clusters = torch.unique(labelled_overlay[labelled_overlay != 0]).size(0)

    score_tally = torch.tensor([]).to(net_output.device)

    for cluster in range(1, num_clusters + 1):
        cluster_mask = (labelled_overlay == cluster).float()

        pred_cluster = torch.logical_and(net_output, cluster_mask).to(net_output.device)
        gt_cluster = torch.logical_and(y_onehot, cluster_mask).to(net_output.device)

        score = dice(pred_cluster, gt_cluster)
        score_tally = torch.cat((score_tally, score.unsqueeze(0)))
    
    return torch.mean(score_tally), torch.tensor([0])

def panoptic_scores(net_output, gt):
    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)
    
    labelled_pred = torch.zeros_like(net_output)
    labelled_gt = torch.zeros_like(y_onehot)

    for batch in range(net_output.shape[0]):
        for channel in range(net_output.shape[1]):
            components = cc3d.connected_components(net_output[batch, channel].cpu().numpy(), connectivity=26)
            components = components.astype(np.uint8)
            labelled_pred[batch, channel] = torch.tensor(components, device=net_output.device)
    
    for batch in range(y_onehot.shape[0]):
        for channel in range(y_onehot.shape[1]):
            components = cc3d.connected_components(y_onehot[batch, channel].cpu().numpy(), connectivity=26)
            components = components.astype(np.uint8)
            labelled_gt[batch, channel] = torch.tensor(components, device=net_output.device)

    pred_label_cc = net_output
    gt_label_cc = y_onehot

    # num_gt_labels = torch.unique(gt_label_cc).size(0) - 1  # Exclude background (0)

    matches = create_match_dict(pred_label_cc, gt_label_cc)
    match_data = get_all_matches(matches)

    fp = sum(1 for pred, gt, _ in match_data if gt is None)
    fn = sum(1 for pred, gt, _ in match_data if pred is None)

    optimal_matches = optimal_matching(match_data)

    tp = len(optimal_matches)
    
    if tp == 0:
        return 0.0

    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = sum(score for _, _, score in optimal_matches) / tp

    # print('TP:', tp, 'FP:', fp, 'FN:', fn, 'num_gt_labels:', num_gt_labels)

    return rq * sq, tp, fp, fn

def create_match_dict(pred_label_cc, gt_label_cc):
    pred_to_gt = {}
    gt_to_pred = {}
    dice_scores = {}

    pred_labels = torch.unique(pred_label_cc)[1:]  # Exclude background (0)
    gt_labels = torch.unique(gt_label_cc)[1:]  # Exclude background (0)

    pred_masks = {label.item(): pred_label_cc == label for label in pred_labels}
    gt_masks = {label.item(): gt_label_cc == label for label in gt_labels}

    for pred_item, pred_mask in pred_masks.items():
        for gt_item, gt_mask in gt_masks.items():
            if torch.any(torch.logical_and(pred_mask, gt_mask)):
                pred_to_gt.setdefault(pred_item, []).append(gt_item)
                gt_to_pred.setdefault(gt_item, []).append(pred_item)
                dice_scores[(pred_item, gt_item)] = dice(pred_mask, gt_mask)

    for gt_item in gt_labels:
        gt_to_pred.setdefault(gt_item.item(), [])
    for pred_item in pred_labels:
        pred_to_gt.setdefault(pred_item.item(), [])

    return {"pred_to_gt": pred_to_gt, "gt_to_pred": gt_to_pred, "dice_scores": dice_scores}

def get_all_matches(matches):
    match_data = []

    for gt, preds in matches["gt_to_pred"].items():
        if not preds:
            match_data.append((None, gt, 0.0))
        else:
            for pred in preds:
                dice_score = matches["dice_scores"].get((pred, gt), 0.0)
                match_data.append((pred, gt, dice_score))

    for pred, gts in matches["pred_to_gt"].items():
        if not gts:
            match_data.append((pred, None, 0.0))

    return match_data

def optimal_matching(match_data):
    predictions = set()
    ground_truths = set()
    valid_matches = []

    for pred, gt, score in match_data:
        if pred is not None and gt is not None:
            predictions.add(pred)
            ground_truths.add(gt)
            valid_matches.append((pred, gt, score))

    pred_to_index = {pred: i for i, pred in enumerate(predictions)}
    gt_to_index = {gt: i for i, gt in enumerate(ground_truths)}

    cost_matrix = torch.ones((len(predictions), len(ground_truths)))

    for pred, gt, score in valid_matches:
        i, j = pred_to_index[pred], gt_to_index[gt]
        cost_matrix[i, j] = 1 - score

    #todo: Use a torch variant here?
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

    optimal_matches = []
    for i, j in zip(row_ind, col_ind):
        pred = list(predictions)[i]
        gt = list(ground_truths)[j]
        score = 1 - cost_matrix[i, j].item()
        optimal_matches.append((pred, gt, score))

    return optimal_matches
    

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
    
    # print('Dice Score nnuent function:', (2 * tp) / (2 * tp + fp + fn))
    # Output: Dice Score nnuent function: tensor([0.9943, 0.0000], device='cuda:0')

    return tp, fp, fn, tn
