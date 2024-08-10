from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
import numpy as np

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

# def get_connected_components(img, connectivity=None):
#     img_cupy = cp.asarray(img.cpu().numpy())
#     labeled_img, num_features = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
#     labeled_img_torch = torch.tensor(labeled_img, device=img.device, dtype=torch.float32)
#     return labeled_img_torch, num_features


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

# def dice(pred, gt):
#     intersection = torch.sum(pred * gt)
#     sum_pred = torch.sum(pred)
#     sum_gt = torch.sum(gt)
#     return 2.0 * intersection / (sum_pred + sum_gt)

# def instance_scores(net_output, gt):

#     with torch.no_grad():
#         if net_output.ndim != gt.ndim:
#             gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

#         if net_output.shape == gt.shape:
            
#             y_onehot = gt
#         else:
#             y_onehot = torch.zeros(net_output.shape, device=net_output.device)
#             y_onehot.scatter_(1, gt.long(), 1)
        
#     for batch_idx in range(y_onehot.shape[0]):
#         for channel_idx in range(y_onehot.shape[1]):
#             lbl = y_onehot[batch_idx, channel_idx]
#             lbl = lbl.cpu().numpy()
#             components = cc3d.connected_components(lbl, connectivity=26)
#             y = torch.tensor(components.astype(np.uint8)).to(y_onehot.device)
#             y_onehot[batch_idx, channel_idx] = y
    
#     for batch_idx in range(net_output.shape[0]):
#         for channel_idx in range(1, net_output.shape[1]):
#             pred = net_output[batch_idx, channel_idx]
#             pred = pred.cpu().numpy()
#             components = cc3d.connected_components(pred, connectivity=26)
#             o = torch.tensor(components.astype(np.uint8)).to(net_output.device)
#             net_output[batch_idx, channel_idx] = o
    
#     total_dice_scores = torch.tensor([]).to(net_output.device)
#     total_counts = torch.tensor([]).to(net_output.device)

#     for batch_idx in range(net_output.shape[0]):
#         for channel_idx in range(1, net_output.shape[1]):

#             pred_cc_volume = net_output[batch_idx, channel_idx]
#             gt_cc_volume = y_onehot[batch_idx, channel_idx]

#             num_lesions = torch.unique(pred_cc_volume[pred_cc_volume != 0]).size(0)

#             lesion_dice_scores = 0
#             tp = torch.tensor([]).to(pred_cc_volume.device)

#             for gtcomp in range(1, num_lesions + 1):
                
#                 gt_tmp = (gt_cc_volume == gtcomp)
#                 intersecting_cc = torch.unique(pred_cc_volume[gt_tmp])
#                 intersecting_cc = intersecting_cc[intersecting_cc != 0]

#                 if len(intersecting_cc) > 0:
#                     pred_tmp = torch.zeros_like(pred_cc_volume, dtype=torch.bool)
#                     pred_tmp[torch.isin(pred_cc_volume, intersecting_cc)] = True
#                     dice_score = dice(pred_tmp, gt_tmp)
#                     lesion_dice_scores += dice_score
#                     tp = torch.cat([tp, intersecting_cc])
#                 else:
#                     lesion_dice_scores += 0

#             mask = (pred_cc_volume != 0) & (~torch.isin(pred_cc_volume, tp))
#             fp = torch.unique(pred_cc_volume[mask], sorted=True).to(pred_cc_volume.device)
#             fp = fp[fp != 0]

#             if num_lesions + len(fp) > 0:
#                 volume_dice_score = lesion_dice_scores / (num_lesions + len(fp))
#             else:
#                 # Handle the case where the denominator is zero, e.g., set volume_dice_score to 0 or handle it appropriately for your use case
#                 volume_dice_score = 0  # or any other appropriate value
            
#             count = num_lesions - len(tp)

#             volume_dice_score = torch.tensor([volume_dice_score]).to(net_output.device)
#             count = torch.tensor([count]).to(net_output.device)

#             total_dice_scores = torch.cat([total_dice_scores, volume_dice_score])
#             total_counts = torch.cat([total_counts, count])

#     total_dice_scores = total_dice_scores.mean()
#     total_counts = total_counts.mean()

#     return total_dice_scores, total_counts

# import pandas as pd
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def dice_torch(im1, im2):
#     intersection = torch.sum(im1 * im2)
#     sum_im1 = torch.sum(im1)
#     sum_im2 = torch.sum(im2)
#     return 2.0 * intersection / (sum_im1 + sum_im2)

# def collect_legacy_metrics(pred_label_cc, gt_label_cc):
#     legacy_metrics = []
#     tp = torch.tensor([], device=device)
#     fn = torch.tensor([], device=device)

#     num_gt_lesions = torch.unique(gt_label_cc[gt_label_cc != 0]).size(0)

#     for gtcomp in range(1, num_gt_lesions + 1):
#         gt_tmp = (gt_label_cc == gtcomp)
#         intersecting_cc = torch.unique(pred_label_cc[gt_tmp])
#         intersecting_cc = intersecting_cc[intersecting_cc != 0]

#         for cc in intersecting_cc:
#             tp = torch.cat([tp, torch.tensor([cc], device=device)])
#             legacy_metrics.append({'GT': gtcomp, 'Pred': cc.item(), 'Dice': dice_torch(pred_label_cc == cc, gt_tmp)})

#         if len(intersecting_cc) == 0:
#             legacy_metrics.append({'GT': gtcomp, 'Pred': 0, 'Dice': 0})
#             fn = torch.cat([fn, torch.tensor([gtcomp], device=device)])

#     zero_tensor = torch.tensor([0], device=device)
#     fp = torch.unique(pred_label_cc[torch.isin(pred_label_cc, torch.cat((tp, zero_tensor)), invert=True)])
#     fp = fp[fp != 0]
#     return legacy_metrics, tp, fp, fn

# def find_overlapping_components(prediction_cc, gt_cc):
#     overlapping_components = {}
#     overlapping_components_inverse = {}

#     # Iterate over all non-zero elements in the prediction_cc tensor
#     for i, j, k in zip(*torch.nonzero(prediction_cc, as_tuple=True)):
#         prediction_component = prediction_cc[i, j, k].item()
#         gt_component = gt_cc[i, j, k].item()

#         if prediction_component != 0 and gt_component != 0:
#             if prediction_component not in overlapping_components:
#                 overlapping_components[prediction_component] = set()
#             overlapping_components[prediction_component].add(gt_component)

#             if gt_component not in overlapping_components_inverse:
#                 overlapping_components_inverse[gt_component] = set()
#             overlapping_components_inverse[gt_component].add(prediction_component)

#     # Filter out entries with only one overlapping component
#     overlapping_components = {k: v for k, v in overlapping_components.items() if len(v) > 1}
#     overlapping_components_inverse = {k: v for k, v in overlapping_components_inverse.items() if len(v) > 1}

#     return overlapping_components, overlapping_components_inverse

# def generate_overlap_metrics(pred_label_cc, gt_label_cc, overlapping_components):
#     overlap_metrics = []

#     for pred_component, gt_components in overlapping_components.items():
#         gtcomps = list(gt_components)
#         pred_cc_tmp = (pred_label_cc == pred_component).to(torch.int32)
#         gt_cc_tmp = (gt_label_cc[..., None] == torch.tensor(gtcomps, device=gt_label_cc.device)).any(-1).to(torch.int32)
#         overlap_metrics.append({'GT': gtcomps, 'Pred': pred_component, 'Dice': dice_torch(pred_cc_tmp, gt_cc_tmp)})

#     return overlap_metrics

# def generate_overlap_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components):
#     overlap_metrics = []

#     for gt_component, pred_components in overlapping_components.items():
#         predcomps = list(pred_components)
#         gt_cc_tmp = (gt_label_cc == gt_component).to(torch.int32)
#         pred_cc_tmp = (pred_label_cc[..., None] == torch.tensor(predcomps, device=pred_label_cc.device)).any(-1).to(torch.int32)
#         overlap_metrics.append({'GT': gt_component, 'Pred': predcomps, 'Dice': dice_torch(pred_cc_tmp, gt_cc_tmp)})

#     return overlap_metrics

# def collect_all_metrics(pred_label_cc, gt_label_cc, overlapping_components, overlapping_components_inverse):
#     legacy_metrics, tp, fp, fn = collect_legacy_metrics(pred_label_cc, gt_label_cc)
#     legacy_metrics = pd.DataFrame(legacy_metrics)
    
#     overlap_metrics = generate_overlap_metrics(pred_label_cc, gt_label_cc, overlapping_components)
#     overlap_metrics = pd.DataFrame(overlap_metrics)
    
#     overlap_metrics_inverse = generate_overlap_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components_inverse)
#     overlap_metrics_inverse = pd.DataFrame(overlap_metrics_inverse)

    
#     initial_metrics_df = pd.concat([legacy_metrics, overlap_metrics, overlap_metrics_inverse], ignore_index=True)
#     return initial_metrics_df, tp, fp, fn

# def process_metric_df(df):
#     if df.empty:
#         return []
#     else:
#         gt_list = []
#         pred_list = []
#         for gt, pred in zip(df['GT'], df['Pred']):
#             if isinstance(gt, list):
#                 gt_list.extend(gt)
#             if isinstance(pred, list):
#                 pred_list.extend(pred)
#         combined = set(gt_list + pred_list)
#         indices_to_drop = []
#         # if statement to chheck if ground truth is even present in the combined set
#         for idx, (gt, pred) in enumerate(zip(df['GT'], df['Pred'])):
#             if isinstance(gt, int) and gt in combined and isinstance(pred, int):
#                 indices_to_drop.append(idx)
#         df.drop(indices_to_drop, inplace=True)
#         df['GT'] = df['GT'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
#         df['Pred'] = df['Pred'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
#         df.drop_duplicates(subset=['GT', 'Pred'], inplace=True)
#         return df['Dice'].to_list()

# def instance_scoresv2(net_output, gt):

#     with torch.no_grad():
#         if net_output.ndim != gt.ndim:
#             gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

#         if net_output.shape == gt.shape:
            
#             y_onehot = gt
#         else:
#             y_onehot = torch.zeros(net_output.shape, device=net_output.device)
#             y_onehot.scatter_(1, gt.long(), 1)
        
#     for batch_idx in range(y_onehot.shape[0]):
#         for channel_idx in range(y_onehot.shape[1]):
#             lbl = y_onehot[batch_idx, channel_idx]
#             lbl = lbl.cpu().numpy()
#             components = cc3d.connected_components(lbl, connectivity=26)
#             y = torch.tensor(components.astype(np.uint8)).to(y_onehot.device)
#             y_onehot[batch_idx, channel_idx] = y
    
#     for batch_idx in range(net_output.shape[0]):
#         for channel_idx in range(1, net_output.shape[1]):
#             pred = net_output[batch_idx, channel_idx]
#             pred = pred.cpu().numpy()
#             components = cc3d.connected_components(pred, connectivity=26)
#             o = torch.tensor(components.astype(np.uint8)).to(net_output.device)
#             net_output[batch_idx, channel_idx] = o

#     total_dice_scores = torch.tensor([]).to(net_output.device)

#     for batch in range(y_onehot.shape[0]):
#         for channel in range(1, y_onehot.shape[1]):
#             pred_label_cc = net_output[batch, channel]
#             gt_label_cc = y_onehot[batch, channel]

#             overlapping_components, overlapping_components_inverse = find_overlapping_components(pred_label_cc, gt_label_cc)    
#             final_metric, tp, fp, fn = collect_all_metrics(pred_label_cc, gt_label_cc, overlapping_components, overlapping_components_inverse)
#             dice_score = process_metric_df(final_metric)

#             if len(dice_score) == 0:
#                 return torch.tensor([0]), torch.tensor([0])
            
#             else:
#                 final_score = sum(dice_score) / (len(dice_score) + len(fp))
#                 total_dice_scores = torch.cat([total_dice_scores, torch.tensor([final_score]).to(net_output.device)])
    
#     final_dice_score = total_dice_scores.mean()
        
#     return final_dice_score, torch.tensor([0])

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
