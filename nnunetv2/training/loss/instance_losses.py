from typing import Callable

import torch
from torch import nn, Tensor
# from torch.nn import functional as F

import numpy as np

# import cc3d
from scipy.ndimage import distance_transform_edt

from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.utilities.ddp_allgather import AllGatherGrad

from nnunetv2.training.loss.blob_helper import compute_loss
from nnunetv2.training.loss.region_helper import RegionLoss

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

#? ###########
#? Base Losses
#? ###########

"""
1. Dice x
2. Tversky x
3. Cross-Entropy (CE) x
4. Top-K x
5. Region-CE (RCE) x
6. Blob-Dice (bDice) x
7. Blob-Tversky (bTversky) x
"""
# 8. Cluster-Dice (cDice)

############
''' Dice '''
############
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

###############
''' Tversky '''
###############
class MemoryEfficientTverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientTverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

        self.fp = 0.8
        self.fn = 0.2

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

        numerator = intersect
        denominator = intersect + self.fp * sum_pred + self.fn * sum_gt
        tversky_index = numerator / (denominator + self.smooth)
        loss = 1 - tversky_index
        return loss.mean()

##########################
''' Cross-Entropy (CE) '''
##########################
class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

#############
''' Top-K '''
#############
class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target.unsqueeze(1)
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

######################
''' Region-CE (RCE)'''
######################
class RCELoss(RegionLoss):
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        if len(labels.shape) == len(inputs.shape):
            assert labels.shape[1] == 1
            labels = labels[:, 0]
        labels = labels.long()

        ce = RobustCrossEntropyLoss()
        loss_ce = ce(inputs, labels)

        gt_proportion, valid_mask = self.get_gt_proportion(self.mode, labels, inputs.shape)
        pred_proportion = self.get_pred_proportion(self.mode, inputs, temp=self.temp, valid_mask=valid_mask)
        loss_reg = (pred_proportion - gt_proportion).abs().mean()

        loss = loss_ce + self.alpha * loss_reg

        return loss

#########################
''' blob-Dice (bDice) '''
#########################
class bDiceLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(bDiceLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

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

################
''' bTversky '''
################
class bTverskyLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(bTverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        tversky = MemoryEfficientTverskyLoss(fp=0.8, fn=0.2, apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)
        
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

#! Not working - How do I differentiate between clusters?
# ############################
# ''' Cluster-Dice (cDice) '''
# ############################
# class cDiceLoss(nn.Module):
#     def forward(self, x, y, loss_mask=None):

#         overlay = x + y
#         overlay[overlay > 0] = 1

#         labelled_overlay = torch.zeros_like(overlay)

#         overlay = overlay.to(x.device)
#         labelled_overlay = labelled_overlay.to(x.device)

#         overlay = overlay.detach().cpu().numpy()
#         for batch in range(x.shape[0]):
#             for channel in range(x.shape[1]):

#                 labelled_overlay[batch, channel], _ = get_connected_components(overlay[batch, channel])
                        
#         num_clusters = torch.unique(labelled_overlay[labelled_overlay != 0]).size(0)

#         score_tally = torch.tensor([]).to(x.device)

#         for cluster in range(1, num_clusters + 1):
#             cluster_mask = (labelled_overlay == cluster).float()

#             pred_cluster = torch.logical_and(x, cluster_mask).to(x.device)
#             gt_cluster = torch.logical_and(y, cluster_mask).to(x.device)

#             score = dice(pred_cluster, gt_cluster)
#             score_tally = torch.cat((score_tally, score.unsqueeze(0)))
        
#         return torch.mean(score_tally)
    
#? ###############
#? Compound Losses
#? ###############

"""
# 1. Dice + CE
2. Dice + RCE
# 3. Dice + TopK

4. bDice + CE
5. bDice + RCE
6. bDice + TopK

7. Tversky + CE
8. Tversky + RCE
9. Tversky + TopK

10. bTversky + CE
11. bTversky + RCE
12. bTversky + TopK
"""

# 13. cDice + CE
# 14. cDice + RCE
# 15. cDice + TopK

# 16. cDice-CE
# 17. cDice-RCE
# 18. cDice-TopK

# 19. cDice + CE
# 20. cDice + RCE
# 21. cDice + TopK

# 22. cDice + bTversky + CE
# 23. cDice + bTversky + RCE


##################
''' Dice + RCE '''
##################
class Dice_RCELoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):

        super(Dice_RCELoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        # ce_loss = self.ce(net_output, target[:, 0]) \
        #     if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        ce_loss = RCELoss()(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

###################
''' Dice + TopK '''
###################
class Dice_TopkLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):

        super(Dice_TopkLoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        # ce_loss = self.ce(net_output, target[:, 0]) \
        #     if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        ce_loss = TopKLoss()(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

##################
''' bDice + CE '''
##################
class bDice_CELoss(nn.Module):
    def __init__(self, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(bDice_CELoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

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
            blob_criterion_dict={"ce": {"name": "ce", "loss": RobustCrossEntropyLoss(), "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

###################
''' bDice + RCE '''
###################
class bDice_RCELoss(nn.Module):
    def __init__(self, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(bDice_RCELoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

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
            blob_criterion_dict={"rce": {"name": "rce", "loss": RCELoss(), "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

####################
''' bDice + TopK '''
####################
class bDice_TopKLoss(nn.Module):
    def __init__(self, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(bDice_TopKLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

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
            blob_criterion_dict={"topk": {"name": "topk", "loss": TopKLoss(), "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

####################
''' Tversky + CE '''
####################
class Tversky_CELoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientTverskyLoss):

        super(Tversky_CELoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

#####################
''' Tversky + RCE '''
#####################
class Tversky_RCELoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientTverskyLoss):

        super(Tversky_RCELoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = RCELoss()(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

######################
''' Tversky + TopK '''
######################
class Tversky_TopKLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientTverskyLoss):

        super(Tversky_TopKLoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = TopKLoss()(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

#####################
''' bTversky + CE '''
#####################
class bTversky_CELoss(nn.Module):
    def __init__(self, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(bTversky_CELoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        tversky = MemoryEfficientTverskyLoss(fp=0.8, fn=0.2, apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)
        
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
            blob_criterion_dict={"ce": {"name": "ce", "loss": RobustCrossEntropyLoss(), "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

######################
''' bTversky + RCE '''
######################
class bTversky_RCELoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(bTversky_RCELoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        tversky = MemoryEfficientTverskyLoss(fp=0.8, fn=0.2, apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)
        
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
            blob_criterion_dict={"rce": {"name": "rce", "loss": RCELoss(), "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

#######################
''' bTversky + TopK '''
#######################
class bTversky_TopKLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(bTversky_TopKLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        tversky = MemoryEfficientTverskyLoss(fp=0.8, fn=0.2, apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)
        
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
            blob_criterion_dict={"topk": {"name": "topk", "loss": TopKLoss(), "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

class RegionDiceLoss(nn.Module):
    def __init__(self):
        super(RegionDiceLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_regions(self, gt):
        # Step 1: Connected Components for 3D volume
        labeled_array, num_features = self.get_connected_components(gt.cpu())

        # Step 2: Compute distance transform for each 3D region
        distance_map = torch.zeros_like(gt, dtype=torch.float32)
        region_map = torch.zeros_like(gt, dtype=torch.long)

        for region_label in range(1, num_features + 1):
            region_mask = (labeled_array == region_label)

            # Convert to numpy for distance transform
            region_mask_np = region_mask.cpu().numpy()
            distance = torch.from_numpy(
                distance_transform_edt(~region_mask_np)
            ).to(self.device)

            if region_label == 1 or distance_map.max() == 0:
                distance_map = distance
                region_map = region_label * torch.ones_like(gt, dtype=torch.long)
            else:
                update_mask = distance < distance_map
                distance_map[update_mask] = distance[update_mask]
                region_map[update_mask] = region_label

        return region_map, num_features

    def forward(self, x, y):

        """
        Args:
            pred (torch.Tensor): Predicted segmentation mask (B, C, D, H, W)
            target (torch.Tensor): Ground truth segmentation mask (B, C, D, H, W)
        Returns:
            torch.Tensor: Region-based Dice loss for 3D volumes
        """

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

            region_map, num_features = self.get_regions(y_volume)

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

# ##################
# ''' cDice + CE '''
# ##################
# class cDice_CELoss(nn.Module):
#     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_cdice=2, ignore_label=None,
#                  dice_class=MemoryEfficientSoftDiceLoss): #! weight_cdice=2 because want to mimic blob loss
#         """
#         Weights for CE and Dice do not need to sum to one. You can set whatever you want.
#         :param soft_dice_kwargs:
#         :param ce_kwargs:
#         :param aggregate:
#         :param square_dice:
#         :param weight_ce:
#         :param weight_cdice:
#         """
#         super(Dice_RCELoss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_cdice = weight_cdice
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):
#         """
#         target must be b, c, x, y(, z) with c=1
#         :param net_output:
#         :param target:
#         :return:
#         """
#         if self.ignore_label is not None:
#             assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
#                                          '(DC_and_CE_loss)'
#             mask = target != self.ignore_label
#             # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
#             # ignore gradients in those areas anyway
#             target_dice = torch.where(mask, target, 0)
#             num_fg = mask.sum()
#         else:
#             target_dice = target
#             mask = None

#         dc_loss = cDiceLoss()(net_output, target_dice, loss_mask=mask) \
#             if self.weight_cdice != 0 else 0
#         # ce_loss = self.ce(net_output, target[:, 0]) \
#         #     if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
#         ce_loss = RCELoss()(net_output, target[:, 0]) \
#             if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

#         result = self.weight_ce * ce_loss + self.weight_cdice * dc_loss
#         return result

# ###################
# ''' cDice + RCE '''
# ###################
# class cDice_RCELoss(nn.Module):
#     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_cdice=2, ignore_label=None,
#                  dice_class=MemoryEfficientSoftDiceLoss): #! weight_cdice=2 because want to mimic blob loss
#         """
#         Weights for CE and Dice do not need to sum to one. You can set whatever you want.
#         :param soft_dice_kwargs:
#         :param ce_kwargs:
#         :param aggregate:
#         :param square_dice:
#         :param weight_ce:
#         :param weight_cdice:
#         """
#         super(Dice_RCELoss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_cdice = weight_cdice
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):
#         """
#         target must be b, c, x, y(, z) with c=1
#         :param net_output:
#         :param target:
#         :return:
#         """
#         if self.ignore_label is not None:
#             assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
#                                          '(DC_and_CE_loss)'
#             mask = target != self.ignore_label
#             # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
#             # ignore gradients in those areas anyway
#             target_dice = torch.where(mask, target, 0)
#             num_fg = mask.sum()
#         else:
#             target_dice = target
#             mask = None

#         dc_loss = cDiceLoss()(net_output, target_dice, loss_mask=mask) \
#             if self.weight_cdice != 0 else 0
#         # ce_loss = self.ce(net_output, target[:, 0]) \
#         #     if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
#         ce_loss = RCELoss()(net_output, target[:, 0]) \
#             if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

#         result = self.weight_ce * ce_loss + self.weight
#         return result

# ####################
# ''' cDice + TopK '''
# ####################
# class cDice_TopKLoss(nn.Module):
#     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_cdice=2, ignore_label=None,
#                  dice_class=MemoryEfficientSoftDiceLoss): #! weight_cdice=2 because want to mimic blob loss
#         """
#         Weights for CE and Dice do not need to sum to one. You can set whatever you want.
#         :param soft_dice_kwargs:
#         :param ce_kwargs:
#         :param aggregate:
#         :param square_dice:
#         :param weight_ce:
#         :param weight_cdice:
#         """
#         super(Dice_RCELoss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_cdice = weight_cdice
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):
#         """
#         target must be b, c, x, y(, z) with c=1
#         :param net_output:
#         :param target:
#         :return:
#         """
#         if self.ignore_label is not None:
#             assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
#                                          '(DC_and_CE_loss)'
#             mask = target != self.ignore_label
#             # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
#             # ignore gradients in those areas anyway
#             target_dice = torch.where(mask, target, 0)
#             num_fg = mask.sum()
#         else:
#             target_dice = target
#             mask = None

#         dc_loss = cDiceLoss()(net_output, target_dice, loss_mask=mask) \
#             if self.weight_cdice != 0 else 0
#         # ce_loss = self.ce(net_output, target[:, 0]) \
#         #     if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
#         ce_loss = RCELoss()(net_output, target[:, 0]) \
#             if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

#         result = self.weight_ce * ce_loss + self.weight_cd
#         return

# #####################
# ''' cDice-CE Loss '''
# #####################
# class cDiceCELoss(nn.Module):
#     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
#                  dice_class=MemoryEfficientSoftDiceLoss):

#         super(cDiceCELoss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_dice = weight_dice
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)
#         self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         overlay = x + y
#         overlay[overlay > 0] = 1

#         labelled_overlay, num_clusters = get_connected_components(overlay)

#         dice_scores = torch.tensor([])
#         ce_losses = torch.tensor([])

#         for cluster in range(1, num_clusters + 1):
#             cluster_mask = (labelled_overlay == cluster).float()

#             pred_cluster = torch.logical_and(x, cluster_mask)
#             gt_cluster = torch.logical_and(y, cluster_mask)

#             dice_score = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)(pred_cluster, gt_cluster)
#             dice_scores = torch.cat((dice_scores, dice_score.unsqueeze(0)))

#             ce_loss = self.ce(pred_cluster, gt_cluster)
#             ce_losses = torch.cat((ce_losses, ce_loss.unsqueeze(0)))
        
#         average_dice_score = torch.mean(dice_scores) if dice_scores.numel() > 0 else torch.tensor(0.0).to(x.device)
#         average_ce_loss = torch.mean(ce_losses) if ce_losses.numel() > 0 else torch.tensor(0.0).to(x.device)

#         total_loss = self.weight_ce * average_ce_loss + self.weight_dice * average_dice_score

#         return total_loss


# ######################
# ''' cDice-RCE Loss '''
# ######################
# class cDiceRCELoss(nn.Module):
#     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
#                  dice_class=MemoryEfficientSoftDiceLoss):

#         super(cDice_RCELoss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_dice = weight_dice
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = RCELoss()
#         self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

#     def forward(self, x, y, loss_mask=None):

#         overlay = x + y
#         overlay[overlay > 0] = 1

#         labelled_overlay, num_clusters = get_connected_components(overlay)

#         # Initialize lists to hold Dice scores and Region-CE losses for each cluster
#         dice_scores = torch.tensor([])
#         rce_losses = torch.tensor([])

#         # Compute the Region-CE Loss and Dice score for each cluster
#         for cluster in range(1, num_clusters + 1):
#             cluster_mask = (labelled_overlay == cluster).float()

#             pred_cluster = torch.logical_and(x, cluster_mask)
#             gt_cluster = torch.logical_and(y, cluster_mask)

#             # Compute Dice score for the cluster
#             dice_score = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)(pred_cluster, gt_cluster)
#             dice_scores = torch.cat((dice_scores, dice_score.unsqueeze(0)))

#             # Compute Region-CE Loss for the cluster
#             ce_loss = self.ce(pred_cluster, gt_cluster)
#             rce_losses = torch.cat((rce_losses, ce_loss.unsqueeze(0)))

#         # Compute average Dice score and Region-CE loss
#         average_dice_score = torch.mean(dice_scores) if dice_scores.numel() > 0 else torch.tensor(0.0).to(x.device)
#         average_rce_loss = torch.mean(rce_losses) if rce_losses.numel() > 0 else torch.tensor(0.0).to(x.device)

#         # Combine the losses
#         total_loss = self.weight_ce * average_rce_loss + self.weight_dice * average_dice_score

#         return total_loss

# #######################
# ''' cDice-TopK Loss '''
# #######################
# class cDiceTopKLoss(nn.Module):
#     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
#                  dice_class=MemoryEfficientSoftDiceLoss):

#         super(cDiceTopKLoss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_dice = weight_dice
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = TopKLoss(**ce_kwargs)
#         self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         overlay = x + y
#         overlay[overlay > 0] = 1

#         labelled_overlay, num_clusters = get_connected_components(overlay)

#         dice_scores = torch.tensor([])
#         ce_losses = torch.tensor([])

#         for cluster in range(1, num_clusters + 1):
#             cluster_mask = (labelled_overlay == cluster).float()

#             pred_cluster = torch.logical_and(x, cluster_mask)
#             gt_cluster = torch.logical_and(y, cluster_mask)

#             dice_score = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)(pred_cluster, gt_cluster)
#             dice_scores = torch.cat((dice_scores, dice_score.unsqueeze(0)))

#             ce_loss = self.ce(pred_cluster, gt_cluster)
#             ce_losses = torch.cat((ce_losses, ce_loss.unsqueeze(0)))
        
#         average_dice_score = torch.mean(dice_scores) if dice_scores.numel() > 0 else torch.tensor(0.0).to(x.device)
#         average_ce_loss = torch.mean(ce_losses) if ce_losses.numel() > 0 else torch.tensor(0.0).to(x.device)

#         total_loss = self.weight_ce * average_ce_loss + self.weight_dice * average_dice_score

#         return total_loss

# #############################
# ''' cDice + bTversky + CE'''
# #############################
# class cDice_bTversky_CELoss(nn.Module):
#     def __init__(self, ce_kwargs, weight_ce=1, weight_cdice=2, weight_blob=2, ignore_label=None): #! weight_cdice=2 because want to mimic blob loss

#         super(Dice_RCELoss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_cdice = weight_cdice
#         self.weight_blob = weight_blob
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)
#         self.dc = cDiceLoss()

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):

#         if self.ignore_label is not None:
#             assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
#                                          '(DC_and_CE_loss)'
#             mask = target != self.ignore_label
#             # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
#             # ignore gradients in those areas anyway
#             target_dice = torch.where(mask, target, 0)
#             num_fg = mask.sum()
#         else:
#             target_dice = target
#             mask = None

#         dc_loss = cDiceLoss()(net_output, target_dice, loss_mask=mask) \
#             if self.weight_cdice != 0 else 0
#         blob_loss = bTverskyLoss()(net_output, target[:, 0]) \
#             if self.weight_blob != 0 and (self.ignore_label is None or num_fg > 0) else 0
#         ce_loss = self.ce(net_output, target[:, 0]) \
#             if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

#         result = self.weight_ce * ce_loss + self.weight_cdice * dc_loss + self.weight_blob * blob_loss
#         return result

# #############################
# ''' cDice + bTversky + RCE'''
# #############################
# class cDice_bTversky_RCELoss(nn.Module):
#     def __init__(self, ce_kwargs, weight_ce=1, weight_cdice=2, weight_blob=2, ignore_label=None): #! weight_cdice=2 because want to mimic blob loss

#         super(Dice_RCELoss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_cdice = weight_cdice
#         self.weight_blob = weight_blob
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = RCELoss()
#         self.dc = cDiceLoss()

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):

#         if self.ignore_label is not None:
#             assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
#                                          '(DC_and_CE_loss)'
#             mask = target != self.ignore_label
#             # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
#             # ignore gradients in those areas anyway
#             target_dice = torch.where(mask, target, 0)
#             num_fg = mask.sum()
#         else:
#             target_dice = target
#             mask = None

#         dc_loss = cDiceLoss()(net_output, target_dice, loss_mask=mask) \
#             if self.weight_cdice != 0 else 0
#         blob_loss = bTverskyLoss()(net_output, target[:, 0]) \
#             if self.weight_blob != 0 and (self.ignore_label is None or num_fg > 0) else 0
#         ce_loss = self.ce(net_output, target[:, 0]) \
#             if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

#         result = self.weight_ce * ce_loss + self.weight_cdice * dc_loss + self.weight_blob * blob_loss
#         return result