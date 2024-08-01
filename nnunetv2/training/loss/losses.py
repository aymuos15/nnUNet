from typing import Callable

import torch
from torch import nn, Tensor
import numpy as np

from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.training.loss.blob_helper import compute_loss
from nnunetv2.training.loss.region_helper import RegionLoss
# from blob_helper import compute_loss
# from region_helper import RegionLoss
# print("Warnings filtered out")


import cc3d

import warnings
warnings.filterwarnings("ignore")

##################################### Base Losses #####################################

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
    def __init__(self, fp, fn, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
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

        self.fp = 0.3
        self.fn = 0.7

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

#####################
''' Cross-Entropy '''
#####################
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

######################################## Combined Losses #########################################
################
''' Dice__CE '''
################
class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):

        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        target = target.unsqueeze(1)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
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
''' Blob_Dice__CE '''
#####################
class blobDice_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(blobDice_and_CE_loss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        dice = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)
        ce = RobustCrossEntropyLoss()
        
        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"ce": {"name": "ce", "loss": ce, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"dice": {"name": "dice", "loss": dice, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

###########################
''' Blob_DiceCE__DiceCE '''
###########################
class blobDiceCE_and_DiceCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(blobDiceCE_and_DiceCE_loss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        Loss = DC_and_CE_loss({}, {}, weight_ce=1, weight_dice=1, dice_class=MemoryEfficientSoftDiceLoss)
        
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"DiceCE": {"name": "dice", "loss": Loss, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"DiceCE": {"name": "dice", "loss": Loss, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

################
''' DC__TopK '''
################
class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        target = target.unsqueeze(1)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
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

#######################
''' Blob_Dice__TopK '''
#######################
class blobDice_TopK_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(blobDice_TopK_loss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.dice = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)
        self.topk = TopKLoss(**ce_kwargs)

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):
        
        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"topk": {"name": "dice", "loss": self.topk, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"dice": {"name": "dice", "loss": self.dice, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

###############################
''' Blob_DiceTopK__DiceTopK '''
###############################
class blobDiceTopK__DiceTopK_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, alpha=2, beta=1, 
                 ignore_label=None):
        
        super(blobDiceTopK__DiceTopK_loss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        Loss = DC_and_topk_loss({}, {}, weight_ce=1, weight_dice=1)
        
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"DiceTopK": {"name": "dice", "loss": Loss, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"DiceTopK": {"name": "dice", "loss": Loss, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

###################
''' Tversky__CE '''
###################
class Tversky_and_CE_loss(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientTverskyLoss): # NOTICE TVERSKY LOSS !!!!

        super(Tversky_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.alpha = 0.3
        self.beta = 0.7

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tversky = MemoryEfficientTverskyLoss(self.alpha, self.beta, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        target = target.unsqueeze(1)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(tversky_and_CE_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        tversky_loss = self.tversky(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * tversky_loss
        return result

###########################
''' Blob_Tversky__CE '''
###########################
class blobTversky_and_CE_loss(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                dice_class=MemoryEfficientTverskyLoss): # NOTICE TVERSKY LOSS !!!!

        super(blobTversky_and_CE_loss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tversky = MemoryEfficientTverskyLoss(self.alpha, self.beta, **soft_dice_kwargs)

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"ce": {"name": "ce", "loss": self.ce, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"tversky": {"name": "tversky", "loss": self.tversky, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

#################################
''' Blob_TverskyCE__TverskyCE '''
#################################
class blobTverskyCE_and_TverskyCE_loss(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                dice_class=MemoryEfficientTverskyLoss): # NOTICE TVERSKY LOSS !!!!

        super(blobTverskyCE_and_TverskyCE_loss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tversky = MemoryEfficientTverskyLoss(self.alpha, self.beta, **soft_dice_kwargs)

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        Loss = Tversky_and_CE_loss(soft_dice_kwargs={}, ce_kwargs={}, alpha=0.3, beta=0.7)
        
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"TverskyCE": {"name": "tversky", "loss": Loss, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"TverskyCE": {"name": "tversky", "loss": Loss, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

#####################
''' Tversky__TopK '''
#####################
class Tversky_and_TopK_loss(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientTverskyLoss): # NOTICE TVERSKY LOSS !!!!

        super(Tversky_and_TopK_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.alpha = 0.3
        self.beta = 0.7

        self.ce = TopKLoss(**ce_kwargs)
        self.tversky = MemoryEfficientTverskyLoss(self.alpha, self.beta, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        target = target.unsqueeze(1)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(tversky_and_CE_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        tversky_loss = self.tversky(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        topk_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * topk_loss + self.weight_dice * tversky_loss
        return result

##########################
''' Blob_Tversky__TopK '''
##########################
class blobTversky__TopK(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                dice_class=MemoryEfficientTverskyLoss): # NOTICE TVERSKY LOSS !!!!

        super(blobTversky__TopK, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ce = TopKLoss(**ce_kwargs)
        self.tversky = MemoryEfficientTverskyLoss(self.alpha, self.beta, **soft_dice_kwargs)

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"topk": {"name": "tversky", "loss": self.ce, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"tversky": {"name": "tversky", "loss": self.tversky, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

#####################################
''' Blob_TverskyTopK__TverskyTopK '''
#####################################
class blobTverskyTopK__TverskyTopK_loss(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                dice_class=MemoryEfficientTverskyLoss): # NOTICE TVERSKY LOSS !!!!

        super(blobTverskyTopK__TverskyTopK_loss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        Loss = Tversky_and_TopK_loss(soft_dice_kwargs={}, ce_kwargs={}, alpha=0.3, beta=0.7)
        
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"TverskyCE": {"name": "tversky", "loss": Loss, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"TverskyCE": {"name": "tversky", "loss": Loss, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

#################################
''' Blob_Tversky__TverskyTopK '''
#################################
class blobTversky__TverskyTopK_loss(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                dice_class=MemoryEfficientTverskyLoss): # NOTICE TVERSKY LOSS !!!!

        super(blobTversky__TverskyTopK_loss, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ce = TopKLoss(**ce_kwargs)
        self.tversky = MemoryEfficientTverskyLoss(self.alpha, self.beta, **soft_dice_kwargs)

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"tversky": {"name": "tversky", "loss": self.ce, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"tversky": {"name": "tversky", "loss": self.tversky, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

#################
''' Region CE '''
#################
class Region_CE(RegionLoss):
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

########################
''' Dice + Region CE '''
########################
class DC_and_RegionCE(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):

        super(DC_and_RegionCE, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.alpha = 0.3
        self.beta = 0.7

        self.ce = Region_CE()
        self.tversky = MemoryEfficientSoftDiceLoss(self.alpha, self.beta, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        target = target.unsqueeze(1)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(tversky_and_CE_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        region_ce_loss = self.tversky(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        dice_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * dice_loss + self.weight_dice * region_ce_loss
        return result

############################
''' Blob_Dice__Region_CE '''
############################
class BlobDice__RegionCE(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):

        super(BlobDice__RegionCE, self).__init__()

        self.alpha = alpha
        self.beta = beta

        self.ce = Region_CE()
        self.dice = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=True, smooth=0, ddp=False)

        self.ignore_label = ignore_label

    def forward(self, x, y, loss_mask=None):

        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)
        
        loss, _, _ = compute_loss(
            blob_loss_dict={"main_weight": self.alpha, "blob_weight": self.beta},
            criterion_dict={"ce": {"name": "ce", "loss": self.ce, "weight": 0.5, "sigmoid": False}},
            blob_criterion_dict={"dice": {"name": "dice", "loss": self.dice, "weight": 0.5, "sigmoid": False}},
            raw_network_outputs=x,
            binary_label=y,
            multi_label=multi_label,
        )

        return loss

# #############################################
# ''' Dice+CE-RegionConstraintSoumya + Dice '''
# #############################################
# class Constrained__DC_and_CE_loss(nn.Module):
#     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
#                  dice_class=MemoryEfficientSoftDiceLoss):

#         super(Constrained__DC_and_CE_loss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_dice = weight_dice
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)
#         self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):

#         count_constraint = get_count_constraint(net_output, target)

#         # print("Count Constraint: ", count_constraint)
#         # print(target.shape)

#         # target = target.unsqueeze(1)
#         # print(target.shape)

#         if self.ignore_label is not None:
#             assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
#                                          '(DC_and_CE_loss)'
#             mask = target != self.ignore_label
#             target_dice = torch.where(mask, target, 0)
#             num_fg = mask.sum()
#         else:
#             target_dice = target
#             mask = None
        
#         dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
#             if self.weight_dice != 0 else 0
#         ce_loss = self.ce(net_output, target[:, 0]) \
#             if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

#         # result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
#         result = count_constraint * ce_loss + self.weight_dice * dc_loss
#         return result

#####################
''' InstanceDice  '''
#####################
class InstanceDiceLoss(nn.Module):
    def __init__(self, ignore_label=None):
        super(InstanceDiceLoss, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask=None):
        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        x_label_cc = torch.zeros_like(x)
        for i in range(x.shape[0]):
            x_label_cc = x_label_cc.detach().cpu().numpy()
            x_label_cc[i] = cc3d.connected_components(x[i].detach().cpu().numpy(), connectivity=26)
            x_label_cc = torch.tensor(x_label_cc)
        x_label_cc = x_label_cc.to(y.device)

        num_gt_lesions = torch.unique(multi_label[multi_label != 0]).size(0)

        lesion_dice_scores = 0
        tp = torch.tensor([], device=y.device)

        for gtcomp in range(1, num_gt_lesions + 1):
            gt_tmp = (multi_label == gtcomp)
            intersecting_cc = torch.unique(x_label_cc[gt_tmp])
            intersecting_cc = intersecting_cc[intersecting_cc != 0]

            if len(intersecting_cc) > 0:
                pred_tmp = torch.zeros_like(x_label_cc, dtype=torch.float32, requires_grad=True)
                pred_tmp = torch.where(torch.isin(x_label_cc, intersecting_cc), torch.tensor(1., device=y.device), pred_tmp)
                dice_score = self.dice(pred_tmp, gt_tmp)
                lesion_dice_scores += dice_score
                tp = torch.cat([tp, intersecting_cc])
            else:
                pass
            
        mask = (x_label_cc != 0) & (~torch.isin(x_label_cc, tp))
        fp = torch.unique(x_label_cc[mask], sorted=True)
        fp = fp[fp != 0]

        #count = Number of GT Lesions - Number of Predicted Lesions

        return lesion_dice_scores / (num_gt_lesions + len(fp))

##########################
''' InstanceDice  + CE '''
##########################
class InstanceDice_and_CE(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):

        super(InstanceDice_and_CE, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss()
        self.instancedice = InstanceDiceLoss()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        target = target.unsqueeze(1)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(instancedice_and_CE_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        instance_dice = self.instancedice(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        normal_ce = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * normal_ce + self.weight_dice * instance_dice
        return result


###############
''' Counter '''
###############
def instance_count(x: torch.Tensor, y: torch.Tensor):
    y = y.squeeze(1)
    multi_label = torch.zeros_like(y)
    for i in range(x.shape[0]):
        multi_label = multi_label.detach().cpu().numpy()
        multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
        multi_label = torch.tensor(multi_label)
    multi_label = multi_label.unsqueeze(1)
    y = y.unsqueeze(1)

    x = x.to(y.device)
    x_label_cc = torch.zeros_like(x)
    for i in range(x.shape[0]):
        x_label_cc = x_label_cc.detach().cpu().numpy()
        x_label_cc[i] = cc3d.connected_components(x[i].detach().cpu().numpy(), connectivity=26)
        x_label_cc = torch.tensor(x_label_cc)
    x_label_cc = x_label_cc.to(y.device)

    num_gt_lesions = torch.unique(multi_label[multi_label != 0]).size(0)
    num_pred_lesions = torch.unique(x_label_cc[x_label_cc != 0]).size(0)

    #count = Number of GT Lesions - Number of Predicted Lesions
    count = num_gt_lesions - num_pred_lesions

    return count

# ########################
# ''' Counter + DiceCE '''
# ########################
# class DC_and_CE_Countloss(nn.Module):
#     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
#                  dice_class=MemoryEfficientSoftDiceLoss):

#         super(DC_and_CE_Countloss, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label

#         self.weight_dice = weight_dice
#         self.weight_ce = weight_ce
#         self.ignore_label = ignore_label

#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)
#         self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):

#         target = target.unsqueeze(1)

#         if self.ignore_label is not None:
#             assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
#                                          '(DC_and_CE_loss)'
#             mask = target != self.ignore_label
#             target_dice = torch.where(mask, target, 0)
#             num_fg = mask.sum()
#         else:
#             target_dice = target
#             mask = None

#         dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
#             if self.weight_dice != 0 else 0
#         ce_loss = self.ce(net_output, target[:, 0]) \
#             if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

#         count = instance_count(net_output, target)


#         result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
#         return result

#########################
''' New InstanceDice  '''
#########################
# def get_connected_components(img, connectivity=None):
#     img_cupy = cp.asarray(img)
#     labeled_img, num_features = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
#     labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)
#     return labeled_img_torch, num_features

def dice(im1, im2):
    intersection = torch.sum(im1 * im2)
    sum_im1 = torch.sum(im1)
    sum_im2 = torch.sum(im2)
    return 2.0 * intersection / (sum_im1 + sum_im2)

def collect_legacy_metrics(pred_label_cc, gt_label_cc):
    legacy_metrics = []
    tp = torch.tensor([], device=device)
    intersection_counts = {}

    for gtcomp in range(1, torch.max(gt_label_cc) + 1):
        gt_tmp = gt_label_cc == gtcomp
        intersecting_cc = torch.unique(pred_label_cc[gt_tmp])
        intersecting_cc = intersecting_cc[intersecting_cc != 0]

        for cc in intersecting_cc:
            cc = cc.item()
            intersection_counts[(gtcomp, cc)] = intersection_counts.get((gtcomp, cc), 0) + 1

        if len(intersecting_cc) == 0:
            legacy_metrics.append({'GT': gtcomp, 'Pred': 0, 'Dice': 0})
        else:
            max_count = 0
            max_predcomp = None
            for predcomp in intersecting_cc:
                predcomp = predcomp.item()
                count = intersection_counts[(gtcomp, predcomp)]
                if count > max_count:
                    max_count = count
                    max_predcomp = predcomp

            pred_tmp = pred_label_cc == max_predcomp
            legacy_metrics.append({'GT': gtcomp, 'Pred': max_predcomp, 'Dice': dice(pred_tmp, gt_tmp)})
            # tp.append(max_predcomp)
            tp = torch.cat([tp, torch.tensor([max_predcomp], device=device)])

    zero_tensor = torch.tensor([0], device=device)
    fp = torch.unique(pred_label_cc[torch.isin(pred_label_cc, torch.cat((torch.tensor(tp), zero_tensor)), invert=True)])   
    return legacy_metrics, fp, tp

def find_overlapping_components(prediction_cc, gt_cc):
    overlapping_components = {}
    overlapping_components_inverse = {}
    
    # for i, j, k in zip(*prediction_cc.nonzero()):
    for i, j, k in zip(*torch.nonzero(prediction_cc, as_tuple=True)):
        prediction_component = prediction_cc[i, j, k]
        gt_component = gt_cc[i, j, k]
        if prediction_component != 0 and gt_component != 0:
            if prediction_component not in overlapping_components:
                overlapping_components[prediction_component] = set()
            overlapping_components[prediction_component].add(gt_component)
            if gt_component not in overlapping_components_inverse:
                overlapping_components_inverse[gt_component] = set()
            overlapping_components_inverse[gt_component].add(prediction_component)

    overlapping_components = {k: v for k, v in overlapping_components.items() if len(v) > 1}
    overlapping_components_inverse = {k: v for k, v in overlapping_components_inverse.items() if len(v) > 1}
    return overlapping_components, overlapping_components_inverse

def generate_overlap_metrics(pred_label_cc, gt_label_cc, overlapping_components):
    overlap_metrics = []
    for pred_components, gt_components in overlapping_components.items():
        gtcomps = list(gt_components)
        pred_cc_tmp = (pred_label_cc == pred_components).astype(int)
        gt_cc_tmp = (gt_label_cc[..., None] == gtcomps).any(-1).astype(int)
        overlap_metrics.append({'GT': gtcomps, 'Pred': pred_components, 'Dice': dice(pred_cc_tmp, gt_cc_tmp)})
    return overlap_metrics

def generate_overlap_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components):
    overlap_metrics = []
    for gt_components, pred_components in overlapping_components.items():
        predcomps = list(pred_components)
        gt_cc_tmp = (gt_label_cc == gt_components).astype(int)
        pred_cc_tmp = (pred_label_cc[..., None] == predcomps).any(-1).astype(int)
        overlap_metrics.append({'GT': gt_components, 'Pred': predcomps, 'Dice': dice(pred_cc_tmp, gt_cc_tmp)})
    return overlap_metrics

def collect_all_metrics(pred_label_cc, gt_label_cc, overlapping_components, overlapping_components_inverse):
    legacy_metrics, fp, tp = collect_legacy_metrics(pred_label_cc, gt_label_cc)
    legacy_metrics = pd.DataFrame(legacy_metrics)
    
    overlap_metrics = generate_overlap_metrics(pred_label_cc, gt_label_cc, overlapping_components)
    overlap_metrics = pd.DataFrame(overlap_metrics)
    
    overlap_metrics_inverse = generate_overlap_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components_inverse)
    overlap_metrics_inverse = pd.DataFrame(overlap_metrics_inverse)
    
    initial_metrics_df = pd.concat([legacy_metrics, overlap_metrics, overlap_metrics_inverse], ignore_index=True)
    return initial_metrics_df, fp, tp

def process_metric_df(df):
    gt_list = []
    pred_list = []
    for gt, pred in zip(df['GT'], df['Pred']):
        if isinstance(gt, list):
            gt_list.extend(gt)
        if isinstance(pred, list):
            pred_list.extend(pred)
    combined = set(gt_list + pred_list)
    indices_to_drop = []
    for idx, (gt, pred) in enumerate(zip(df['GT'], df['Pred'])):
        if isinstance(gt, int) and gt in combined and isinstance(pred, int):
            indices_to_drop.append(idx)
    df.drop(indices_to_drop, inplace=True)
    df['GT'] = df['GT'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    df['Pred'] = df['Pred'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    df.drop_duplicates(subset=['GT', 'Pred'], inplace=True)
    return df['Dice'].to_list()

# def proposed_metric(pred, gt, dimensions=3):

#     pred_label_cc, _ = get_connected_components(pred)
#     gt_label_cc, _ = get_connected_components(gt)
    
#     overlapping_components, overlapping_components_inverse = find_overlapping_components(pred_label_cc, gt_label_cc)    
#     final_metric, fp, tp = collect_all_metrics(pred_label_cc, gt_label_cc, overlapping_components, overlapping_components_inverse)
#     dice_score = process_metric_df(final_metric)
#     dice_score = sum(dice_score) / (len(dice_score) + len(fp))
#     return dice_score

class InstanceDiceLossv2(nn.Module):
    def __init__(self, ignore_label=None):
        super(InstanceDiceLossv2, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask=None):
        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label = multi_label.detach().cpu().numpy()
            multi_label[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
            multi_label = torch.tensor(multi_label)
        multi_label = multi_label.unsqueeze(1)
        # y = y.unsqueeze(1)

        x = x.to(y.device)
        x_label_cc = torch.zeros_like(x)
        for i in range(x.shape[0]):
            x_label_cc = x_label_cc.detach().cpu().numpy()
            x_label_cc[i] = cc3d.connected_components(x[i].detach().cpu().numpy(), connectivity=26)
            x_label_cc = torch.tensor(x_label_cc)
        x_label_cc = x_label_cc.to(y.device)

        overlapping_components, overlapping_components_inverse = find_overlapping_components(x_label_cc, multi_label)    
        final_metric, fp, tp = collect_all_metrics(x_label_cc, multi_label, overlapping_components, overlapping_components_inverse)
        dice_score = process_metric_df(final_metric)
        dice_score = sum(dice_score) / (len(dice_score) + len(fp))
        return dice_score
