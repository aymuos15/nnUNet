from typing import Callable

import torch
from torch import nn, Tensor
import numpy as np

from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
# from nnunetv2.training.loss.blob_helper import compute_loss
# from nnunetv2.training.loss.region_helper import RegionLoss
from blob_helper import compute_loss
from region_helper import RegionLoss

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
class blobTversky__TopK_loss(nn.Module):
    def __init__(self, alpha, beta, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                dice_class=MemoryEfficientTverskyLoss): # NOTICE TVERSKY LOSS !!!!

        super(blobTversky__TopK_loss, self).__init__()

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

#############################################
''' Dice+CE-RegionConstraintSoumya + Dice '''
#############################################
class Constrained__DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=MemoryEfficientSoftDiceLoss):

        super(Constrained__DC_and_CE_loss, self).__init__()
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
        
        count_constraint = get_count_constraint(net_output, target)

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        result = count_constraint * ce_loss + self.weight_dice * dc_loss
        return result

def get_count_constraint(pred, gt):

    y = gt
    x = pred

    # print("Pred: ", x.shape)
    # print("GT: ", y.shape)

    y = y.squeeze(1)
    gt_label_cc = torch.zeros_like(y)
    for i in range(x.shape[0]):
        gt_label_cc = gt_label_cc.detach().cpu().numpy()
        gt_label_cc[i] = cc3d.connected_components(y[i].detach().cpu().numpy(), connectivity=26)
        gt_label_cc = torch.tensor(gt_label_cc)
        
    gt_label_cc = gt_label_cc.unsqueeze(1)
    y = y.unsqueeze(1)

    # print("GT: ", gt_label_cc.shape)

    x = x.cpu().numpy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = cc3d.connected_components(x[i][j], connectivity=26)
    pred_label_cc = torch.tensor(x)

    tp = []

    num_gt_lesions = torch.unique(gt_label_cc)
    num_gt_lesions = len(num_gt_lesions[num_gt_lesions != 0])

    for gtcomp in range(num_gt_lesions):
        gtcomp += 1

        ## Extracting current lesion
        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gtcomp] = 1
        
        ## Extracting Predicted true positive lesions
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp = pred_tmp*gt_tmp

        intersecting_cc = np.unique(pred_tmp) 
        intersecting_cc = intersecting_cc[intersecting_cc != 0] 

        for cc in intersecting_cc:
            tp.append(cc)
    
    # print("TP: ", len(tp))
    # print("GT: ", num_gt_lesions)
    return num_gt_lesions - len(tp)

################### Test Losses ######################
pred = torch.zeros((2, 3, 32, 32, 32))
pred[0, 0, 10:20, 10:20, 10:20] = 1
ref = torch.zeros((2, 32, 32, 32))
ref[0, 10:20, 10:20, 10:20] = 1

new_dc_ce = Constrained__DC_and_CE_loss({}, {}, weight_ce=1, weight_dice=1, dice_class=MemoryEfficientSoftDiceLoss)
new_dc_ce_loss = new_dc_ce(pred, ref)
print(new_dc_ce_loss)

# region_ce = Region_CE()
# region_ce_loss = region_ce(pred, ref)
# print(region_ce_loss)

# blobDice__Region_CE = BlobDice__RegionCE(alpha=2, beta=1, soft_dice_kwargs={}, ce_kwargs={})
# blobDice__Region_CE_loss = blobDice__Region_CE(pred, ref)
# print(blobDice__Region_CE_loss)

# dice = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
# dice_loss = dice(pred, ref)

# tversky = MemoryEfficientTverskyLoss(fp=0.3, fn=0.7)
# tversky_loss = tversky(pred, ref)

# ce = RobustCrossEntropyLoss()
# ce_loss = ce(pred, ref)

# topk = TopKLoss()
# topk_loss = topk(pred, ref) # NOTE: ref is used for TopKLoss

# #################################################################################################

# dc_ce = DC_and_CE_loss({}, {}, weight_ce=1, weight_dice=1, dice_class=MemoryEfficientSoftDiceLoss)
# dc_ce_loss = dc_ce(pred, ref)

# blob_dice__ce = blobDice_and_CE_loss({}, {}, alpha=2, beta=1)
# blob_dice__ce_loss = blob_dice__ce(pred, ref)

# blob_dicece__dicece = blobDiceCE_and_DiceCE_loss({}, {}, alpha=2, beta=1)
# blob_dicece__dicece_loss = blob_dicece__dicece(pred, ref)

# dc_topk = DC_and_topk_loss({}, {}, weight_ce=1, weight_dice=1)
# dc_topk_loss = dc_topk(pred, ref)

# blob_dice__topk = blobDice_TopK_loss({}, {}, alpha=2, beta=1)
# blob_dice__topk_loss = blob_dice__topk(pred, ref)

# blob_dicetopk__dicetopk = blobDiceTopK__DiceTopK_loss({}, {}, alpha=2, beta=1)
# blob_dicetopk__dicetopk_loss = blob_dicetopk__dicetopk(pred, ref)

# tversky_ce = Tversky_and_CE_loss(soft_dice_kwargs={}, ce_kwargs={}, alpha=0.3, beta=0.7)
# tversky_ce_loss = tversky_ce(pred, ref)

# blob_tversky__ce = blobTversky_and_CE_loss(alpha=0.1, beta=0.9, soft_dice_kwargs={}, ce_kwargs={})
# blob_tversky__ce_loss = blob_tversky__ce(pred, ref)

# blob_tverskyce__tverskyce = blobTverskyCE_and_TverskyCE_loss(alpha=0.3, beta=0.7, soft_dice_kwargs={}, ce_kwargs={})
# blob_tverskyce__tverskyce_loss = blob_tverskyce__tverskyce(pred, ref)

# tversky_topk = Tversky_and_TopK_loss(soft_dice_kwargs={}, ce_kwargs={}, alpha=0.3, beta=0.7)
# tversky_topk_loss = tversky_topk(pred, ref)

# blob_tversky__ce = blobTversky__TopK_loss(soft_dice_kwargs={}, ce_kwargs={}, alpha=0.3, beta=0.7)
# blob_tversky__ce_loss = blob_tversky__ce(pred, ref)

# blob_tverskyce__tverskyce = blobTverskyTopK__TverskyTopK_loss(alpha=0.3, beta=0.7, soft_dice_kwargs={}, ce_kwargs={})
# blob_tverskyce__tverskyce_loss = blob_tverskyce__tverskyce(pred, ref)

# blob_tversky__tverskyce = blobTversky__TopK_loss(alpha=0.3, beta=0.7, soft_dice_kwargs={}, ce_kwargs={})
# blob_tversky__tverskyce_loss = blob_tversky__tverskyce(pred, ref)


# from itertools import count
# import pandas as pd

# # Define your losses and initialize a counter
# losses = [
#     ('Dice', dice_loss),
#     ('Tversky', tversky_loss),
#     ('CE', ce_loss),
#     ('TopK', topk_loss),

#     ('DC_CE', dc_ce_loss),
#     ('Blob_Dice__CE', blob_dice__ce_loss),
#     ('Blob_DiceCE__DiceCE', blob_dicece__dicece_loss),

#     ('DC_TopK', dc_topk_loss),
#     ('Blob_Dice__TopK', blob_dice__topk_loss),
#     ('Blob_DiceTopK__DiceTopK', blob_dicetopk__dicetopk_loss),

#     ('Tversky_CE', tversky_ce_loss),
#     ('Blob_Tversky__CE', blob_tversky__ce_loss),
#     ('Blob_TverskyCE__TverskyCE', blob_tverskyce__tverskyce_loss),

#     ('Tversky_TopK', tversky_topk_loss),
#     ('Blob_Tversky__TopK', blob_tversky__ce_loss),
#     ('Blob_TverskyCE__TverskyCE', blob_tverskyce__tverskyce_loss),

#     ('Blob_Tversky__TverskyTopK', blob_tversky__tverskyce_loss)
# ]

# counter = count(start=1)

# # Print serial number along with each loss
# data = [(next(counter), name, loss_value) for name, loss_value in losses]
# df = pd.DataFrame(data, columns=["Serial Number", "Loss Name", "Loss Value"])
# print(df)
