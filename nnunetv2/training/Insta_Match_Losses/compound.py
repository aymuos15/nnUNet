import torch
import torch.nn as nn

from nnunetv2.training.Insta_Match_Losses.instance import RegionDiceLoss, RegionTverskyLoss, ClusterDiceLoss, ClusterTverskyLoss
from nnunetv2.training.Insta_Match_Losses.distribution import RobustCrossEntropyLoss, RCELoss

class RegionDice_CELoss(nn.Module):
    def __init__(self, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=RegionDiceLoss):

        super(RegionDice_CELoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = RegionDiceLoss()

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
        ce_loss = self.ce()(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class RegionTversky_CELoss(nn.Module):
    def __init__(self, ce_kwargs, weight_ce=1, weight_tversky=1, ignore_label=None,
                 tversky_class=RegionTverskyLoss):

        super(RegionTversky_CELoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tversky = RegionTverskyLoss()

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
            target_tversky = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_tversky = target
            mask = None

        tversky_loss = self.tversky(net_output, target_tversky, loss_mask=mask) \
            if self.weight_tversky != 0 else 0
        ce_loss = self.ce()(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_tversky * tversky_loss
        return result
    
class RegionTversky_RCELoss(nn.Module):
    def __init__(self, ce_kwargs, weight_ce=1, weight_tversky=1, ignore_label=None,
                 tversky_class=RegionTverskyLoss):

        super(RegionTversky_RCELoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RCELoss(**ce_kwargs)
        self.tversky = RegionTverskyLoss()

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
            target_tversky = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_tversky = target
            mask = None

        tversky_loss = self.tversky(net_output, target_tversky, loss_mask=mask) \
            if self.weight_tversky != 0 else 0
        ce_loss = self.ce()(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_tversky * tversky_loss
        return result

class ClusterDice_CELoss(nn.Module):
    def __init__(self, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=ClusterDiceLoss):

        super(ClusterDice_CELoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = ClusterDiceLoss()

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
        ce_loss = self.ce()(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class ClusterTversky_CELoss(nn.Module):
    def __init__(self, ce_kwargs, weight_ce=1, weight_tversky=1, ignore_label=None,
                 tversky_class=ClusterTverskyLoss):

        super(ClusterTversky_CELoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tversky = ClusterTverskyLoss()

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
            target_tversky = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_tversky = target
            mask = None

        tversky_loss = self.tversky(net_output, target_tversky, loss_mask=mask) \
            if self.weight_tversky != 0 else 0
        ce_loss = self.ce()(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_tversky * tversky_loss
        return result

class RegionClusterTversky_CELoss(nn.Module):
        def __init__(self, ce_kwargs, weight_ce=1, weight_tversky1=1, weight_tversky2=1, ignore_label=None,
                 tversky_class=ClusterTverskyLoss):

            super(RegionClusterTversky_CELoss, self).__init__()
            if ignore_label is not None:
                ce_kwargs['ignore_index'] = ignore_label

            self.weight_tversky1 = weight_tversky1
            self.weight_tversky2 = weight_tversky2
            self.weight_ce = weight_ce
            self.ignore_label = ignore_label

            self.ce = RobustCrossEntropyLoss(**ce_kwargs)
            self.tversky1 = ClusterTverskyLoss()
            self.tversky2 = RegionTverskyLoss()

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
                target_tversky = torch.where(mask, target, 0)
                num_fg = mask.sum()
            else:
                target_tversky = target
                mask = None

            tversky1_loss = self.tversky1(net_output, target_tversky, loss_mask=mask) \
                if self.weight_tversky1 != 0 else 0
            tversky2_loss = self.tversky2(net_output, target_tversky, loss_mask=mask) \
                if self.weight_tversky2 != 0 else 0
            ce_loss = self.ce()(net_output, target[:, 0]) \
                if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

            result = self.weight_ce * ce_loss + self.weight_tversky1 * tversky1_loss + self.weight_tversky2 * tversky2_loss
            return result