import numpy as np
import torch

from nnunetv2.training.losses.implementations.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.losses.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.losses.implementations.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.trainer.network_config import _do_i_compile


def _build_loss(trainer_instance):
    """
    Build loss function for the given trainer instance.

    Args:
        trainer_instance: The nnUNetTrainer instance containing configuration and label manager

    Returns:
        Loss function (potentially wrapped with DeepSupervisionWrapper)
    """
    # Check if custom loss builder is provided in config
    if (hasattr(trainer_instance, 'trainer_config') and
        trainer_instance.trainer_config is not None and
        trainer_instance.trainer_config.loss_builder is not None):
        return trainer_instance.trainer_config.loss_builder(trainer_instance)

    # Default loss configuration
    if trainer_instance.label_manager.has_regions:
        loss = DC_and_BCE_loss({},
                               {'batch_dice': trainer_instance.configuration_manager.batch_dice,
                                'do_bg': True, 'smooth': 1e-5, 'ddp': trainer_instance.is_ddp},
                               use_ignore_label=trainer_instance.label_manager.ignore_label is not None,
                               dice_class=MemoryEfficientSoftDiceLoss)
    else:
        loss = DC_and_CE_loss({'batch_dice': trainer_instance.configuration_manager.batch_dice,
                               'smooth': 1e-5, 'do_bg': False, 'ddp': trainer_instance.is_ddp}, {}, weight_ce=1, weight_dice=1,
                              ignore_label=trainer_instance.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

    if _do_i_compile(trainer_instance):
        loss.dc = torch.compile(loss.dc)

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss

    if trainer_instance.enable_deep_supervision:
        deep_supervision_scales = _get_deep_supervision_scales(trainer_instance)
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        if trainer_instance.is_ddp and not _do_i_compile(trainer_instance):
            # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
            # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
            # Anywho, the simple fix is to set a very low weight to this.
            weights[-1] = 1e-6
        else:
            weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)

    return loss


def _get_deep_supervision_scales(trainer_instance):
    """
    Get deep supervision scales for the given trainer instance.

    Args:
        trainer_instance: The nnUNetTrainer instance containing configuration manager

    Returns:
        Deep supervision scales (list of lists) or None if deep supervision is disabled
    """
    if trainer_instance.enable_deep_supervision:
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            trainer_instance.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
    else:
        deep_supervision_scales = None  # for train and val_transforms
    return deep_supervision_scales