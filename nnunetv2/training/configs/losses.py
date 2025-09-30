"""Loss function configuration presets."""

import numpy as np
import torch

from nnunetv2.training.configs import TrainerConfig, register_config
from nnunetv2.training.losses.implementations.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.losses.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.losses.implementations.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.core.helpers import softmax_helper_dim1


def dice_loss_builder(trainer_instance):
    """Build Dice loss only (no CE/BCE)."""
    loss = MemoryEfficientSoftDiceLoss(
        **{
            'batch_dice': trainer_instance.configuration_manager.batch_dice,
            'do_bg': trainer_instance.label_manager.has_regions,
            'smooth': 1e-5,
            'ddp': trainer_instance.is_ddp
        },
        apply_nonlin=torch.sigmoid if trainer_instance.label_manager.has_regions else softmax_helper_dim1
    )

    if trainer_instance.enable_deep_supervision:
        deep_supervision_scales = trainer_instance._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)

    return loss


def dice_ce_no_smooth_builder(trainer_instance):
    """Build Dice+CE loss with smooth=0."""
    if trainer_instance.label_manager.has_regions:
        loss = DC_and_BCE_loss(
            {},
            {
                'batch_dice': trainer_instance.configuration_manager.batch_dice,
                'do_bg': True,
                'smooth': 0,
                'ddp': trainer_instance.is_ddp
            },
            use_ignore_label=trainer_instance.label_manager.ignore_label is not None,
            dice_class=MemoryEfficientSoftDiceLoss
        )
    else:
        loss = DC_and_CE_loss(
            {
                'batch_dice': trainer_instance.configuration_manager.batch_dice,
                'smooth': 0,
                'do_bg': False,
                'ddp': trainer_instance.is_ddp
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=trainer_instance.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

    if trainer_instance.enable_deep_supervision:
        deep_supervision_scales = trainer_instance._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)

    return loss


# Dice loss only
DICE_LOSS_CONFIG = TrainerConfig(
    name="dice_loss",
    description="Use Dice loss only (no CE/BCE)",
    loss_builder=dice_loss_builder
)
register_config(DICE_LOSS_CONFIG)


# Dice+CE loss with smooth=0
DICE_CE_NO_SMOOTH_CONFIG = TrainerConfig(
    name="dice_ce_no_smooth",
    description="Use Dice+CE loss with smooth=0",
    loss_builder=dice_ce_no_smooth_builder
)
register_config(DICE_CE_NO_SMOOTH_CONFIG)