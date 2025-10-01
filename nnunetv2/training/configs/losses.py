"""Loss function configuration presets."""

import numpy as np
import torch

from nnunetv2.training.configs import TrainerConfig, register_config
from nnunetv2.training.losses.implementations.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.losses.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.losses.implementations.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.losses.implementations.region_dice import RegionDiceLoss
from nnunetv2.training.losses.implementations.blob_dice import BlobDiceLoss, BlobDiceCELoss
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

# Region Dice loss only (multi-class region-based connected component Dice)

def region_dice_loss_builder(trainer_instance):
    # apply appropriate nonlinearity: sigmoid if regions-style (rare), else softmax
    from nnunetv2.utilities.core.helpers import softmax_helper_dim1
    if trainer_instance.label_manager.has_regions:
        # region-based training: use sigmoid over independent region channels
        apply_nonlin = torch.sigmoid
        loss = RegionDiceLoss(apply_nonlin=apply_nonlin, include_background=False, has_regions=True,
                              has_ignore=trainer_instance.label_manager.has_ignore_label)
    else:
        from nnunetv2.utilities.core.helpers import softmax_helper_dim1
        apply_nonlin = softmax_helper_dim1
        loss = RegionDiceLoss(apply_nonlin=apply_nonlin, include_background=False, has_regions=False,
                              has_ignore=False)

    if trainer_instance.enable_deep_supervision:
        # Region loss is defined on full resolution only; we set weight 1 for highest res, 0 for others
        deep_supervision_scales = trainer_instance._get_deep_supervision_scales()
        weights = np.zeros(len(deep_supervision_scales), dtype=float)
        weights[0] = 1.0
        loss = DeepSupervisionWrapper(loss, weights)
    return loss

REGION_DICE_LOSS_CONFIG = TrainerConfig(
    name="region_dice_loss",
    description="Region-based connected-component Dice loss (multi-class, background excluded)",
    loss_builder=region_dice_loss_builder
)
register_config(REGION_DICE_LOSS_CONFIG)

register_config(DICE_CE_NO_SMOOTH_CONFIG)


# Blob Dice loss (per-instance connected component Dice)
def blob_dice_loss_builder(trainer_instance):
    """
    Build blob-wise Dice loss.

    Computes Dice per connected component (blob) in ground truth.
    Encourages accurate per-instance segmentation.
    """
    from nnunetv2.utilities.core.helpers import softmax_helper_dim1

    if trainer_instance.label_manager.has_regions:
        # Region-based: use sigmoid
        apply_nonlin = torch.sigmoid
    else:
        # Standard multiclass: use softmax
        apply_nonlin = softmax_helper_dim1

    loss = BlobDiceLoss(
        apply_nonlin=apply_nonlin,
        include_background=False,
        smooth=1e-6
    )

    if trainer_instance.enable_deep_supervision:
        # Blob loss works best on full resolution (connected components need spatial context)
        deep_supervision_scales = trainer_instance._get_deep_supervision_scales()
        weights = np.zeros(len(deep_supervision_scales), dtype=float)
        weights[0] = 1.0  # Only compute on highest resolution
        loss = DeepSupervisionWrapper(loss, weights)

    return loss


BLOB_DICE_LOSS_CONFIG = TrainerConfig(
    name="blob_dice_loss",
    description="Blob-wise Dice loss - computes Dice per connected component for instance-aware segmentation (1 epoch test)",
    loss_builder=blob_dice_loss_builder,
    num_epochs=1  # Quick test by default
)
register_config(BLOB_DICE_LOSS_CONFIG)


# Blob Dice + CE combined loss
def blob_dice_ce_loss_builder(trainer_instance):
    """
    Build combined blob Dice + Cross-Entropy loss.

    Blob Dice encourages per-instance accuracy, CE provides stable gradients.
    """
    from nnunetv2.utilities.core.helpers import softmax_helper_dim1

    if trainer_instance.label_manager.has_regions:
        # Region-based mode not recommended for this loss (CE needs multiclass)
        raise ValueError("BlobDiceCELoss requires standard multiclass labels (has_regions=False)")

    # Use softmax for dice component
    apply_nonlin = softmax_helper_dim1

    loss = BlobDiceCELoss(
        blob_weight=1.0,
        ce_weight=1.0,
        apply_nonlin=apply_nonlin,
        include_background=False
    )

    if trainer_instance.enable_deep_supervision:
        # Apply to full resolution only (blob detection needs spatial context)
        deep_supervision_scales = trainer_instance._get_deep_supervision_scales()
        weights = np.zeros(len(deep_supervision_scales), dtype=float)
        weights[0] = 1.0
        loss = DeepSupervisionWrapper(loss, weights)

    return loss


BLOB_DICE_CE_LOSS_CONFIG = TrainerConfig(
    name="blob_dice_ce_loss",
    description="Combined Blob Dice + Cross-Entropy loss for instance-aware segmentation (1 epoch test)",
    loss_builder=blob_dice_ce_loss_builder,
    num_epochs=1  # Quick test by default
)
register_config(BLOB_DICE_CE_LOSS_CONFIG)