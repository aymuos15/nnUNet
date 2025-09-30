"""Data augmentation configuration presets."""

from nnunetv2.training.configs import TrainerConfig, register_config
from nnunetv2.data.transform_builders import get_validation_transforms


def no_da_transforms_builder(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
                              do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
                              foreground_labels=None, regions=None, ignore_label=None):
    """Return validation transforms for training (no data augmentation)."""
    return get_validation_transforms(
        deep_supervision_scales,
        is_cascaded=is_cascaded,
        foreground_labels=foreground_labels,
        regions=regions,
        ignore_label=ignore_label
    )


# No data augmentation
NO_DA_CONFIG = TrainerConfig(
    name="no_da",
    description="Disable all data augmentation (use validation transforms for training)",
    training_transforms_builder=no_da_transforms_builder,
    mirror_axes=None,
    inference_allowed_mirroring_axes=None
)
register_config(NO_DA_CONFIG)


# No dummy 2D data augmentation
NO_DUMMY_2D_DA_CONFIG = TrainerConfig(
    name="no_dummy_2d_da",
    description="Disable dummy 2D data augmentation",
    do_dummy_2d_data_aug=False
)
register_config(NO_DUMMY_2D_DA_CONFIG)