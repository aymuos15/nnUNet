"""Mirroring configuration presets."""

from nnunetv2.training.configs import TrainerConfig, register_config


# No mirroring variant
NO_MIRRORING_CONFIG = TrainerConfig(
    name="no_mirroring",
    description="Disable mirroring during training and inference",
    mirror_axes=None,
    inference_allowed_mirroring_axes=None
)
register_config(NO_MIRRORING_CONFIG)


# Only mirror along axes 0 and 1 for 3D, axis 0 for 2D
# Note: This would need dynamic determination based on patch size dimensions
# For now, we set it to (0, 1) and users can override if needed
MIRROR_01_CONFIG = TrainerConfig(
    name="mirror_01",
    description="Only mirror along spatial axes 0 and 1",
    mirror_axes=(0, 1),
    inference_allowed_mirroring_axes=(0, 1)
)
register_config(MIRROR_01_CONFIG)