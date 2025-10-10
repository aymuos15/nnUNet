"""
UIU-Net architecture configs for nnU-Net.

Provides configs that use the DynamicUIUNet3D architecture with nested RSU blocks.
The actual builder functions are located in nnunetv2.architecture.networks.uiunet.

Available Configs:
-----------------
- UIUNET_CONFIG: Full RSU heights (1 epoch for testing)
- UIUNET_MINIMAL_CONFIG: Reduced RSU heights and 50% features for 24GB GPUs (5 epochs)
"""

from nnunetv2.training.configs.base import TrainerConfig, register_config
from nnunetv2.architecture.networks import (
    build_uiunet,
    build_uiunet_minimal
)


# UIU-Net config (1 epoch for testing)
UIUNET_CONFIG = TrainerConfig(
    name="uiunet",
    description="DynamicUIUNet3D architecture with nested RSU blocks (1 epoch for testing)",
    num_epochs=1,
    network_builder=build_uiunet,
)

# UIU-Net minimal config for 24GB GPUs (5 epochs)
UIUNET_MINIMAL_CONFIG = TrainerConfig(
    name="uiunet_minimal",
    description="DynamicUIUNet3D for 24GB GPU (batch_size=1, reduced RSU heights, 50% features, 5 epochs)",
    num_epochs=5,
    batch_size=1,
    network_builder=build_uiunet_minimal,
)

# Register all configs
register_config(UIUNET_CONFIG)
register_config(UIUNET_MINIMAL_CONFIG)
