"""
KiU-Net architecture configs for nnU-Net.

Provides pre-configured TrainerConfig instances that use the DynamicKiUNet
architecture instead of the default U-Net. The actual builder functions are
located in nnunetv2.architecture.factory.kiunet_builder.

Available Configs:
-----------------
- KIUNET_CONFIG: MaxPool downsampling (2 epochs for testing)
- KIUNET_CONV_CONFIG: Strided convolutions (2 epochs for testing)
- KIUNET_MINIMAL_CONFIG: Reduced memory footprint (1 epoch for testing)
- KIUNET_LARGE_CONFIG: Full features, production use (1000 epochs)
"""

from nnunetv2.training.configs.base import TrainerConfig, register_config
from nnunetv2.architecture.factory import (
    build_kiunet_maxpool,
    build_kiunet_conv,
    build_kiunet_minimal
)


# KiU-Net config with MaxPool (2 epochs for testing)
KIUNET_CONFIG = TrainerConfig(
    name="kiunet",
    description="DynamicKiUNet architecture with MaxPool downsampling (2 epochs for testing)",
    num_epochs=2,
    network_builder=build_kiunet_maxpool,
)

# KiU-Net config with strided convolutions (2 epochs for testing)
KIUNET_CONV_CONFIG = TrainerConfig(
    name="kiunet_conv",
    description="DynamicKiUNet architecture with strided convolutions (2 epochs for testing)",
    num_epochs=2,
    network_builder=build_kiunet_conv,
)

# Optimal config for 24GB GPUs (reduced features to fit dual-branch architecture)
KIUNET_MINIMAL_CONFIG = TrainerConfig(
    name="kiunet_minimal",
    description="DynamicKiUNet for 24GB GPU (batch_size=1, 3x3x3 kernels, 50% features, strided conv, 1 epoch)",
    num_epochs=1,
    batch_size=1,
    network_builder=build_kiunet_minimal,  # Reduced feature channels only
)

# Optimized config for 24GB GPUs (production use)
KIUNET_LARGE_CONFIG = TrainerConfig(
    name="kiunet_large",
    description="DynamicKiUNet optimized for 24GB GPU (batch_size=1, 3x3x3 kernels, full features, strided conv)",
    num_epochs=1000,
    batch_size=1,
    network_builder=build_kiunet_conv,  # Use strided conv for efficiency
)

# Register all configs
register_config(KIUNET_CONFIG)
register_config(KIUNET_CONV_CONFIG)
register_config(KIUNET_MINIMAL_CONFIG)
register_config(KIUNET_LARGE_CONFIG)
