"""
Network architecture factory module.

This module contains builder functions for different network architectures.
Each builder function has the same signature and can be used interchangeably
via TrainerConfig.network_builder.

Available Builders:
------------------
From unet_builder:
    - build_unet: Default UNet architecture (PlainConvUNet, ResidualEncoderUNet, etc.)

From kiunet_builder:
    - build_kiunet_maxpool: KiU-Net with MaxPool downsampling (original paper)
    - build_kiunet_conv: KiU-Net with strided convolutions (faster)
    - build_kiunet_minimal: KiU-Net with reduced memory footprint

Usage:
------
    from nnunetv2.architecture.factory import build_unet, build_kiunet_conv
    from nnunetv2.training.configs import TrainerConfig

    config = TrainerConfig(
        name="my_kiunet",
        network_builder=build_kiunet_conv
    )
"""

from .unet_builder import build_unet
from .kiunet_builder import (
    build_kiunet_maxpool,
    build_kiunet_conv,
    build_kiunet_minimal
)

__all__ = [
    'build_unet',
    'build_kiunet_maxpool',
    'build_kiunet_conv',
    'build_kiunet_minimal',
]
