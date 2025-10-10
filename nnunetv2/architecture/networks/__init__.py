"""
Network architecture builders for nnU-Net.

This module contains network architectures and their builder functions.
Each builder has the same signature and can be used interchangeably
via TrainerConfig.network_builder.

Available Networks:
-------------------
From unet.py:
    - build_unet: Default UNet architecture (PlainConvUNet, ResidualEncoderUNet, etc.)

From kiunet.py:
    - DynamicKiUNet: Dynamic KiU-Net architecture with dual-branch encoder-decoder
    - CRFB: Cross-Refinement Block for feature exchange between branches
    - build_kiunet_maxpool: KiU-Net with MaxPool downsampling (original paper)
    - build_kiunet_conv: KiU-Net with strided convolutions (faster)
    - build_kiunet_minimal: KiU-Net with reduced memory footprint

From uiunet.py:
    - DynamicUIUNet3D: Dynamic UIU-Net architecture with nested RSU blocks
    - DynamicRSU3D: Residual U-block with internal U-Net structure
    - build_uiunet: UIU-Net with full RSU heights
    - build_uiunet_minimal: UIU-Net with reduced memory footprint

Usage:
------
    from nnunetv2.architecture.networks import build_unet, build_kiunet_conv
    from nnunetv2.training.configs import TrainerConfig

    config = TrainerConfig(
        name="my_kiunet",
        network_builder=build_kiunet_conv
    )
"""

# UNet builders
from .unet import build_unet

# KiUNet network classes and builders
from .kiunet import (
    DynamicKiUNet,
    CRFB,
    build_kiunet_maxpool,
    build_kiunet_conv,
    build_kiunet_minimal
)

# UIUNet network classes and builders
from .uiunet import (
    DynamicUIUNet3D,
    DynamicRSU3D,
    build_uiunet,
    build_uiunet_minimal
)

__all__ = [
    # UNet
    'build_unet',

    # KiUNet
    'DynamicKiUNet',
    'CRFB',
    'build_kiunet_maxpool',
    'build_kiunet_conv',
    'build_kiunet_minimal',

    # UIUNet
    'DynamicUIUNet3D',
    'DynamicRSU3D',
    'build_uiunet',
    'build_uiunet_minimal',
]
