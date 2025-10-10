"""
nnU-Net Architecture Module

This module contains all network architecture-related code, providing a centralized
location for custom model development and network configuration.

Public API:
-----------
From builder.py:
    - build_network_architecture: High-level function to build networks from plans

From instantiation.py:
    - get_network_from_plans: Low-level function to instantiate networks

From config.py:
    - _do_i_compile: Determine whether to use torch.compile
    - set_deep_supervision_enabled: Enable/disable deep supervision
    - plot_network_architecture: Visualize network architecture

From networks/:
    - build_unet: Default UNet architecture builder
    - build_kiunet_maxpool: KiU-Net with MaxPool downsampling
    - build_kiunet_conv: KiU-Net with strided convolutions
    - build_kiunet_minimal: KiU-Net with reduced memory footprint
    - build_uiunet: UIU-Net with full RSU heights
    - build_uiunet_minimal: UIU-Net with reduced memory footprint
    - DynamicKiUNet: Dynamic KiU-Net architecture with dual-branch encoder-decoder
    - CRFB: Cross-Refinement Block for feature exchange between branches
    - DynamicUIUNet3D: Dynamic UIU-Net architecture with nested RSU blocks
    - DynamicRSU3D: Residual U-block with internal U-Net structure

Usage:
------
For training:
    from nnunetv2.architecture import build_network_architecture

For custom architectures via config:
    from nnunetv2.architecture import build_kiunet_conv
    from nnunetv2.training.configs import TrainerConfig

    config = TrainerConfig(
        name="my_kiunet",
        network_builder=build_kiunet_conv
    )

For direct network instantiation:
    from nnunetv2.architecture import get_network_from_plans
    from nnunetv2.architecture import DynamicKiUNet

For network configuration:
    from nnunetv2.architecture import set_deep_supervision_enabled, _do_i_compile
"""

from .builder import build_network_architecture
from .instantiation import get_network_from_plans
from .config import _do_i_compile, set_deep_supervision_enabled, plot_network_architecture
from .networks import (
    build_unet,
    build_kiunet_maxpool,
    build_kiunet_conv,
    build_kiunet_minimal,
    build_uiunet,
    build_uiunet_minimal,
    DynamicKiUNet,
    CRFB,
    DynamicUIUNet3D,
    DynamicRSU3D
)

__all__ = [
    'build_network_architecture',
    'get_network_from_plans',
    '_do_i_compile',
    'set_deep_supervision_enabled',
    'plot_network_architecture',
    'build_unet',
    'build_kiunet_maxpool',
    'build_kiunet_conv',
    'build_kiunet_minimal',
    'build_uiunet',
    'build_uiunet_minimal',
    'DynamicKiUNet',
    'CRFB',
    'DynamicUIUNet3D',
    'DynamicRSU3D',
]