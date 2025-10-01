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

From custom/:
    - DynamicKiUNet: Dynamic KiU-Net architecture with dual-branch encoder-decoder
    - CRFB: Cross-Refinement Block for feature exchange between branches

Usage:
------
For training:
    from nnunetv2.architecture import build_network_architecture

For custom architectures:
    from nnunetv2.architecture import get_network_from_plans
    from nnunetv2.architecture import DynamicKiUNet

For network configuration:
    from nnunetv2.architecture import set_deep_supervision_enabled, _do_i_compile
"""

from .builder import build_network_architecture
from .instantiation import get_network_from_plans
from .config import _do_i_compile, set_deep_supervision_enabled, plot_network_architecture
from .custom import DynamicKiUNet, CRFB

__all__ = [
    'build_network_architecture',
    'get_network_from_plans',
    '_do_i_compile',
    'set_deep_supervision_enabled',
    'plot_network_architecture',
    'DynamicKiUNet',
    'CRFB',
]