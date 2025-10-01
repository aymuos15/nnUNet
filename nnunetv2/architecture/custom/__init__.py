"""
Custom Architecture Implementations for nnU-Net

This module contains custom neural network architectures that can be used
with nnU-Net's planning and training system.

Available Architectures:
------------------------
- DynamicKiUNet: Dynamic KiU-Net with dual-branch encoder-decoder and
                 cross-refinement blocks
- DynamicUIUNet3D: Dynamic UIU-Net with nested RSU blocks for multi-scale
                   feature extraction and uncertainty-inspired fusion

Usage:
------
Reference in plans or custom trainers as:
    "nnunetv2.architecture.custom.kiunet.DynamicKiUNet"
    "nnunetv2.architecture.custom.uiunet.DynamicUIUNet3D"

Or import directly:
    from nnunetv2.architecture.custom import DynamicKiUNet, DynamicUIUNet3D
"""

from .kiunet import DynamicKiUNet, CRFB
from .uiunet import DynamicUIUNet3D, DynamicRSU3D

__all__ = ['DynamicKiUNet', 'CRFB', 'DynamicUIUNet3D', 'DynamicRSU3D']
