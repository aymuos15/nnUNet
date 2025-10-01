"""
Custom Architecture Implementations for nnU-Net

This module contains custom neural network architectures that can be used
with nnU-Net's planning and training system.

Available Architectures:
------------------------
- DynamicKiUNet: Dynamic KiU-Net with dual-branch encoder-decoder and
                 cross-refinement blocks

Usage:
------
Reference in plans or custom trainers as:
    "nnunetv2.architecture.custom.kiunet.DynamicKiUNet"

Or import directly:
    from nnunetv2.architecture.custom import DynamicKiUNet
"""

from .kiunet import DynamicKiUNet, CRFB

__all__ = ['DynamicKiUNet', 'CRFB']
