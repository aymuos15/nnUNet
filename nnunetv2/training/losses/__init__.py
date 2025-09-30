"""
Loss functions and utilities for nnU-Net training.

This module provides:
- builder: Functions to build loss functions based on configuration
- deep_supervision: Wrapper for applying losses to multi-scale outputs
- implementations: Actual loss function implementations (dice, CE, compounds)
"""

from .builder import _build_loss, _get_deep_supervision_scales
from .deep_supervision import DeepSupervisionWrapper
from .implementations import (
    SoftDiceLoss,
    MemoryEfficientSoftDiceLoss,
    RobustCrossEntropyLoss,
    TopKLoss,
    DC_and_CE_loss,
    DC_and_BCE_loss
)

__all__ = [
    # Builder functions
    '_build_loss',
    '_get_deep_supervision_scales',
    # Wrappers
    'DeepSupervisionWrapper',
    # Loss implementations
    'SoftDiceLoss',
    'MemoryEfficientSoftDiceLoss',
    'RobustCrossEntropyLoss',
    'TopKLoss',
    'DC_and_CE_loss',
    'DC_and_BCE_loss'
]
