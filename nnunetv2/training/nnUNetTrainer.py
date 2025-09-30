"""
Backward compatibility module.
This module redirects to the new location of nnUNetTrainer.
"""

from nnunetv2.training.trainer.main import nnUNetTrainer

__all__ = ['nnUNetTrainer']