"""
Metric implementations for nnU-Net.

This module contains the actual metric computation implementations:
- Dice metrics (TP, FP, FN, TN computation)
"""

from .dice import get_tp_fp_fn_tn

__all__ = [
    'get_tp_fp_fn_tn',
]
