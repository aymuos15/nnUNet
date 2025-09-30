"""
Metrics for nnU-Net.

This module provides metric computation functions that can be used during:
- Training validation
- Inference
- Evaluation

The module is organized similar to the losses module:
- implementations: Actual metric computation implementations
"""

from .implementations import get_tp_fp_fn_tn

__all__ = [
    'get_tp_fp_fn_tn',
]
