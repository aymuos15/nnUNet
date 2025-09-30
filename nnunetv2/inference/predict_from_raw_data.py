"""
Backward compatibility module.
This module redirects to the new location of predict_entry_point.
"""

from nnunetv2.inference.predictor.cli import predict_entry_point, predict_entry_point_modelfolder

__all__ = ['predict_entry_point', 'predict_entry_point_modelfolder']