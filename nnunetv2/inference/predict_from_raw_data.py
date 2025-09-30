"""
Backward compatibility layer for nnUNet predictor.

This module maintains backward compatibility for existing code that imports from predict_from_raw_data.py.
The actual implementation has been refactored into a modular architecture under nnunetv2.inference.predictor.
"""

# Import the refactored predictor class
from .predictor.main import nnUNetPredictor

# Import CLI functions for backward compatibility
from .predictor.cli import predict_entry_point_modelfolder, predict_entry_point, _getDefaultValue

# Re-export for backward compatibility
__all__ = ['nnUNetPredictor', 'predict_entry_point_modelfolder', 'predict_entry_point', '_getDefaultValue']