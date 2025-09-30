# For backward compatibility, import nnUNetPredictor from the new location
from .predictor.main import nnUNetPredictor

# Export the main predictor class
__all__ = ['nnUNetPredictor']