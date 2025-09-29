"""Configuration and device setup for nnUNet predictor."""

import torch
from typing import Optional


def setup_predictor_state(predictor,
                         tile_step_size: float = 0.5,
                         use_gaussian: bool = True,
                         use_mirroring: bool = True,
                         perform_everything_on_device: bool = True,
                         device: torch.device = torch.device('cuda'),
                         verbose: bool = False,
                         verbose_preprocessing: bool = False,
                         allow_tqdm: bool = True):
    """
    Setup initial predictor state and configuration.

    Args:
        predictor: The nnUNetPredictor instance
        tile_step_size: Step size for sliding window
        use_gaussian: Whether to use Gaussian weighting
        use_mirroring: Whether to use test-time mirroring
        perform_everything_on_device: Keep everything on GPU
        device: Device to use for prediction
        verbose: Verbose output
        verbose_preprocessing: Verbose preprocessing output
        allow_tqdm: Allow tqdm progress bars
    """
    predictor.verbose = verbose
    predictor.verbose_preprocessing = verbose_preprocessing
    predictor.allow_tqdm = allow_tqdm

    # Initialize all attributes to None
    predictor.plans_manager = None
    predictor.configuration_manager = None
    predictor.list_of_parameters = None
    predictor.network = None
    predictor.dataset_json = None
    predictor.trainer_name = None
    predictor.allowed_mirroring_axes = None
    predictor.label_manager = None

    predictor.tile_step_size = tile_step_size
    predictor.use_gaussian = use_gaussian
    predictor.use_mirroring = use_mirroring

    # Configure device settings
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
        perform_everything_on_device = False

    predictor.device = device
    predictor.perform_everything_on_device = perform_everything_on_device