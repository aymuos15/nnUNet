"""Test-time mirroring augmentation for nnUNet predictor."""

import itertools
import torch


@torch.inference_mode()
def maybe_mirror_and_predict(predictor, x: torch.Tensor) -> torch.Tensor:
    """
    Perform prediction with optional test-time mirroring.

    Args:
        predictor: The nnUNetPredictor instance
        x: Input tensor

    Returns:
        Prediction tensor (potentially averaged across mirror augmentations)
    """
    mirror_axes = predictor.allowed_mirroring_axes if predictor.use_mirroring else None
    prediction = predictor.network(x)

    if mirror_axes is not None:
        # Check for invalid numbers in mirror_axes
        # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
        assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

        # Adjust mirror axes for batch dimension
        mirror_axes = [m + 2 for m in mirror_axes]
        axes_combinations = [
            c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
        ]

        # Apply mirroring augmentations and average predictions
        for axes in axes_combinations:
            prediction += torch.flip(predictor.network(torch.flip(x, axes)), axes)
        prediction /= (len(axes_combinations) + 1)

    return prediction