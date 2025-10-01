"""
Utilities for building plan dictionaries from configuration components.
"""
from typing import Dict, Any, Tuple, Callable, Union, List
import numpy as np


def build_plan_dict(
    data_identifier: str,
    preprocessor_name: str,
    batch_size: int,
    patch_size: Tuple[int, ...],
    median_shape: Union[np.ndarray, Tuple[int, ...]],
    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
    normalization_schemes: List[str],
    mask_is_used_for_norm: List[bool],
    resampling_data: Callable,
    resampling_seg: Callable,
    resampling_data_kwargs: Dict[str, Any],
    resampling_seg_kwargs: Dict[str, Any],
    resampling_softmax: Callable,
    resampling_softmax_kwargs: Dict[str, Any],
    architecture_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Builds a plan dictionary from configuration components.

    Args:
        data_identifier: Identifier for this configuration's data
        preprocessor_name: Name of the preprocessor to use
        batch_size: Batch size for training
        patch_size: Patch size for training
        median_shape: Median image shape in voxels
        spacing: Voxel spacing
        normalization_schemes: List of normalization scheme names
        mask_is_used_for_norm: Whether to use mask for normalization for each channel
        resampling_data: Resampling function for data
        resampling_seg: Resampling function for segmentation
        resampling_data_kwargs: Kwargs for data resampling
        resampling_seg_kwargs: Kwargs for segmentation resampling
        resampling_softmax: Resampling function for softmax/probabilities
        resampling_softmax_kwargs: Kwargs for softmax resampling
        architecture_kwargs: Architecture configuration

    Returns:
        Plan dictionary
    """
    plan = {
        'data_identifier': data_identifier,
        'preprocessor_name': preprocessor_name,
        'batch_size': batch_size,
        'patch_size': patch_size,
        'median_image_size_in_voxels': median_shape,
        'spacing': spacing,
        'normalization_schemes': normalization_schemes,
        'use_mask_for_norm': mask_is_used_for_norm,
        'resampling_fn_data': resampling_data.__name__,
        'resampling_fn_seg': resampling_seg.__name__,
        'resampling_fn_data_kwargs': resampling_data_kwargs,
        'resampling_fn_seg_kwargs': resampling_seg_kwargs,
        'resampling_fn_probabilities': resampling_softmax.__name__,
        'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
        'architecture': architecture_kwargs
    }
    return plan
