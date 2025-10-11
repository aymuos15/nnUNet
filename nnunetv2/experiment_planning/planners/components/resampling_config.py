"""
Utilities for determining resampling and normalization configuration for experiment planning.
"""
from typing import Tuple, List, Callable, Dict, Any

from nnunetv2.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape


def determine_normalization_scheme_and_whether_mask_is_used_for_norm(
    dataset_json: dict,
    dataset_fingerprint: dict
) -> Tuple[List[str], List[bool]]:
    """
    Determines normalization schemes and whether to use mask for normalization.

    Args:
        dataset_json: Dataset configuration JSON
        dataset_fingerprint: Dataset statistics fingerprint

    Returns:
        Tuple of (normalization_schemes, use_nonzero_mask_for_norm)
    """
    if 'channel_names' not in dataset_json.keys():
        print('WARNING: "modalities" should be renamed to "channel_names" in dataset.json. This will be '
              'enforced soon!')
    modalities = dataset_json['channel_names'] if 'channel_names' in dataset_json.keys() else \
        dataset_json['modality']
    normalization_schemes = [get_normalization_scheme(m) for m in modalities.values()]
    if dataset_fingerprint['median_relative_size_after_cropping'] < (3 / 4.):
        use_nonzero_mask_for_norm = [i.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true for i in
                                     normalization_schemes]
    else:
        use_nonzero_mask_for_norm = [False] * len(normalization_schemes)
        assert all([i in (True, False) for i in use_nonzero_mask_for_norm]), 'use_nonzero_mask_for_norm must be ' \
                                                                             'True or False and cannot be None'
    normalization_schemes = [i.__name__ for i in normalization_schemes]
    return normalization_schemes, use_nonzero_mask_for_norm


def determine_resampling(
    *args, **kwargs
) -> Tuple[Callable, Dict[str, Any], Callable, Dict[str, Any]]:
    """
    Returns what functions to use for resampling data and seg, respectively. Also returns kwargs
    resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

    determine_resampling is called within get_plans_for_configuration to allow for different functions for each
    configuration

    Returns:
        Tuple of (resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs)
    """
    resampling_data = resample_data_or_seg_to_shape
    resampling_data_kwargs = {
        "is_seg": False,
        "order": 3,
        "order_z": 0,
        "force_separate_z": None,
    }
    resampling_seg = resample_data_or_seg_to_shape
    resampling_seg_kwargs = {
        "is_seg": True,
        "order": 1,
        "order_z": 0,
        "force_separate_z": None,
    }
    return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs


def determine_segmentation_softmax_export_fn(
    *args, **kwargs
) -> Tuple[Callable, Dict[str, Any]]:
    """
    Function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
    used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

    determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
    functions for each configuration

    Returns:
        Tuple of (resampling_fn, resampling_fn_kwargs)
    """
    resampling_fn = resample_data_or_seg_to_shape
    resampling_fn_kwargs = {
        "is_seg": False,
        "order": 1,
        "order_z": 0,
        "force_separate_z": None,
    }
    return resampling_fn, resampling_fn_kwargs
