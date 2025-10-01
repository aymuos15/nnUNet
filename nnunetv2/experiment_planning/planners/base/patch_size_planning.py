"""
Utilities for computing and refining patch sizes during experiment planning.
"""
import numpy as np
from copy import deepcopy
from typing import Tuple, Union, List

from nnunetv2.experiment_planning.planners.base.network_topology import get_pool_and_conv_props


def compute_initial_patch_size(
    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
    median_shape: Union[np.ndarray, Tuple[int, ...]],
) -> np.ndarray:
    """
    Computes initial patch size based on spacing and median shape.

    The initial patch size is computed to:
    1. Have an aspect ratio matching the spacing
    2. Have the same volume as a 256^3 patch (3D) or 2048^2 patch (2D)
    3. Be clipped to the median shape (no point being larger than median)

    Args:
        spacing: Spacing of the data
        median_shape: Median shape of the dataset

    Returns:
        Initial patch size as numpy array
    """
    # we first use the spacing to get an aspect ratio
    tmp = 1 / np.array(spacing)

    # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
    # volume as a patch of size 256 ** 3)
    # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
    # ideal because large initial patch sizes increase computation time because more iterations in the while loop
    # further down may be required.
    if len(spacing) == 3:
        initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
    elif len(spacing) == 2:
        initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
    else:
        raise RuntimeError(f"Spacing must be 2D or 3D, got {len(spacing)}D")

    # clip initial patch size to median_shape. It makes little sense to have it be larger than that. Note that
    # this is different from how nnU-Net v1 does it!
    # todo patch size can still get too large because we pad the patch size to a multiple of 2**n
    initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)])])

    return initial_patch_size


def reduce_patch_size_step(
    patch_size: Tuple[int, ...],
    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
    median_shape: Union[np.ndarray, Tuple[int, ...]],
    shape_must_be_divisible_by: Union[np.ndarray, List[int]],
    featuremap_min_edge_length: int
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...], np.ndarray]:
    """
    Reduces patch size by one step, selecting the axis that violates aspect ratio the most.

    This function:
    1. Finds the axis that is largest relative to median shape
    2. Reduces that axis by an appropriate divisor
    3. Recomputes network topology

    Args:
        patch_size: Current patch size
        spacing: Spacing of the data
        median_shape: Median shape of the dataset
        shape_must_be_divisible_by: Divisibility constraint for each axis
        featuremap_min_edge_length: Minimum edge length for feature maps

    Returns:
        Tuple of (new_patch_size, pool_op_kernel_sizes, conv_kernel_sizes, shape_must_be_divisible_by)
    """
    # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
    # aspect ratio the most (that is the largest relative to median shape)
    axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]

    # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
    # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
    # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
    # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
    # subtract shape_must_be_divisible_by, then recompute it and then subtract the
    # recomputed shape_must_be_divisible_by. Annoying.
    patch_size = list(patch_size)
    tmp = deepcopy(patch_size)
    tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
    _, _, _, _, shape_must_be_divisible_by = \
        get_pool_and_conv_props(spacing, tmp, featuremap_min_edge_length, 999999)
    patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

    # now recompute topology
    network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
    shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size, featuremap_min_edge_length, 999999)

    return patch_size, pool_op_kernel_sizes, conv_kernel_sizes, shape_must_be_divisible_by
