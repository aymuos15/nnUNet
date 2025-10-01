"""
Utilities for VRAM estimation and batch size computation for experiment planning.
"""
import torch
import numpy as np
from typing import Tuple

from nnunetv2.architecture import get_network_from_plans
from nnunetv2.training.runtime_utils.default_n_proc_DA import get_allowed_n_proc_DA


def static_estimate_VRAM_usage(
    patch_size: Tuple[int],
    input_channels: int,
    output_channels: int,
    arch_class_name: str,
    arch_kwargs: dict,
    arch_kwargs_req_import: Tuple[str, ...]
) -> int:
    """
    Estimates VRAM usage for a given network configuration.
    Works for PlainConvUNet, ResidualEncoderUNet

    Args:
        patch_size: Patch size for the network
        input_channels: Number of input channels
        output_channels: Number of output channels
        arch_class_name: Architecture class name as string
        arch_kwargs: Architecture keyword arguments
        arch_kwargs_req_import: Tuple of architecture kwargs that require import

    Returns:
        Estimated VRAM usage
    """
    a = torch.get_num_threads()
    torch.set_num_threads(get_allowed_n_proc_DA())
    # print(f'instantiating network, patch size {patch_size}, pool op: {arch_kwargs["strides"]}')
    net = get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels,
                                 output_channels,
                                 allow_init=False)
    ret = net.compute_conv_feature_map_size(patch_size)
    torch.set_num_threads(a)
    return ret


def compute_batch_size(
    vram_estimate: int,
    reference_val: int,
    reference_bs: int,
    vram_target_GB: float,
    reference_corresp_GB: float,
    min_batch_size: int,
    patch_size: Tuple[int],
    approximate_n_voxels_dataset: float,
    max_dataset_covered: float
) -> int:
    """
    Computes appropriate batch size based on VRAM estimates and dataset size.

    Args:
        vram_estimate: Estimated VRAM usage
        reference_val: Reference VRAM value (2D or 3D)
        reference_bs: Reference batch size corresponding to reference_val
        vram_target_GB: Target VRAM in GB
        reference_corresp_GB: GB corresponding to reference values
        min_batch_size: Minimum allowed batch size
        patch_size: Patch size
        approximate_n_voxels_dataset: Approximate total voxels in dataset
        max_dataset_covered: Maximum fraction of dataset to cover in one batch

    Returns:
        Computed batch size
    """
    # adapt for our vram target
    reference = reference_val * (vram_target_GB / reference_corresp_GB)

    # alright now let's determine the batch size. This will give min_batch_size if the while loop was
    # executed. If not, additional vram headroom is used to increase batch size
    batch_size = round((reference / vram_estimate) * reference_bs)

    # we need to cap the batch size to cover at most max_dataset_covered of the entire dataset.
    # Overfitting precaution. We cannot go smaller than min_batch_size though
    bs_corresponding_to_max_percent = round(
        approximate_n_voxels_dataset * max_dataset_covered / np.prod(patch_size, dtype=np.float64))
    batch_size = max(min(batch_size, bs_corresponding_to_max_percent), min_batch_size)

    return batch_size
