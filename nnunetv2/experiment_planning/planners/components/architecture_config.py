"""
Utilities for network architecture configuration for experiment planning.
"""
from typing import Tuple, Dict, Any
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm


def compute_features_per_stage(
    num_stages: int,
    max_num_features: int,
    base_num_features: int = 32
) -> Tuple[int, ...]:
    """
    Computes number of features for each stage of the network.

    Args:
        num_stages: Number of stages in the network
        max_num_features: Maximum number of features
        base_num_features: Base number of features (default: 32)

    Returns:
        Tuple of feature counts per stage
    """
    return tuple([min(max_num_features, base_num_features * 2 ** i) for i in range(num_stages)])


def build_architecture_kwargs(
    spacing: Tuple[float, ...],
    num_stages: int,
    pool_op_kernel_sizes: Tuple[Tuple[int, ...], ...],
    conv_kernel_sizes: Tuple[Tuple[int, ...], ...],
    max_num_features: int,
    unet_class: type,
    blocks_per_stage_encoder: Tuple[int, ...],
    blocks_per_stage_decoder: Tuple[int, ...],
    base_num_features: int = 32
) -> Dict[str, Any]:
    """
    Builds architecture configuration dictionary for network instantiation.

    Args:
        spacing: Spacing of the data
        num_stages: Number of stages in the network
        pool_op_kernel_sizes: Pooling kernel sizes for each stage
        conv_kernel_sizes: Convolution kernel sizes for each stage
        max_num_features: Maximum number of features
        unet_class: UNet class to use
        blocks_per_stage_encoder: Number of blocks per encoder stage
        blocks_per_stage_decoder: Number of blocks per decoder stage
        base_num_features: Base number of features (default: 32)

    Returns:
        Dictionary containing architecture configuration
    """
    unet_conv_op = convert_dim_to_conv_op(len(spacing))
    norm = get_matching_instancenorm(unet_conv_op)

    architecture_kwargs = {
        'network_class_name': unet_class.__module__ + '.' + unet_class.__name__,
        'arch_kwargs': {
            'n_stages': num_stages,
            'features_per_stage': compute_features_per_stage(num_stages, max_num_features, base_num_features),
            'conv_op': unet_conv_op.__module__ + '.' + unet_conv_op.__name__,
            'kernel_sizes': conv_kernel_sizes,
            'strides': pool_op_kernel_sizes,
            'n_conv_per_stage': blocks_per_stage_encoder[:num_stages],
            'n_conv_per_stage_decoder': blocks_per_stage_decoder[:num_stages - 1],
            'conv_bias': True,
            'norm_op': norm.__module__ + '.' + norm.__name__,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': 'torch.nn.LeakyReLU',
            'nonlin_kwargs': {'inplace': True},
        },
        '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin'),
    }

    return architecture_kwargs
