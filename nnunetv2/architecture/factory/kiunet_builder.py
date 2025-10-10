"""
KiU-Net architecture builders.

This module contains builder functions for the DynamicKiUNet architecture variants.
KiU-Net is a dual-branch architecture that combines:
- U-Net branch: Standard encoder-decoder with downsampling
- Ki-Net branch: Inverted encoder with upsampling (overcomplete)
- Cross-Refinement Blocks: Feature exchange between branches

Builder variants:
- build_kiunet_maxpool: Uses MaxPool downsampling (matches original paper)
- build_kiunet_conv: Uses strided convolutions (faster, more modern)
- build_kiunet_minimal: Reduced memory footprint (50% features)
"""

import pydoc
from typing import Union, List, Tuple
from torch import nn

from nnunetv2.architecture.custom import DynamicKiUNet


def build_kiunet_maxpool(
    architecture_class_name: str,
    arch_init_kwargs: dict,
    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True
) -> nn.Module:
    """
    Build DynamicKiUNet with MaxPool downsampling.

    This builder uses MaxPool for downsampling to match the original KiU-Net paper.
    It preserves all architecture configuration from the plans while swapping
    to the KiU-Net architecture.

    Args:
        architecture_class_name: Ignored (we always use DynamicKiUNet)
        arch_init_kwargs: Architecture kwargs from plans
        arch_init_kwargs_req_import: Keys that need module import
        num_input_channels: Number of input channels (modalities)
        num_output_channels: Number of output classes
        enable_deep_supervision: Whether to use deep supervision

    Returns:
        DynamicKiUNet with MaxPool downsampling

    Example:
        >>> from nnunetv2.training.configs import TrainerConfig
        >>> config = TrainerConfig(
        ...     name="kiunet_maxpool",
        ...     network_builder=build_kiunet_maxpool
        ... )
    """
    # Import required modules for arch_init_kwargs
    architecture_kwargs = dict(**arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if architecture_kwargs[key] is not None:
            architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

    # Build DynamicKiUNet with MaxPool (matches original paper)
    network = DynamicKiUNet(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        pool_type='max',  # Original KiU-Net uses MaxPool
        **architecture_kwargs
    )

    return network


def build_kiunet_conv(
    architecture_class_name: str,
    arch_init_kwargs: dict,
    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True
) -> nn.Module:
    """
    Build DynamicKiUNet with strided convolutions.

    This builder uses strided convolutions for downsampling instead of MaxPool.
    This is faster and more memory-efficient than MaxPool, making it better
    for modern hardware.

    Args:
        architecture_class_name: Ignored (we always use DynamicKiUNet)
        arch_init_kwargs: Architecture kwargs from plans
        arch_init_kwargs_req_import: Keys that need module import
        num_input_channels: Number of input channels (modalities)
        num_output_channels: Number of output classes
        enable_deep_supervision: Whether to use deep supervision

    Returns:
        DynamicKiUNet with strided convolution downsampling

    Example:
        >>> from nnunetv2.training.configs import TrainerConfig
        >>> config = TrainerConfig(
        ...     name="kiunet_conv",
        ...     network_builder=build_kiunet_conv
        ... )
    """
    # Import required modules for arch_init_kwargs
    architecture_kwargs = dict(**arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if architecture_kwargs[key] is not None:
            architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

    # Build DynamicKiUNet with strided convolutions (faster, more modern)
    network = DynamicKiUNet(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        pool_type='conv',  # Strided convolutions (faster)
        **architecture_kwargs
    )

    return network


def build_kiunet_minimal(
    architecture_class_name: str,
    arch_init_kwargs: dict,
    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True
) -> nn.Module:
    """
    Build DynamicKiUNet with reduced memory footprint.

    This builder reduces the feature channels by 50% to fit the dual-branch
    architecture on GPUs with limited memory (e.g., 24GB). It keeps kernel
    sizes at 3x3x3 but halves features_per_stage.

    For example, if the default features are [32, 64, 128, 256], this builder
    will use [16, 32, 64, 128].

    Args:
        architecture_class_name: Ignored (we always use DynamicKiUNet)
        arch_init_kwargs: Architecture kwargs from plans
        arch_init_kwargs_req_import: Keys that need module import
        num_input_channels: Number of input channels (modalities)
        num_output_channels: Number of output classes
        enable_deep_supervision: Whether to use deep supervision

    Returns:
        DynamicKiUNet with 50% reduced feature channels

    Example:
        >>> from nnunetv2.training.configs import TrainerConfig
        >>> config = TrainerConfig(
        ...     name="kiunet_minimal",
        ...     batch_size=1,  # Recommend batch_size=1 for large volumes
        ...     network_builder=build_kiunet_minimal
        ... )
    """
    # Import required modules for arch_init_kwargs
    architecture_kwargs = dict(**arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if architecture_kwargs[key] is not None:
            architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

    # Reduce feature channels by 50% (dual-branch uses ~2x memory)
    # Default is [32, 64, 128, 256], reduce to [16, 32, 64, 128]
    architecture_kwargs['features_per_stage'] = [
        f // 2 for f in architecture_kwargs['features_per_stage']
    ]

    # Build DynamicKiUNet with reduced features but normal kernel sizes
    network = DynamicKiUNet(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        pool_type='conv',  # Strided convolutions for efficiency
        **architecture_kwargs
    )

    return network
