"""
KiU-Net architecture configs for nnU-Net.

Provides configs that use the DynamicKiUNet architecture instead of the default U-Net.
"""

from typing import Union, List, Tuple
import torch.nn as nn
from nnunetv2.training.configs.base import TrainerConfig, register_config
from nnunetv2.architecture import DynamicKiUNet


def build_kiunet_maxpool(
    architecture_class_name: str,
    arch_init_kwargs: dict,
    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True
) -> nn.Module:
    """
    Network builder for DynamicKiUNet with MaxPool downsampling.

    Uses MaxPool to match the original KiU-Net paper.
    """
    # Import required modules for arch_init_kwargs
    import pydoc
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
    Network builder for DynamicKiUNet with strided convolutions.

    Uses strided convolutions for faster training (modern approach).
    """
    import pydoc
    architecture_kwargs = dict(**arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if architecture_kwargs[key] is not None:
            architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

    # Build DynamicKiUNet with strided convolutions
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
    Network builder for DynamicKiUNet with reduced memory footprint.

    Uses smaller kernel sizes (1x1x1), reduced feature channels, and strided convolutions.
    """
    import pydoc
    architecture_kwargs = dict(**arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if architecture_kwargs[key] is not None:
            architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

    # Reduce kernel sizes to 1x1x1 for minimal memory usage
    n_stages = architecture_kwargs['n_stages']
    architecture_kwargs['kernel_sizes'] = [[1, 1, 1]] * n_stages

    # Reduce feature channels by 50% (dual-branch uses ~2x memory)
    # Default is [32, 64, 128, 256], reduce to [16, 32, 64, 128]
    architecture_kwargs['features_per_stage'] = [f // 2 for f in architecture_kwargs['features_per_stage']]

    # Build DynamicKiUNet with minimal kernel sizes and features
    network = DynamicKiUNet(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        pool_type='conv',  # Strided convolutions
        **architecture_kwargs
    )

    return network


# KiU-Net config with MaxPool (2 epochs for testing)
KIUNET_CONFIG = TrainerConfig(
    name="kiunet",
    description="DynamicKiUNet architecture with MaxPool downsampling (2 epochs for testing)",
    num_epochs=2,
    network_builder=build_kiunet_maxpool,
)

# KiU-Net config with strided convolutions (2 epochs for testing)
KIUNET_CONV_CONFIG = TrainerConfig(
    name="kiunet_conv",
    description="DynamicKiUNet architecture with strided convolutions (2 epochs for testing)",
    num_epochs=2,
    network_builder=build_kiunet_conv,
)

# Minimal config for low-memory GPUs (< 4GB)
KIUNET_MINIMAL_CONFIG = TrainerConfig(
    name="kiunet_minimal",
    description="DynamicKiUNet minimal memory (batch_size=1, 1x1x1 kernels, 50% features, strided conv, 2 epochs)",
    num_epochs=2,
    batch_size=1,
    network_builder=build_kiunet_minimal,  # Reduced kernel sizes and feature channels
)

# Register all configs
register_config(KIUNET_CONFIG)
register_config(KIUNET_CONV_CONFIG)
register_config(KIUNET_MINIMAL_CONFIG)
