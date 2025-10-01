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

    Uses 3x3x3 kernels (same as default), reduced feature channels, and strided convolutions.
    """
    import pydoc
    architecture_kwargs = dict(**arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if architecture_kwargs[key] is not None:
            architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

    # Keep kernel sizes at 3x3x3 (no reduction needed for 24GB GPU)
    # Reduce feature channels by 50% (dual-branch uses ~2x memory)
    # Default is [32, 64, 128, 256], reduce to [16, 32, 64, 128]
    architecture_kwargs['features_per_stage'] = [f // 2 for f in architecture_kwargs['features_per_stage']]

    # Build DynamicKiUNet with reduced features but normal kernel sizes
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

# Optimal config for 24GB GPUs (reduced features to fit dual-branch architecture)
KIUNET_MINIMAL_CONFIG = TrainerConfig(
    name="kiunet_minimal",
    description="DynamicKiUNet for 24GB GPU (batch_size=1, 3x3x3 kernels, 50% features, strided conv, 1000 epochs)",
    num_epochs=1000,
    batch_size=1,
    network_builder=build_kiunet_minimal,  # Reduced feature channels only
)

# Optimized config for 24GB GPUs (production use)
KIUNET_LARGE_CONFIG = TrainerConfig(
    name="kiunet_large",
    description="DynamicKiUNet optimized for 24GB GPU (batch_size=1, 3x3x3 kernels, full features, strided conv)",
    num_epochs=1000,
    batch_size=1,
    network_builder=build_kiunet_conv,  # Use strided conv for efficiency
)

# Register all configs
register_config(KIUNET_CONFIG)
register_config(KIUNET_CONV_CONFIG)
register_config(KIUNET_MINIMAL_CONFIG)
register_config(KIUNET_LARGE_CONFIG)
