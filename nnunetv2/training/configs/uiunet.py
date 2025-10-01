"""
UIU-Net architecture configs for nnU-Net.

Provides configs that use the DynamicUIUNet3D architecture with nested RSU blocks.
"""

from typing import Union, List, Tuple
import torch.nn as nn
from nnunetv2.training.configs.base import TrainerConfig, register_config
from nnunetv2.architecture.custom.uiunet import DynamicUIUNet3D


def build_uiunet(
    architecture_class_name: str,
    arch_init_kwargs: dict,
    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True
) -> nn.Module:
    """
    Network builder for DynamicUIUNet3D.

    Uses standard RSU heights for full architecture fidelity.
    """
    import pydoc
    architecture_kwargs = dict(**arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if architecture_kwargs[key] is not None:
            architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

    # Build DynamicUIUNet3D
    network = DynamicUIUNet3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        rsu_heights=None,  # Auto-calculate: starts at 7, decreases per stage
        minimal=False,
        **architecture_kwargs
    )

    return network


def build_uiunet_minimal(
    architecture_class_name: str,
    arch_init_kwargs: dict,
    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True
) -> nn.Module:
    """
    Network builder for DynamicUIUNet3D with reduced memory footprint.

    Uses:
    - Reduced RSU heights (starts at 5 instead of 7)
    - 50% feature channels (dual-nested structure is very memory intensive)
    """
    import pydoc
    architecture_kwargs = dict(**arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if architecture_kwargs[key] is not None:
            architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

    # Reduce feature channels by 50% (nested U-Nets use significant memory)
    architecture_kwargs['features_per_stage'] = [f // 2 for f in architecture_kwargs['features_per_stage']]

    # Build DynamicUIUNet3D with minimal settings
    network = DynamicUIUNet3D(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        deep_supervision=enable_deep_supervision,
        rsu_heights=None,  # Auto-calculate with minimal=True: starts at 5
        minimal=True,  # Use reduced RSU heights
        **architecture_kwargs
    )

    return network


# UIU-Net config (1 epoch for testing)
UIUNET_CONFIG = TrainerConfig(
    name="uiunet",
    description="DynamicUIUNet3D architecture with nested RSU blocks (1 epoch for testing)",
    num_epochs=1,
    network_builder=build_uiunet,
)

# UIU-Net minimal config for 24GB GPUs (1 epoch for testing)
UIUNET_MINIMAL_CONFIG = TrainerConfig(
    name="uiunet_minimal",
    description="DynamicUIUNet3D for 24GB GPU (batch_size=1, reduced RSU heights, 50% features, 1 epoch)",
    num_epochs=1,
    batch_size=1,
    network_builder=build_uiunet_minimal,
)

# Register all configs
register_config(UIUNET_CONFIG)
register_config(UIUNET_MINIMAL_CONFIG)
