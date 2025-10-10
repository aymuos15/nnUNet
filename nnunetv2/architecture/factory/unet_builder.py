"""
Default UNet architecture builder.

This module contains the builder function for the standard nnU-Net architectures
(PlainConvUNet, ResidualEncoderUNet, etc.) from dynamic_network_architectures.

The builder uses dynamic class loading via pydoc.locate to instantiate architectures
based on the plans configuration.
"""

import pydoc
import warnings
from typing import Union, List, Tuple
from torch import nn

from nnunetv2.utilities.core.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join


def build_unet(
    architecture_class_name: str,
    arch_init_kwargs: dict,
    arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True
) -> nn.Module:
    """
    Build default UNet architecture from plans configuration.

    This is the standard builder for nnU-Net's default architectures, which uses
    dynamic class loading to instantiate networks like PlainConvUNet, ResidualEncoderUNet,
    etc. from the dynamic_network_architectures package.

    The function:
    1. Locates the network class by name using pydoc
    2. Imports required modules for architecture kwargs
    3. Instantiates the network with proper configuration
    4. Calls network.initialize() if available

    Args:
        architecture_class_name: Fully qualified class name (e.g.,
            "dynamic_network_architectures.architectures.unet.PlainConvUNet")
        arch_init_kwargs: Architecture initialization kwargs from plans
        arch_init_kwargs_req_import: Keys in arch_init_kwargs that need pydoc.locate
            (e.g., ["conv_op", "norm_op", "nonlin"])
        num_input_channels: Number of input channels (modalities)
        num_output_channels: Number of output classes (segmentation heads)
        enable_deep_supervision: Whether to enable deep supervision (default: True)

    Returns:
        nn.Module: Instantiated network ready for training

    Raises:
        ImportError: If the network class cannot be found

    Example:
        >>> network = build_unet(
        ...     architecture_class_name="dynamic_network_architectures.architectures.unet.PlainConvUNet",
        ...     arch_init_kwargs={
        ...         "n_stages": 6,
        ...         "features_per_stage": [32, 64, 128, 256, 320, 320],
        ...         "conv_op": "torch.nn.modules.conv.Conv3d",
        ...         "kernel_sizes": [[3, 3, 3]] * 6,
        ...         "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        ...         # ... more kwargs
        ...     },
        ...     arch_init_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        ...     num_input_channels=4,
        ...     num_output_channels=3,
        ...     enable_deep_supervision=True
        ... )
    """
    # Import required modules for architecture kwargs
    architecture_kwargs = dict(**arch_init_kwargs)
    for key in arch_init_kwargs_req_import:
        if architecture_kwargs[key] is not None:
            architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

    # Locate the network class
    network_class = pydoc.locate(architecture_class_name)

    # Fallback: search within dynamic_network_architectures if not found
    if network_class is None:
        warnings.warn(
            f'Network class {architecture_class_name} not found. '
            f'Attempting to locate it within dynamic_network_architectures.architectures...'
        )
        import dynamic_network_architectures
        network_class = recursive_find_python_class(
            join(dynamic_network_architectures.__path__[0], "architectures"),
            architecture_class_name.split(".")[-1],
            'dynamic_network_architectures.architectures'
        )
        if network_class is not None:
            print(f'FOUND IT: {network_class}')
        else:
            raise ImportError(
                'Network class could not be found, please check/correct your plans file'
            )

    # Override deep supervision if specified
    architecture_kwargs['deep_supervision'] = enable_deep_supervision

    # Instantiate the network
    network = network_class(
        input_channels=num_input_channels,
        num_classes=num_output_channels,
        **architecture_kwargs
    )

    # Initialize weights if the network has an initialize method
    if hasattr(network, 'initialize'):
        network.apply(network.initialize)

    return network
