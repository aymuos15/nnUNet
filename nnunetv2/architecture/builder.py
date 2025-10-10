"""
Network architecture building utilities.

This module provides high-level functions for building network architectures
from plans. These functions are used by both training and inference.

The default implementation uses the factory module's build_unet() function,
but custom trainers can override this by specifying a custom network_builder
in TrainerConfig.
"""

from typing import Union, List, Tuple
from torch import nn

from nnunetv2.architecture.factory import build_unet


def build_network_architecture(architecture_class_name: str,
                               arch_init_kwargs: dict,
                               arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                               num_input_channels: int,
                               num_output_channels: int,
                               enable_deep_supervision: bool = True) -> nn.Module:
    """
    Build network architecture according to plans.

    This is the default builder used by nnU-Net, which delegates to the factory
    module's build_unet() function. The factory module provides a clean separation
    of different architecture builders (unet_builder, kiunet_builder, etc.).

    This function is called in both training and inference! This is needed so that all
    network architecture variants can be loaded at inference time (inference will use the
    same nnUNetTrainer that was used for training, so if you change the network architecture
    during training by deriving a new trainer class then inference will know about it).

    To use custom architectures, specify a custom network_builder in TrainerConfig instead
    of subclassing the trainer.

    If you need to know how many segmentation outputs your custom architecture needs to have,
    use the following snippet:
    > label_manager = plans_manager.get_label_manager(dataset_json)
    > label_manager.num_segmentation_heads
    (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
    the number of outputs is != the number of classes. Also there is the ignore label for which no output
    should be generated. label_manager takes care of all that for you.)

    Args:
        architecture_class_name: Name of the architecture class (e.g.,
            "dynamic_network_architectures.architectures.unet.PlainConvUNet")
        arch_init_kwargs: Keyword arguments for architecture initialization
        arch_init_kwargs_req_import: Keys in arch_init_kwargs that need pydoc.locate
        num_input_channels: Number of input channels (modalities)
        num_output_channels: Number of output channels (segmentation heads)
        enable_deep_supervision: Whether to enable deep supervision (default: True)

    Returns:
        nn.Module: The constructed network architecture

    Example:
        >>> # Default usage (called by trainer)
        >>> network = build_network_architecture(
        ...     architecture_class_name="dynamic_network_architectures.architectures.unet.PlainConvUNet",
        ...     arch_init_kwargs={...},
        ...     arch_init_kwargs_req_import=["conv_op", "norm_op"],
        ...     num_input_channels=4,
        ...     num_output_channels=3
        ... )

        >>> # Custom architecture via TrainerConfig
        >>> from nnunetv2.architecture.factory import build_kiunet_conv
        >>> from nnunetv2.training.configs import TrainerConfig
        >>> config = TrainerConfig(
        ...     name="my_kiunet",
        ...     network_builder=build_kiunet_conv  # Override default builder
        ... )
    """
    return build_unet(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision)
