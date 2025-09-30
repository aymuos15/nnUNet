"""
Network architecture building utilities.

This module provides high-level functions for building network architectures
from plans. These functions are used by both training and inference.
"""

from typing import Union, List, Tuple
from torch import nn

from nnunetv2.utilities.planning.get_network_from_plans import get_network_from_plans


def build_network_architecture(architecture_class_name: str,
                               arch_init_kwargs: dict,
                               arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                               num_input_channels: int,
                               num_output_channels: int,
                               enable_deep_supervision: bool = True) -> nn.Module:
    """
    Build network architecture according to plans.
    
    This is where you build the architecture according to the plans. There is no obligation to use
    get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
    you want. Even ignore the plans and just return something static (as long as it can process the requested
    patch size) but don't bug us with your bugs arising from fiddling with this :-P
    
    This function is called in both training and inference! This is needed so that all network architecture
    variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
    training, so if you change the network architecture during training by deriving a new trainer class then
    inference will know about it).

    If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
    > label_manager = plans_manager.get_label_manager(dataset_json)
    > label_manager.num_segmentation_heads
    (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
    the number of outputs is != the number of classes. Also there is the ignore label for which no output
    should be generated. label_manager takes care of all that for you.)
    
    Args:
        architecture_class_name: Name of the architecture class
        arch_init_kwargs: Keyword arguments for architecture initialization
        arch_init_kwargs_req_import: Required imports for architecture
        num_input_channels: Number of input channels
        num_output_channels: Number of output channels (segmentation heads)
        enable_deep_supervision: Whether to enable deep supervision (default: True)
    
    Returns:
        nn.Module: The constructed network architecture
    """
    return get_network_from_plans(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=enable_deep_supervision)
