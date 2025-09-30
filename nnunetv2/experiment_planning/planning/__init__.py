"""Planning-related utilities for nnUNetv2."""

from .label_handling import (
    LabelManager,
    convert_labelmap_to_one_hot,
    determine_num_input_channels,
    get_labelmanager_class_from_plans,
)
from .plans_handler import ConfigurationManager, PlansManager

# Backward compatibility: network-related functions moved to nnunetv2.architecture
from nnunetv2.architecture import get_network_from_plans, build_network_architecture

__all__ = [
    "LabelManager",
    "convert_labelmap_to_one_hot",
    "determine_num_input_channels",
    "get_labelmanager_class_from_plans",
    "ConfigurationManager",
    "PlansManager",
    "get_network_from_plans",
    "build_network_architecture",
]
