"""
Custom Trainer for DynamicKiUNet Architecture

This trainer uses the DynamicKiUNet architecture instead of the default U-Net.
It's configured for quick integration testing with only 2 epochs.
"""

import torch
from typing import Union, Tuple, List
from nnunetv2.training.trainer.main import nnUNetTrainer
from nnunetv2.architecture import DynamicKiUNet


class nnUNetTrainerKiUNet(nnUNetTrainer):
    """
    nnU-Net Trainer using DynamicKiUNet architecture.

    This trainer replaces the standard U-Net with KiU-Net, which combines:
    - U-Net branch: Standard encoder-decoder with downsampling
    - Ki-Net branch: Inverted encoder with upsampling (overcomplete)
    - Cross-Refinement Blocks: Feature exchange between branches

    For quick testing, max_num_epochs is set to 2.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'),
                 trainer_config=None):
        """Initialize the KiU-Net trainer."""
        super().__init__(plans, configuration, fold, dataset_json, device, trainer_config)

        # Override for quick integration testing
        self.num_epochs = 2

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True
    ) -> torch.nn.Module:
        """
        Build DynamicKiUNet instead of the default architecture.

        This method overrides the base trainer's network building to use KiU-Net.
        All configuration from plans is preserved, but the architecture is swapped.

        Args:
            architecture_class_name: Ignored (we use DynamicKiUNet)
            arch_init_kwargs: Architecture kwargs from plans
            arch_init_kwargs_req_import: Keys that need module import
            num_input_channels: Number of input channels
            num_output_channels: Number of output classes
            enable_deep_supervision: Whether to use deep supervision

        Returns:
            DynamicKiUNet model instance
        """
        # Import required modules for arch_init_kwargs
        import pydoc
        architecture_kwargs = dict(**arch_init_kwargs)
        for key in arch_init_kwargs_req_import:
            if architecture_kwargs[key] is not None:
                architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

        # Use MaxPool to match original KiU-Net paper more closely
        # Change to pool_type='conv' for strided convolutions (faster)
        pool_type = 'max'

        # Build DynamicKiUNet with configuration from plans
        network = DynamicKiUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            pool_type=pool_type,
            **architecture_kwargs
        )

        return network


class nnUNetTrainerKiUNetConv(nnUNetTrainerKiUNet):
    """
    nnU-Net Trainer using DynamicKiUNet with strided convolutions.

    Identical to nnUNetTrainerKiUNet but uses strided convolutions
    instead of MaxPool for downsampling. This is faster but differs
    slightly from the original paper.
    """

    def build_network_architecture(
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True
    ) -> torch.nn.Module:
        """Build DynamicKiUNet with strided convolutions."""
        import pydoc
        architecture_kwargs = dict(**arch_init_kwargs)
        for key in arch_init_kwargs_req_import:
            if architecture_kwargs[key] is not None:
                architecture_kwargs[key] = pydoc.locate(architecture_kwargs[key])

        # Use strided convolutions (faster, more modern)
        pool_type = 'conv'

        network = DynamicKiUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            pool_type=pool_type,
            **architecture_kwargs
        )

        return network
