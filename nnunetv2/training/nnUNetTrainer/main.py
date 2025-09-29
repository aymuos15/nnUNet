import torch
from typing import Union

# Import our modular components
from .checkpointing.save import save_checkpoint
from .checkpointing.load import load_checkpoint
from .data.datasets import do_split, get_tr_and_val_datasets
from .data.loaders import get_dataloaders
from .initialization.network import set_deep_supervision_enabled
from .lifecycle.training import run_training
from .loss.builder import _build_loss, _get_deep_supervision_scales
from .state.initialization import setup_trainer_state, initialize_trainer
from .trainer_logging.main import print_to_log_file, print_plans, _save_debug_information
from .training.step import train_step
from .training.optimizer import configure_optimizers
from .validation.runner import perform_actual_validation
from .validation.step import validation_step


class nnUNetTrainer(object):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)
        # apex predator of grug is complexity
        # complexity bad - so we broke this down into modules!

        setup_trainer_state(self, plans, configuration, fold, dataset_json, device)

    def initialize(self):
        """Initialize trainer components (network, optimizer, loss, etc.)."""
        initialize_trainer(self)

    def _save_debug_information(self):
        return _save_debug_information(self)

    def _get_deep_supervision_scales(self):
        return _get_deep_supervision_scales(self)

    def _build_loss(self):
        return _build_loss(self)

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        return print_to_log_file(self, *args, also_print_to_console=also_print_to_console, add_timestamp=add_timestamp)

    def print_plans(self):
        return print_plans(self)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        return configure_optimizers(self)

    def do_split(self):
        """Perform dataset split - delegated to data.datasets module."""
        return do_split(self)

    def get_tr_and_val_datasets(self):
        """Get training and validation datasets - delegated to data.datasets module."""
        return get_tr_and_val_datasets(self)

    def get_dataloaders(self):
        """Get training and validation data loaders - delegated to data.loaders module."""
        return get_dataloaders(self)

    def set_deep_supervision_enabled(self, enabled: bool):
        """Enable/disable deep supervision - delegated to initialization.network module."""
        set_deep_supervision_enabled(self, enabled)

    def save_checkpoint(self, filename: str) -> None:
        save_checkpoint(self, filename)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        load_checkpoint(self, filename_or_checkpoint)

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Perform full validation - delegated to validation.runner module."""
        return perform_actual_validation(self, save_probabilities)

    def run_training(self):
        """Main training loop - delegated to lifecycle.training module."""
        return run_training(self)