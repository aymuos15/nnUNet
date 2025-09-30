import torch
from typing import Union

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

# Import our modular components
from .checkpointing.save import save_checkpoint
from .checkpointing.load import load_checkpoint
from .data.datasets import do_split, get_tr_and_val_datasets
from .lifecycle.training import run_training
from .loss.builder import _build_loss, _get_deep_supervision_scales
from .state.initialization import setup_trainer_state, initialize_trainer
from .trainer_logging.main import print_to_log_file, print_plans, _save_debug_information
from .training.step import train_step
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
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def do_split(self):
        """Perform dataset split - delegated to data.datasets module."""
        return do_split(self)

    def get_tr_and_val_datasets(self):
        """Get training and validation datasets - delegated to data.datasets module."""
        return get_tr_and_val_datasets(self)

    def get_dataloaders(self):
        """Get training and validation data loaders - delegated to data.loaders module."""
        from .data.loaders import get_dataloaders
        return get_dataloaders(self)

    def set_deep_supervision_enabled(self, enabled: bool):
        """Enable/disable deep supervision - delegated to initialization.network module."""
        from .initialization.network import set_deep_supervision_enabled
        set_deep_supervision_enabled(self, enabled)

    def on_train_start(self):
        """Training start hook - delegated to lifecycle.hooks module."""
        from .lifecycle.hooks import on_train_start
        on_train_start(self)

    def on_train_end(self):
        """Training end hook - delegated to lifecycle.hooks module."""
        from .lifecycle.hooks import on_train_end
        on_train_end(self)

    def on_train_epoch_start(self):
        """Training epoch start hook - delegated to lifecycle.hooks module."""
        from .lifecycle.hooks import on_train_epoch_start
        on_train_epoch_start(self)

    def train_step(self, batch: dict) -> dict:
        """Training step - delegated to training.step module."""
        return train_step(self, batch)

    def on_train_epoch_end(self, train_outputs):
        """Training epoch end hook - delegated to training.step module."""
        from .training.step import on_train_epoch_end
        on_train_epoch_end(self, train_outputs)

    def on_validation_epoch_start(self):
        """Validation epoch start hook - delegated to lifecycle.hooks module."""
        from .lifecycle.hooks import on_validation_epoch_start
        on_validation_epoch_start(self)

    def validation_step(self, batch: dict) -> dict:
        """Validation step - delegated to validation.step module."""
        return validation_step(self, batch)

    def on_validation_epoch_end(self, val_outputs):
        """Validation epoch end hook - delegated to training.step module."""
        from .training.step import on_validation_epoch_end
        on_validation_epoch_end(self, val_outputs)

    def on_epoch_start(self):
        """Epoch start hook - delegated to lifecycle.hooks module."""
        from .lifecycle.hooks import on_epoch_start
        on_epoch_start(self)

    def on_epoch_end(self):
        """Epoch end hook - delegated to lifecycle.hooks module."""
        from .lifecycle.hooks import on_epoch_end
        on_epoch_end(self)

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