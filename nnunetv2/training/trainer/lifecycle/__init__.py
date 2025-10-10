"""
Trainer lifecycle management package.

This package contains modules for managing the complete training lifecycle:
- config: Configuration and initialization functions
- hooks: Training lifecycle hooks (start/end of training, epochs, etc.)
- steps: Core training and validation step execution
- orchestration: Main training orchestration loop
"""

# Configuration functions
from .config import (
    configure_optimizers,
    _set_batch_size_and_oversample,
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
    setup_output_folders,
    setup_cascaded_folders,
    copy_plans_and_dataset_json,
    ensure_output_folder_exists
)

# Lifecycle hooks
from .hooks import (
    on_train_start,
    on_train_end,
    on_train_epoch_start,
    on_train_epoch_end,
    on_validation_epoch_start,
    on_validation_epoch_end,
    on_epoch_start,
    on_epoch_end
)

# Step execution
from .steps import (
    train_step,
    validation_step
)

# Main orchestration
from .orchestration import run_training

__all__ = [
    # Config
    'configure_optimizers',
    '_set_batch_size_and_oversample',
    'configure_rotation_dummyDA_mirroring_and_inital_patch_size',
    'setup_output_folders',
    'setup_cascaded_folders',
    'copy_plans_and_dataset_json',
    'ensure_output_folder_exists',

    # Hooks
    'on_train_start',
    'on_train_end',
    'on_train_epoch_start',
    'on_train_epoch_end',
    'on_validation_epoch_start',
    'on_validation_epoch_end',
    'on_epoch_start',
    'on_epoch_end',

    # Steps
    'train_step',
    'validation_step',

    # Orchestration
    'run_training',
]
