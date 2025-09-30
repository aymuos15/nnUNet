# nnUNet Trainer Configurations

This directory contains the new config-based trainer system that replaces the old inheritance-based trainer variants.

## Overview

Instead of creating new trainer classes for every variation (e.g., `nnUNetTrainerNoMirroring`, `nnUNetTrainer_1000epochs`, etc.), you can now use configuration objects that modify the base trainer behavior.

## Usage

### Command Line

Use the `-tr` flag to specify a config name:

```bash
# Use base config (default)
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD

# Use a preset config
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr no_mirroring
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr 1000epochs
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr adam
```

### Available Preset Configs

#### Mirroring
- `no_mirroring` - Disable mirroring during training and inference
- `mirror_01` - Only mirror along axes 0 and 1

#### Training Length (Epochs)
- `1epochs`, `5epochs`, `10epochs`, `20epochs`, `50epochs`, `100epochs`, `250epochs`, `500epochs`, `750epochs`, `1000epochs`, `1500epochs`, `2000epochs`, `4000epochs`, `8000epochs`
- Combined: `1000epochs_no_mirroring`, etc.

#### Optimizers
- `adam` - Use AdamW optimizer
- `vanilla_adam` - Use vanilla Adam optimizer
- `adam_1e3` - AdamW with lr=1e-3
- `adam_3e4` - AdamW with lr=3e-4
- `vanilla_adam_1e3` - Vanilla Adam with lr=1e-3
- `vanilla_adam_3e4` - Vanilla Adam with lr=3e-4

#### Loss Functions
- `dice_loss` - Use Dice loss only (no CE/BCE)
- `dice_ce_no_smooth` - Dice+CE loss with smooth=0

#### Data Augmentation
- `no_da` - Disable all data augmentation
- `no_dummy_2d_da` - Disable dummy 2D data augmentation

#### LR Schedulers
- `cosine_annealing` - Use cosine annealing LR scheduler

### Creating Custom Configs

#### Simple Configs

Create a new file in `presets/` directory:

```python
from nnunetv2.training.configs import TrainerConfig, register_config

MY_CONFIG = TrainerConfig(
    name="my_custom_config",
    description="My custom configuration",
    num_epochs=500,
    initial_lr=1e-3,
    mirror_axes=None
)
register_config(MY_CONFIG)
```

#### Advanced Configs with Custom Builders

For more complex modifications (optimizer, loss, transforms, network):

```python
from nnunetv2.training.configs import TrainerConfig, register_config

def my_optimizer_builder(trainer_instance):
    # Custom optimizer logic
    optimizer = ...
    lr_scheduler = ...
    return optimizer, lr_scheduler

def my_loss_builder(trainer_instance):
    # Custom loss logic
    loss = ...
    return loss

MY_ADVANCED_CONFIG = TrainerConfig(
    name="my_advanced_config",
    description="Advanced custom configuration",
    num_epochs=500,
    optimizer_builder=my_optimizer_builder,
    loss_builder=my_loss_builder
)
register_config(MY_ADVANCED_CONFIG)
```

## Configuration Parameters

### Simple Parameters
- `num_epochs` - Number of training epochs
- `initial_lr` - Initial learning rate
- `weight_decay` - Weight decay for optimizer
- `mirror_axes` - Axes for mirroring augmentation
- `inference_allowed_mirroring_axes` - Axes for test-time augmentation
- `do_dummy_2d_data_aug` - Enable/disable dummy 2D augmentation
- `enable_deep_supervision` - Enable/disable deep supervision
- `batch_size` - Batch size
- `oversample_foreground_percent` - Foreground oversampling percentage
- `save_every` - Checkpoint save frequency

### Strategy Overrides (Callables)
- `optimizer_builder(trainer_instance)` - Custom optimizer configuration
- `loss_builder(trainer_instance)` - Custom loss function
- `training_transforms_builder(...)` - Custom training transforms
- `network_builder(...)` - Custom network architecture

## Migration from Old Variants

Old variants have been removed. Here's how to migrate:

| Old Variant | New Config |
|------------|------------|
| `nnUNetTrainerNoMirroring` | `-tr no_mirroring` |
| `nnUNetTrainer_1000epochs` | `-tr 1000epochs` |
| `nnUNetTrainerAdam` | `-tr adam` |
| `nnUNetTrainerDiceLoss` | `-tr dice_loss` |
| `nnUNetTrainerNoDA` | `-tr no_da` |
| `nnUNetTrainerCosAnneal` | `-tr cosine_annealing` |

## Architecture

```
configs/
├── base.py              # TrainerConfig dataclass + registry
├── presets/             # Preset configurations
│   ├── mirroring.py
│   ├── epochs.py
│   ├── optimizers.py
│   ├── losses.py
│   ├── data_augmentation.py
│   └── lr_schedulers.py
└── README.md            # This file
```