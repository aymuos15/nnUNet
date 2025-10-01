# Training

This module contains all training infrastructure for nnU-Net, including the training loop, loss functions, learning rate scheduling, and configuration system for custom architectures.

## Overview

The training module orchestrates:

- **Training loop** - Epoch management, validation, checkpointing
- **Loss computation** - Dice, Cross-Entropy, and custom losses (Blob Dice, Region Dice)
- **Data augmentation** - On-the-fly transformations
- **Learning rate scheduling** - Polynomial LR decay
- **Configuration system** - Custom architecture integration via `TrainerConfig`
- **Distributed training** - Multi-GPU via DDP

## Directory Structure

```
training/
├── configs/               # TrainerConfig for custom architectures
├── execution/             # Training execution and multi-GPU utilities
├── logging/               # Training loggers and metrics tracking
├── losses/                # Loss function implementations
├── lr_scheduler/          # Learning rate schedulers
├── runtime_utils/         # Runtime utilities (progress bars, memory stats)
└── trainer/               # Trainer classes (training loop orchestration)
```

## Key Components

### Trainer (`trainer/`)

The `nnUNetTrainer` class orchestrates the entire training process:

**Main responsibilities**:
- Load preprocessed data or preprocess on-the-fly
- Build network architecture from plans
- Initialize optimizer and learning rate scheduler
- Run training loop with automatic data augmentation
- Perform validation and save checkpoints
- Log metrics (loss, Dice scores, learning rate)

**Key methods**:
- `run_training()` - Main training loop
- `train_step()` - Single training iteration
- `validation_step()` - Validation on a single case
- `on_epoch_end()` - Checkpointing and logging
- `build_network_architecture()` - Construct the model

**Variants**:
- `nnUNetTrainer` - Default trainer with standard U-Net
- Custom trainers - Override methods for custom behavior

### Loss Functions (`losses/`)

nnU-Net uses a combination of Dice loss and Cross-Entropy loss by default.

#### Standard Losses

**Dice + CE Loss** (`losses/compound_losses.py`):
```python
loss = dice_loss + cross_entropy_loss
```

- **Dice loss**: Optimizes overlap between prediction and ground truth
- **Cross-Entropy loss**: Pixel-wise classification loss
- Both use soft labels (no argmax), computed on logits
- Deep supervision: weighted sum across multiple resolutions

#### Custom Losses

**Blob Dice Loss** (`losses/implementations/blob_dice.py`):

Instance-aware Dice loss that treats each connected component as a separate instance:

- Computes Dice per connected component (blob)
- Averages across all blobs
- Useful for: counting tasks, instance segmentation
- Modes:
  - `batch_dice=False`: Mean over blobs, then over batch
  - `batch_dice=True`: Pool all blobs across batch, compute Dice once

**Region Dice Loss** (`losses/implementations/region_dice.py`):

Multi-region Dice loss with different penalties per region:

- Segments image into regions (e.g., boundary vs interior)
- Computes Dice per region
- Weights regions differently (e.g., higher weight on boundaries)
- Useful for: emphasizing difficult regions, boundary refinement

**Usage**:

Via `TrainerConfig`:
```python
from nnunetv2.training.configs import TrainerConfig
from nnunetv2.training.losses import DC_and_Blob_CE_loss

config = TrainerConfig(
    loss=DC_and_Blob_CE_loss({'batch_dice': True}, {})
)
```

See `CUSTOM_ARCH_INFO.md` for detailed integration examples.

### Configuration System (`configs/`)

The `TrainerConfig` system enables custom architectures and losses without subclassing trainers:

**Purpose**: Decouple architecture/loss configuration from training logic.

**Structure**:
```python
@dataclass
class TrainerConfig:
    architecture: Type[nn.Module]  # Architecture class
    architecture_kwargs: dict      # Constructor kwargs
    loss: Type[nn.Module]          # Loss function class
    loss_kwargs: dict              # Loss constructor kwargs
```

**Usage**:

1. Define config in `configs/`:
```python
from nnunetv2.training.configs import TrainerConfig
from nnunetv2.architecture.custom import DynamicKiUNet
from nnunetv2.training.losses import DC_and_CE_loss

kiunet_config = TrainerConfig(
    architecture=DynamicKiUNet,
    architecture_kwargs={
        'pool_type': 'max',
        'deep_supervision': True
    },
    loss=DC_and_CE_loss,
    loss_kwargs={'soft_dice_kwargs': {}, 'ce_kwargs': {}}
)
```

2. Register config:
```python
from nnunetv2.training.configs import TRAINER_CONFIGS

TRAINER_CONFIGS['kiunet'] = kiunet_config
```

3. Use via CLI:
```bash
nnUNetv2_train 001 3d_fullres 0 -tr kiunet
```

### Learning Rate Scheduler (`lr_scheduler/`)

**PolynomialLRScheduler** (`lr_scheduler/polylr.py`):

Default scheduler used by nnU-Net:

```python
lr = initial_lr * (1 - epoch / max_epochs) ^ exponent
```

- **Exponent**: Typically 0.9
- **Initial LR**: 0.01
- **Warm-up**: Optional (not used by default)
- **Per-iteration update**: LR updated every training step, not per epoch

### Data Augmentation

nnU-Net uses extensive on-the-fly data augmentation via `batchgenerators`:

**Spatial augmentations**:
- Random rotation
- Random scaling
- Random elastic deformation
- Mirroring

**Intensity augmentations**:
- Brightness and contrast adjustment
- Gamma correction
- Gaussian noise
- Gaussian blur

**Augmentation parameters** are automatically configured during experiment planning.

### Deep Supervision

Deep supervision adds auxiliary losses at intermediate decoder resolutions:

**Benefits**:
- Better gradient flow to early layers
- Multi-scale learning
- Improved convergence

**Implementation**:
- Network outputs a list: `[full_res, half_res, quarter_res, ...]`
- Loss computed at each resolution
- Weighted sum: `loss = w0*loss0 + w1*loss1 + ...`
- Weights decay exponentially: `w_i = 0.5^i` (rescaled to sum to 1)

**Enable**:
```python
deep_supervision = True  # In architecture kwargs
```

### Distributed Training (`execution/`)

Multi-GPU training via PyTorch DistributedDataParallel (DDP):

**Usage**:
```bash
nnUNetv2_train 001 3d_fullres 0 --c
```

**Features**:
- Automatic device placement
- Gradient synchronization across GPUs
- Model checkpointing from rank 0 only

**Environment variables**:
- `CUDA_VISIBLE_DEVICES` - Specify GPUs

## Usage

### CLI Training

**Basic training**:
```bash
nnUNetv2_train DATASET_ID CONFIG FOLD
```

**Examples**:
```bash
# 3D full resolution, fold 0
nnUNetv2_train 001 3d_fullres 0

# 2D configuration, fold all (0-4)
nnUNetv2_train 001 2d all

# Custom trainer config
nnUNetv2_train 001 3d_fullres 0 -tr kiunet

# Continue from checkpoint
nnUNetv2_train 001 3d_fullres 0 -c

# Use pretrained checkpoint
nnUNetv2_train 001 3d_fullres 0 -pretrained_weights /path/to/checkpoint_final.pth
```

**Flags**:
- `--npz` - Use compressed NPZ format for data loading
- `-c` / `--continue_training` - Resume from latest checkpoint
- `-p` / `--plans_identifier` - Use custom plans file
- `-tr` / `--trainer` - Use custom trainer or config
- `-pretrained_weights` - Load pretrained weights
- `--val_best` - Validate on best checkpoint instead of final

### Programmatic API

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

# Initialize trainer
trainer = nnUNetTrainer(
    dataset_id=1,
    configuration='3d_fullres',
    fold=0,
    plans_identifier='nnUNetPlans',
    device='cuda'
)

# Run training
trainer.run_training()
```

### Custom Trainer

For advanced customization beyond `TrainerConfig`:

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class MyTrainer(nnUNetTrainer):
    def configure_optimizers(self):
        """Override optimizer configuration."""
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=0.001
        )
    
    def train_step(self, batch):
        """Override training step logic."""
        data, target = batch['data'], batch['target']
        
        # Custom forward pass
        output = self.network(data)
        
        # Custom loss computation
        loss = self.my_custom_loss(output, target)
        
        return loss
```

Use:
```bash
nnUNetv2_train 001 3d_fullres 0 -tr MyTrainer
```

## Training Outputs

Training artifacts are saved in `nnUNet_results/DatasetXXX_Name/TRAINER_NAME__PLANS__CONFIG/`:

```
nnUNet_results/
  Dataset001_Name/
    nnUNetTrainer__nnUNetPlans__3d_fullres/
      fold_0/
        checkpoint_best.pth        # Best validation checkpoint
        checkpoint_final.pth       # Final epoch checkpoint
        checkpoint_latest.pth      # Latest checkpoint (for resuming)
        training_log.txt           # Training log
        validation_raw/            # Raw validation predictions
        progress.png               # Training curves
        debug.json                 # Debug info (network architecture, etc.)
      fold_1/
      ...
      fold_4/
```

### Checkpoint Contents

Each `.pth` checkpoint contains:

```python
{
    'network_weights': state_dict,
    'optimizer_state': optimizer.state_dict(),
    'grad_scaler_state': scaler.state_dict(),  # For mixed precision
    'epoch': current_epoch,
    'current_epoch': current_epoch,
    'init': trainer initialization args
}
```

### Training Logs

**training_log.txt**:
- Per-epoch metrics: train loss, val Dice scores, learning rate
- Validation performance per class
- Best epoch information

**progress.png**:
- Loss curves (train/val)
- Dice score progression
- Learning rate schedule

## Training Tips

### Memory Management

**CUDA out of memory**:
1. Reduce `batch_size` in plans file
2. Reduce `patch_size` in plans file
3. Enable gradient checkpointing (not default, requires custom trainer)
4. Reduce `features_per_stage` (fewer channels)

**Typical memory usage** (per GPU):
- 2D: 4-8 GB
- 3D full res: 10-24 GB
- 3D low res: 6-12 GB

### Training Time

**Typical training duration**:
- 2D: 1-3 days (250-1000 epochs)
- 3D: 2-7 days (250-1000 epochs)

**Depends on**:
- Dataset size
- Patch size and batch size
- Network architecture
- GPU speed

**Epochs**: nnU-Net trains for 1000 epochs by default, with validation every 50 epochs.

### Validation

**Validation frequency**: Every 50 epochs by default.

**Validation metrics**:
- Dice score per class
- Mean Dice across classes

**Best model**: Determined by mean Dice score across validation cases and classes.

### Checkpointing

**Checkpoint saves**:
- `checkpoint_latest.pth` - After every epoch
- `checkpoint_best.pth` - When validation Dice improves
- `checkpoint_final.pth` - After training completes

**Resume training**:
```bash
nnUNetv2_train 001 3d_fullres 0 -c
```

### Debugging

**Verify network architecture**:
Check `debug.json` for:
- Network parameter count
- Architecture configuration
- Plans used

**Monitor training**:
```bash
tail -f nnUNet_results/.../fold_0/training_log.txt
```

**Visualize predictions**:
```bash
# Predictions saved in validation_raw/
ls nnUNet_results/.../fold_0/validation_raw/
```

## Advanced Topics

### Pretraining & Fine-Tuning

1. **Train on large dataset**:
```bash
nnUNetv2_train 001 3d_fullres 0
```

2. **Transfer plans** to target dataset:
```bash
nnUNetv2_move_plans_between_datasets -s 001 -t 002 -sp nnUNetPlans -tp nnUNetPlans_pretrained
```

3. **Fine-tune** on target dataset:
```bash
nnUNetv2_train 002 3d_fullres 0 -p nnUNetPlans_pretrained -pretrained_weights /path/to/checkpoint_final.pth
```

### Manual Data Splits

Create custom cross-validation splits:

1. Create `splits_final.json` in dataset folder:
```json
[
    {
        "train": ["case_001", "case_002", ...],
        "val": ["case_010", "case_011"]
    },
    ...
]
```

2. Train normally:
```bash
nnUNetv2_train 001 3d_fullres 0
```

nnU-Net will use your custom splits instead of automatic 5-fold CV.

### Ignore Label

Train with incomplete annotations using ignore label:

1. Set `ignore_label` in `dataset.json`:
```json
{
    "labels": {
        "background": 0,
        "tumor": 1,
        "ignore": 255
    },
    "ignore_label": 255
}
```

2. Train normally - loss is not computed on ignore label regions.

See [Sparse Annotations Reference](../documentation/reference/ignore_label.md).

## See Also

- [Custom Architecture Guide](../CUSTOM_ARCH_INFO.md) - KiU-Net, UIU-Net, Blob Dice, Region Dice
- [Pretraining & Fine-Tuning](../documentation/reference/pretraining_and_finetuning.md) - Transfer learning workflows
- [Manual Data Splits](../documentation/reference/manual_data_splits.md) - Custom CV splits
- [Sparse Annotations](../documentation/reference/ignore_label.md) - Ignore label usage
