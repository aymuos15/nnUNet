# Advanced Usage

This guide covers advanced topics: custom configurations, manual plan editing, extending nnU-Net, and specialized training scenarios.

## Custom Training Configurations

### Using Trainer Configs

Trainer configs allow you to customize architecture, loss function, epochs, and other training parameters without modifying core code.

**Example: Train with a custom architecture (KiU-Net)**:

```bash
nnUNetv2_train 001 3d_fullres 0 -tr kiunet_minimal
```

The `-tr` flag specifies a trainer config. Available configs are in `nnunetv2/training/configs/`.

**Built-in custom configs**:
- `kiunet`, `kiunet_minimal`: KiU-Net dual-branch architecture
- `uiunet`, `uiunet_minimal`: UIU-Net nested U-Net with cross-attention
- `blob_dice_loss`: Standard architecture with instance-aware Blob Dice loss
- `region_dice_loss`: Standard architecture with Region Dice loss

See [CUSTOM_ARCH_INFO.md](../CUSTOM_ARCH_INFO.md) for details.

### Creating Your Own Trainer Config

Create a new file in `nnunetv2/training/configs/`:

```python
from nnunetv2.training.configs.base import TrainerConfig

def my_custom_config():
    config = TrainerConfig(
        name="my_custom",
        description="Custom architecture with modified loss",
        network_builder=my_custom_network_builder,  # Function that builds your network
        loss_fn=my_custom_loss,                     # Custom loss function
        num_epochs=500,                              # Override default 1000 epochs
    )
    return config
```

Then register it in `nnunetv2/training/configs/__init__.py`:

```python
from .my_custom import my_custom_config

TRAINER_CONFIGS = {
    ...
    'my_custom': my_custom_config,
}
```

Use it:
```bash
nnUNetv2_train 001 3d_fullres 0 -tr my_custom
```

## Manual Plan Editing

After running `nnUNetv2_plan_and_preprocess`, you can manually edit the generated plans.

### Plans File Location

```
$nnUNet_preprocessed/DatasetXXX_Name/nnUNetPlans.json
```

### Common Edits

#### 1. Reduce Batch Size (for OOM issues)

```json
{
  "configurations": {
    "3d_fullres": {
      "batch_size": 2  // Change from 4 to 2
    }
  }
}
```

**When to do this**: If training crashes with CUDA out of memory.

#### 2. Change Patch Size

```json
{
  "configurations": {
    "3d_fullres": {
      "patch_size": [128, 128, 128]  // Reduce from [160, 160, 160]
    }
  }
}
```

**When to do this**: To reduce memory usage or adjust context window.

**Important**: If you change patch size, you must also adjust the network topology (number of pooling operations).

#### 3. Modify Network Topology

```json
{
  "configurations": {
    "3d_fullres": {
      "UNet_class_name": "PlainConvUNet",
      "UNet_base_num_features": 32,
      "n_conv_per_stage_encoder": [2, 2, 2, 2, 2],  // Depth = 5
      "n_conv_per_stage_decoder": [2, 2, 2, 2],
      "num_pool_per_axis": [4, 4, 4]  // Pool 4 times in each dimension
    }
  }
}
```

**When to do this**: Rarely needed, but useful for experimenting with architecture depth.

#### 4. Change Target Spacing

```json
{
  "configurations": {
    "3d_fullres": {
      "spacing": [1.0, 1.0, 1.0]  // Resample to 1mm isotropic
    }
  }
}
```

**When to do this**: To control preprocessing resolution.

**Important**: After editing plans, you must re-run preprocessing:

```bash
nnUNetv2_preprocess -d 001 -c 3d_fullres
```

## Advanced Training Options

### Training on All Folds Simultaneously

Instead of specifying a fold number, use `all`:

```bash
nnUNetv2_train 001 3d_fullres all
```

This trains folds 0, 1, 2, 3, 4 sequentially. Useful for automation.

### Training Without Cross-Validation

To train a single model on all training data (no validation split):

```bash
nnUNetv2_train 001 3d_fullres 0 --disable_checkpointing
```

**Note**: This is not standard practice. You won't have validation metrics to assess performance.

### Continuing Interrupted Training

If training is interrupted, resume with:

```bash
nnUNetv2_train 001 3d_fullres 0 --c
```

nnU-Net will load the latest checkpoint and continue.

### Validation Only

To run validation on an existing checkpoint:

```bash
nnUNetv2_train 001 3d_fullres 0 --val --npz
```

This computes validation metrics and saves softmax predictions (if `--npz` is specified).

### Custom Number of Epochs

Override in your trainer config:

```python
config = TrainerConfig(
    name="short_training",
    num_epochs=100
)
```

Or modify the trainer after instantiation (not recommended).

### Mixed Precision Training

nnU-Net uses automatic mixed precision (AMP) by default for faster training and lower memory usage.

To disable (rarely needed):
```bash
export nnUNet_use_amp=false
```

## Specialized Training Scenarios

### Region-Based Training

Train on specific regions rather than full segmentation:

1. Prepare your dataset with region labels
2. Use region-based configuration

See `documentation/reference/region_based_training.md` for details.

### Learning from Sparse Annotations

Train with incomplete labels (e.g., only some slices annotated):

1. Use ignore label (e.g., -1) for unannotated areas
2. nnU-Net will exclude these from loss computation

See `documentation/reference/ignore_label.md` for details.

### Pretraining and Fine-Tuning

Transfer learning workflow:

1. **Pretrain** on a large source dataset:
```bash
nnUNetv2_train 001 3d_fullres 0
```

2. **Copy plans** to target dataset:
```bash
nnUNetv2_move_plans_between_datasets -s 001 -t 002 -c 3d_fullres
```

3. **Fine-tune** on target dataset:
```bash
nnUNetv2_train 002 3d_fullres 0 -pretrained_weights $nnUNet_results/Dataset001_Source/.../fold_0/checkpoint_final.pth
```

See `documentation/reference/pretraining_and_finetuning.md` for details.

### Manual Data Splits

By default, nnU-Net creates random 5-fold cross-validation splits. To specify custom splits:

1. Create `splits_final.json` in your dataset folder:
```json
[
  {
    "train": ["case_001", "case_002", "case_003"],
    "val": ["case_004"]
  },
  {
    "train": ["case_001", "case_002", "case_004"],
    "val": ["case_003"]
  }
]
```

2. Place it in:
```
$nnUNet_preprocessed/DatasetXXX_Name/splits_final.json
```

3. Train normally. nnU-Net will use your splits.

See `documentation/reference/manual_data_splits.md` for details.

## Advanced Inference Options

### Test-Time Augmentation

Enable mirroring for inference:

```bash
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -f 0 --mirror
```

This predicts on original + mirrored versions, then averages. Typically improves performance by 0.5-1% Dice at the cost of longer inference time.

### Multi-Configuration Ensemble

Combine predictions from different configurations:

```bash
# Train both 2d and 3d_fullres
nnUNetv2_train 001 2d all --npz
nnUNetv2_train 001 3d_fullres all --npz

# Ensemble
nnUNetv2_ensemble -i $nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation \
                     $nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation \
                  -o OUTPUT_FOLDER
```

### Sliding Window Parameters

Control inference speed vs. quality:

```bash
# Faster inference (larger step size, less overlap)
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -f 0 --step_size 0.7

# Better quality (smaller step size, more overlap)
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -f 0 --step_size 0.5
```

Default step size is 0.5 (50% overlap).

## Extending nnU-Net

### Adding a Custom Network Architecture

1. **Implement your network** in `nnunetv2/architecture/custom/`:

```python
# nnunetv2/architecture/custom/my_net.py
from torch import nn

def build_my_custom_network(input_channels, num_classes, **kwargs):
    """
    Build your custom architecture.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes (including background)
        **kwargs: Additional parameters from plans
    
    Returns:
        nn.Module: Your network
    """
    return MyCustomUNet(input_channels, num_classes)
```

2. **Create a trainer config**:

```python
# nnunetv2/training/configs/my_net.py
from nnunetv2.training.configs.base import TrainerConfig
from nnunetv2.architecture.custom.my_net import build_my_custom_network

def my_net_config():
    return TrainerConfig(
        name="my_net",
        description="My custom architecture",
        network_builder=build_my_custom_network,
        num_epochs=1000,
    )
```

3. **Register the config**:

```python
# nnunetv2/training/configs/__init__.py
from .my_net import my_net_config

TRAINER_CONFIGS = {
    'my_net': my_net_config,
}
```

4. **Use it**:

```bash
nnUNetv2_train 001 3d_fullres 0 -tr my_net
```

### Adding a Custom Loss Function

1. **Implement your loss** in `nnunetv2/training/losses/implementations/`:

```python
# nnunetv2/training/losses/implementations/my_loss.py
import torch
from torch import nn

class MyCustomLoss(nn.Module):
    def forward(self, pred, target):
        """
        Compute loss.
        
        Args:
            pred: (B, C, ...) predicted logits
            target: (B, C, ...) one-hot encoded targets
        
        Returns:
            scalar loss
        """
        # Your loss implementation
        return loss
```

2. **Use in a trainer config**:

```python
from nnunetv2.training.configs.base import TrainerConfig
from nnunetv2.training.losses.implementations.my_loss import MyCustomLoss

def my_loss_config():
    return TrainerConfig(
        name="my_loss",
        loss_fn=MyCustomLoss(),
    )
```

### Adding a Custom Preprocessing Step

Subclass `DefaultPreprocessor` in `nnunetv2/preprocessing/`:

```python
from nnunetv2.preprocessing.default_preprocessor import DefaultPreprocessor

class MyCustomPreprocessor(DefaultPreprocessor):
    def run_case(self, data, properties, seg=None):
        # Your custom preprocessing
        data = my_custom_transform(data)
        
        # Call parent for standard preprocessing
        return super().run_case(data, properties, seg)
```

Update plans to use your preprocessor.

### Adding a Custom Planner

Subclass `ExperimentPlanner` in `nnunetv2/experiment_planning/planners/`:

```python
from nnunetv2.experiment_planning.planners.standard.default_planner import ExperimentPlanner

class MyCustomPlanner(ExperimentPlanner):
    def determine_patch_size(self, ...):
        # Your custom logic for patch size
        return custom_patch_size
```

Use it:
```bash
nnUNetv2_plan_experiment -d 001 -pl MyCustomPlanner
```

## Performance Optimization

### GPU Utilization

Check GPU usage during training:
```bash
watch -n 0.5 nvidia-smi
```

**If GPU usage is low**:
- Increase `nnUNet_n_proc_DA` (data augmentation workers)
- Increase batch size (if memory allows)
- Check CPU bottleneck

### Data Augmentation Workers

Optimal values depend on CPU/GPU ratio:

```bash
# RTX 3090
export nnUNet_n_proc_DA=12

# RTX 4090
export nnUNet_n_proc_DA=16

# A100
export nnUNet_n_proc_DA=28
```

**Rule of thumb**: ~10-12 workers per GPU for consumer cards, 16-32 for datacenter GPUs.

### torch.compile

nnU-Net uses `torch.compile` by default (PyTorch 2.0+) for faster training.

To disable (if causing issues):
```bash
export nnUNet_compile=false
```

### Distributed Training (Multi-GPU)

nnU-Net supports multi-GPU training via DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 001 3d_fullres 0 --ddp
```

**Note**: This trains a single model across multiple GPUs. To train multiple folds in parallel, use separate processes:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 3d_fullres 0 &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 001 3d_fullres 1 &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 001 3d_fullres 2 &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 001 3d_fullres 3 &
```

## Debugging

### Enable Verbose Logging

Set log level:
```bash
export nnUNet_verbose=true
```

### Inspect Plans

After planning, examine the generated plan:

```bash
cat $nnUNet_preprocessed/Dataset001_Name/nnUNetPlans.json | jq
```

Look for:
- Batch size
- Patch size
- Network topology
- Target spacing

### Visualize Network

If `hiddenlayer` is installed, nnU-Net generates network diagrams:

```bash
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
nnUNetv2_train 001 3d_fullres 0
```

Check `$nnUNet_results/.../fold_0/network_architecture.pdf`.

### Check Preprocessed Data

Inspect preprocessed files:
```python
import numpy as np
data = np.load('$nnUNet_preprocessed/Dataset001_Name/3d_fullres/case_001.npz')
print(data['data'].shape)  # (C, H, W, D)
```

### Monitor Training

Training logs are in:
```
$nnUNet_results/DatasetXXX_Name/nnUNetTrainer__nnUNetPlans__CONFIG/fold_X/training_log.txt
```

Key metrics:
- Training loss
- Validation Dice score
- Learning rate
- Time per epoch

## Environment Variables Reference

```bash
# Required
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# Optional
export nnUNet_n_proc_DA=12              # Data augmentation workers
export nnUNet_compile=true              # Use torch.compile
export nnUNet_use_amp=true              # Use mixed precision
export nnUNet_verbose=false             # Verbose logging
```

## Next Steps

- **[Getting Started](getting_started.md)**: Basic usage tutorial
- **[Core Concepts](core_concepts.md)**: Understanding nnU-Net internals
- **[Reference](reference/)**: Detailed API documentation
- **[CUSTOM_ARCH_INFO.md](../CUSTOM_ARCH_INFO.md)**: Custom architecture examples
