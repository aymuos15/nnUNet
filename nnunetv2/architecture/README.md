# Architecture

This module contains neural network architectures used by nnU-Net, including the dynamic U-Net builder and custom architectures.

## Overview

nnU-Net uses a flexible architecture system that adapts network topology to dataset properties. The core is a dynamic U-Net builder that constructs networks based on configuration parameters from experiment planning.

## Directory Structure

```
architecture/
├── custom/              # Custom architecture implementations
│   ├── kiunet.py       # KiU-Net (dual-branch architecture)
│   ├── uiunet.py       # UIU-Net (nested U-Net structure)
│   └── README.md       # Custom architecture documentation
├── builder.py           # Dynamic U-Net construction logic
├── config.py            # Architecture configuration dataclass
├── instantiation.py     # Network instantiation from plans
└── __init__.py
```

## Key Components

### Dynamic U-Net Builder (`builder.py`)

Constructs U-Net architectures dynamically based on configuration parameters:

**Key function**: `build_network_architecture()`

**Parameters**:
- `n_stages` - Number of encoder/decoder stages
- `features_per_stage` - Feature channels at each stage
- `conv_op` - Convolution operation (Conv2d, Conv3d)
- `kernel_sizes` - Kernel sizes per stage
- `strides` - Downsampling strides per stage
- `n_conv_per_stage` - Convolutions per encoder stage
- `n_conv_per_stage_decoder` - Convolutions per decoder stage
- `norm_op` - Normalization operation (InstanceNorm, BatchNorm)
- `dropout_op` - Dropout operation (optional)
- `nonlin` - Activation function (LeakyReLU, ReLU)
- `deep_supervision` - Enable multi-scale auxiliary losses

**Builds**:
- Encoder with specified stages, strides, features
- Decoder with skip connections
- Segmentation head(s) for output

**Returns**: `torch.nn.Module` (U-Net instance)

### Architecture Config (`config.py`)

`ArchitectureConfig` dataclass defines network topology:

```python
@dataclass
class ArchitectureConfig:
    n_stages: int
    features_per_stage: List[int]
    conv_op: Type[nn.Module]
    kernel_sizes: List[List[int]]
    strides: List[List[int]]
    n_conv_per_stage: List[int]
    n_conv_per_stage_decoder: List[int]
    conv_bias: bool
    norm_op: Type[nn.Module]
    norm_op_kwargs: dict
    dropout_op: Optional[Type[nn.Module]]
    dropout_op_kwargs: Optional[dict]
    nonlin: Type[nn.Module]
    nonlin_kwargs: dict
```

Stored in `nnUNetPlans.json` under `architecture` key.

### Network Instantiation (`instantiation.py`)

**Purpose**: Load architecture from plans file or string specification.

**Key function**: `get_network_from_plans()`

```python
from nnunetv2.architecture.instantiation import get_network_from_plans

network = get_network_from_plans(
    plans_manager=plans_manager,
    configuration='3d_fullres',
    num_input_channels=4,
    num_output_channels=3,
    deep_supervision=True
)
```

Handles:
- Loading architecture class from string (e.g., `'torch.nn.Module'`)
- Instantiating with correct parameters
- Supporting both built-in and custom architectures

## Standard U-Net Architecture

nnU-Net's default architecture is a dynamic U-Net with:

### Encoder

Each encoder stage consists of:
1. Convolution block (n_conv_per_stage convolutions)
2. Downsampling (via strided convolution or pooling)

**Convolution block**:
```
Conv → Norm → Activation → Conv → Norm → Activation → ...
```

**Example** (4-stage encoder):
```
Input (4 channels)
  ↓ Conv Block (32 features, stride 1)
Stage 1 (32 features)
  ↓ Conv Block (64 features, stride 2)  [Downsample]
Stage 2 (64 features)
  ↓ Conv Block (128 features, stride 2) [Downsample]
Stage 3 (128 features)
  ↓ Conv Block (256 features, stride 2) [Downsample]
Stage 4 (256 features)
```

### Decoder

Each decoder stage consists of:
1. Upsampling (transpose convolution)
2. Concatenate with encoder skip connection
3. Convolution block (n_conv_per_stage_decoder convolutions)

**Example** (4-stage decoder):
```
Stage 4 (256 features)
  ↓ Upsample + Concat with Stage 3
Stage 3' (128 features)
  ↓ Upsample + Concat with Stage 2
Stage 2' (64 features)
  ↓ Upsample + Concat with Stage 1
Stage 1' (32 features)
  ↓ Segmentation Head (1x1 conv to num_classes)
Output (num_classes channels)
```

### Deep Supervision

When enabled, adds auxiliary segmentation heads at each decoder stage:

```
Stage 4 → Head 4 → Auxiliary Output 4 (lowest resolution)
Stage 3' → Head 3 → Auxiliary Output 3
Stage 2' → Head 2 → Auxiliary Output 2
Stage 1' → Head 1 → Main Output 1 (full resolution)
```

Outputs: `[main_output, aux_output_1, aux_output_2, aux_output_3]`

Loss is computed at each scale and weighted:
```python
loss = w1*loss1 + w2*loss2 + w3*loss3 + w4*loss4
# Weights: w_i = 0.5^i (renormalized)
```

## Custom Architectures

### Available Custom Architectures

See `custom/README.md` for detailed documentation:

- **DynamicKiUNet** - Dual-branch architecture with U-Net and Ki-Net branches
- **DynamicUIUNet3D** - Nested U-Net with RSU blocks

### Integrating Custom Architectures

**Option 1: Direct Reference in Plans**

Edit `nnUNetPlans.json`:
```json
{
  "configurations": {
    "3d_fullres": {
      "architecture": {
        "class_name": "nnunetv2.architecture.custom.kiunet.DynamicKiUNet",
        "kwargs": {
          "n_stages": 5,
          "features_per_stage": [32, 64, 128, 256, 512],
          ...
        }
      }
    }
  }
}
```

**Option 2: TrainerConfig**

Register via `TrainerConfig`:
```python
from nnunetv2.training.configs import TrainerConfig, TRAINER_CONFIGS
from nnunetv2.architecture.custom import DynamicKiUNet

config = TrainerConfig(
    architecture=DynamicKiUNet,
    architecture_kwargs={'pool_type': 'max'},
    loss=DC_and_CE_loss,
    loss_kwargs={}
)

TRAINER_CONFIGS['kiunet'] = config
```

Use:
```bash
nnUNetv2_train 001 3d_fullres 0 -tr kiunet
```

**Option 3: Custom Trainer**

Override `build_network_architecture()` in trainer:
```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.architecture.custom import DynamicKiUNet

class nnUNetTrainerKiUNet(nnUNetTrainer):
    def build_network_architecture(self, *args, **kwargs):
        return DynamicKiUNet(
            input_channels=kwargs['num_input_channels'],
            num_classes=kwargs['num_output_channels'],
            deep_supervision=kwargs.get('enable_deep_supervision', True),
            **kwargs['arch_init_kwargs']
        )
```

## Creating New Architectures

### Requirements

Custom architectures must:

1. **Inherit from `torch.nn.Module`**

2. **Accept standard parameters**:
   - `input_channels` - Number of input channels
   - `num_classes` - Number of output classes
   - `n_stages` - Number of encoder/decoder stages
   - `features_per_stage` - List of feature counts per stage
   - `conv_op` - Convolution operation (Conv2d/Conv3d)
   - `kernel_sizes` - Kernel sizes per stage
   - `strides` - Strides per stage
   - `n_conv_per_stage` - Convolutions per encoder stage
   - `n_conv_per_stage_decoder` - Convolutions per decoder stage
   - `conv_bias` - Use bias in convolutions
   - `norm_op` - Normalization operation
   - `norm_op_kwargs` - Normalization kwargs
   - `dropout_op` - Dropout operation (optional)
   - `dropout_op_kwargs` - Dropout kwargs (optional)
   - `nonlin` - Activation function
   - `nonlin_kwargs` - Activation kwargs
   - `deep_supervision` - Enable deep supervision

3. **Return appropriate output**:
   - If `deep_supervision=False`: Single tensor `[B, num_classes, ...]`
   - If `deep_supervision=True`: List of tensors at different scales

### Example Template

```python
import torch
import torch.nn as nn
from typing import List, Type, Optional

class MyCustomArchitecture(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: Type[nn.Module],
        kernel_sizes: List[List[int]],
        strides: List[List[int]],
        n_conv_per_stage: List[int],
        n_conv_per_stage_decoder: List[int],
        conv_bias: bool,
        norm_op: Type[nn.Module],
        norm_op_kwargs: dict,
        dropout_op: Optional[Type[nn.Module]],
        dropout_op_kwargs: Optional[dict],
        nonlin: Type[nn.Module],
        nonlin_kwargs: dict,
        deep_supervision: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        # Build your architecture here
        self.encoder = self.build_encoder(...)
        self.decoder = self.build_decoder(...)
        self.seg_heads = self.build_segmentation_heads(...)
    
    def forward(self, x):
        # Encoder forward
        encoder_features = self.encoder(x)
        
        # Decoder forward with skip connections
        decoder_outputs = self.decoder(encoder_features)
        
        # Segmentation heads
        outputs = [head(feat) for head, feat in zip(self.seg_heads, decoder_outputs)]
        
        if self.deep_supervision:
            return outputs  # List of multi-scale predictions
        else:
            return outputs[0]  # Only full-resolution prediction
```

### 2D/3D Compatibility

Use `conv_op` to support both:

```python
# conv_op will be either nn.Conv2d or nn.Conv3d
self.conv = conv_op(
    in_channels=input_channels,
    out_channels=features,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding
)

# Use get_matching_pool_op for pooling
from nnunetv2.architecture.builder import get_matching_pool_op

pool_op = get_matching_pool_op(conv_op)  # MaxPool2d or MaxPool3d
```

## Architecture Tips

### Memory Optimization

**Reduce memory**:
1. Decrease `features_per_stage` (fewer channels)
2. Reduce `n_stages` (shallower network)
3. Use smaller `patch_size` in plans

**Example**: 50% feature reduction
```python
# Original: [32, 64, 128, 256, 512]
# Reduced:  [16, 32, 64, 128, 256]
features_per_stage = [f // 2 for f in features_per_stage]
```

### Debugging Architectures

**Check parameter count**:
```python
num_params = sum(p.numel() for p in network.parameters())
print(f"Network parameters: {num_params / 1e6:.2f}M")
```

**Check output shapes**:
```python
dummy_input = torch.randn(2, 4, 128, 128, 128)  # [B, C, X, Y, Z]
output = network(dummy_input)

if isinstance(output, list):
    for i, out in enumerate(output):
        print(f"Output {i} shape: {out.shape}")
else:
    print(f"Output shape: {output.shape}")
```

**Verify deep supervision**:
```python
assert len(output) == n_stages, "Deep supervision should output one prediction per stage"
```

## See Also

- [Custom Architecture Guide](custom/README.md) - Detailed KiU-Net and UIU-Net documentation
- [CUSTOM_ARCH_INFO.md](../../CUSTOM_ARCH_INFO.md) - Integration examples
- [Training Module](../training/) - How architectures are used in training
- [Plans File Reference](../documentation/reference/plans_file.md) - Architecture configuration in plans
