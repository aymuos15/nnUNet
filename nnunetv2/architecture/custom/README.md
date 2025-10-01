# Custom Architectures for nnU-Net

This directory contains custom neural network architectures that extend nnU-Net's capabilities while maintaining compatibility with its planning and training infrastructure.

## Available Architectures

### DynamicKiUNet

A dynamic implementation of KiU-Net that combines two complementary pathways:

- **U-Net Branch**: Standard encoder-decoder with progressive downsampling
- **Ki-Net Branch**: Inverted encoder with progressive upsampling (overcomplete representation)
- **Cross-Refinement Blocks (CRFBs)**: Bidirectional feature exchange between branches at each stage

**Key Features:**
- Fully configurable number of stages and feature channels
- 2D and 3D support via parameterized convolution operations
- Deep supervision compatible
- Automatic integration with nnU-Net's planning system
- Faithful to original KiU-Net paper with configurable options

**Memory Requirements:**
- **Dual-branch architecture uses ~2x memory** compared to standard U-Net
- 24GB GPU: Use 50% features [16,32,64,128] (see `kiunet_minimal` config)
- 32GB+ GPU: Can use full features [32,64,128,256] (see `kiunet_large` config)

**Reference:** Based on [KiU-Net: Towards Accurate Segmentation of Biomedical Images using Over-complete Representations](https://github.com/jeya-maria-jose/KiU-Net-pytorch)

---

### DynamicUIUNet3D

A dynamic implementation of UIU-Net (U-Net in U-Net) with nested RSU blocks:

- **Nested U-Net Structure**: Each encoder/decoder stage contains an RSU block with its own internal U-Net
- **Multi-Scale Feature Extraction**: RSU blocks capture features at multiple scales within each stage
- **Side Outputs**: Multiple prediction heads at different decoder stages
- **Uncertainty-Inspired Fusion**: Combines all side outputs for final prediction

**Key Features:**
- Fully dynamic to adapt to nnU-Net's planning (stages, channels, kernel sizes, strides)
- Configurable RSU heights (depth of internal U-Nets)
- Multi-class segmentation support
- Deep supervision compatible
- Memory-optimized variants for constrained GPUs

**Memory Requirements:**
- **Nested U-Net structure is VERY memory-intensive** (U-Nets within U-Nets)
- Each RSU block contains an internal U-Net with its own encoder-decoder
- 24GB GPU: **MUST use `uiunet_minimal`** (50% features + reduced RSU heights)
- 32GB+ GPU: Can try `uiunet` with full features (may still OOM depending on patch size)

**Architecture Details:**
- **RSU (Residual U-block)**: Core building block containing a nested U-Net structure
  - Input → Conv → Internal U-Net (with downsampling) → Residual Connection
  - RSU height controls internal U-Net depth (e.g., height=7 means 7 downsampling steps)
  - Dilated RSU uses dilated convolutions instead of downsampling (for deeper stages)

- **Encoder**:
  - n_stages RSU blocks (e.g., 4-6 stages)
  - RSU height decreases per stage (e.g., 7→6→5→4→4→4 or 5→4→3→3→3→3 for minimal)
  - Last 2 stages use dilated RSU blocks

- **Decoder**:
  - Mirror encoder structure
  - Concatenates upsampled features with encoder skip connections
  - Each decoder stage has a side output head

- **Fusion**:
  - All side outputs upsampled to full resolution
  - Concatenated and fused via 1x1 convolution
  - Main output combines information from all scales

**Deep Supervision Adaptation:**
- Main output: Fused prediction from all side outputs (UIU-Net style)
- Auxiliary outputs: Intermediate decoder features at native resolutions (nnU-Net style)
- Format: `[fused_output, highest_res_aux, ..., lowest_res_aux]`

**Usage Example:**
```bash
# For 24GB GPUs (REQUIRED - nested structure is very memory-intensive)
export nnUNet_compile="false"
export CUDA_VISIBLE_DEVICES=1
nnUNetv2_train 004 3d_fullres 0 -tr uiunet_minimal

# For >32GB GPUs (may still OOM)
nnUNetv2_train 004 3d_fullres 0 -tr uiunet
```

**Configuration Variants:**
- `uiunet`: Full RSU heights (7,6,5,4,4,4), full features - requires >32GB GPU
- `uiunet_minimal`: Reduced RSU heights (5,4,3,3,3,3), 50% features - for 24GB GPU

**Reference:** Based on [UIU-Net: U-Net in U-Net for Infrared Small Object Detection](https://github.com/danfenghong/IEEE_TIP_UIU-Net)

---

## Architecture Fidelity

This implementation closely matches the original KiU-Net paper with several key design decisions:

### What Matches the Original Paper

1. **MaxPool Downsampling (#1)**: The original KiU-Net uses MaxPool for downsampling in the U-Net branch. Set `pool_type='max'` to match this behavior exactly (lines 210-226 in kiunet.py).

2. **Conv → Interpolate Order (#2)**: The Ki-Net encoder now correctly processes convolutions first, then upsamples (lines 436-452). This matches the original implementation.

3. **Skip Connection Timing (#3)**: Skip connections are stored BEFORE cross-refinement blocks (CRFB) at lines 456-457, matching the original architecture where pre-refinement features are used in the decoder.

### Differences from Original (for Memory Efficiency)

The following changes were made to prevent out-of-memory errors with large 3D medical images:

**⚠️ #4 - Upsampling Limit (Memory Optimization)**
- **What it does**: Ki-Net encoder upsampling is limited to 2x the input size (lines 438-452 in kiunet.py)
- **Why**: Prevents memory explosion with large 3D volumes
- **Trade-off**: Reduces the "overcomplete representation" principle that makes KiU-Net powerful
- **See "Memory Optimization & Extension" section below** for how to remove this limit if you have sufficient compute

**🔧 #5 - CRFB Scale Factors (Implementation Detail)**
- **Current**: Uses dynamic size-based matching with interpolation (lines 77-100 in kiunet.py)
- **Original**: Uses hardcoded exponential scale factors (2^stage) for each CRFB
- **Impact**: Minimal - both approaches achieve cross-scale feature exchange
- **See "Memory Optimization & Extension" section below** for how to use fixed scale factors

---

## ⚠️ Memory Optimization & Extension

This section explains memory-saving constraints and how to remove them if you have larger compute resources.

### 🔧 #4 - Removing the Upsampling Limit

**Current Behavior:**
The Ki-Net encoder upsampling is currently limited to 2x the input size to prevent OOM errors with typical GPU memory (16-24GB).

**Location:** `nnunetv2/architecture/custom/kiunet.py`, lines 438-452

**Current Code:**
```python
# Upsample after conv (with memory limit to prevent OOM)
# Limit to max 2x input size to avoid memory explosion
if stage_idx < self.n_stages - 1:  # Don't upsample after last stage
    kinet_current_size = kinet_features.shape[2:]
    max_allowed_size = tuple(s * 2 for s in input_spatial_size)
    # Only upsample if not exceeding limit
    can_upsample = all(curr < max_size for curr, max_size in zip(kinet_current_size, max_allowed_size))
    if can_upsample:
        kinet_size = tuple(min(s * 2, max_s) for s, max_s in zip(kinet_current_size, max_allowed_size))
        kinet_features = F.interpolate(
            kinet_features,
            size=kinet_size,
            mode=self.interpolation_mode,
            align_corners=False if self.interpolation_mode != 'nearest' else None
        )
```

**How to Remove (for users with >40GB GPU memory):**
Replace the above code block with unrestricted upsampling to match the original paper:
```python
# Unrestricted upsampling (original KiU-Net behavior)
if stage_idx < self.n_stages - 1:  # Don't upsample after last stage
    kinet_current_size = kinet_features.shape[2:]
    # Upsample by 2x at each stage (original overcomplete representation)
    kinet_size = tuple(s * 2 for s in kinet_current_size)
    kinet_features = F.interpolate(
        kinet_features,
        size=kinet_size,
        mode=self.interpolation_mode,
        align_corners=False if self.interpolation_mode != 'nearest' else None
    )
```

**Impact:** This will allow the Ki-Net branch to achieve full "overcomplete representation" as described in the paper, potentially improving segmentation quality for small objects. However, memory usage will increase significantly (2-4x higher).

---

### 🔧 #5 - Using Fixed CRFB Scale Factors

**Current Behavior:**
CRFBs dynamically match spatial dimensions using interpolation based on actual feature map sizes.

**Original Paper Behavior:**
Uses hardcoded exponential scale factors (2^stage) for each CRFB level.

**Location:** `nnunetv2/architecture/custom/kiunet.py`, CRFB class lines 68-116

**How to Modify:**
In the `DynamicKiUNet.__init__` method (lines 280-290), pass fixed scale factors to each CRFB:

```python
# Build Cross-Refinement Blocks (CRFBs) for encoder stages
self.crfb_encoder = nn.ModuleList()
self.crfb_scale_factors = []  # Add this
for stage_idx in range(n_stages):
    scale_factor = 2 ** stage_idx  # Original paper uses exponential scaling
    self.crfb_scale_factors.append(scale_factor)
    crfb = CRFB(
        unet_channels=features_per_stage[stage_idx],
        kinet_channels=kinet_features_per_stage[stage_idx],
        conv_op=conv_op,
        norm_op=norm_op,
        norm_op_kwargs=norm_op_kwargs,
        interpolation_mode=self.interpolation_mode,
        scale_factor=scale_factor,  # Pass fixed scale factor
    )
    self.crfb_encoder.append(crfb)
```

Then modify the CRFB class to use the fixed scale factor instead of dynamic size matching. This requires updating the `CRFB.__init__` and `CRFB.forward` methods.

**Impact:** Minimal - the current dynamic approach works well in practice. This is only needed for exact paper replication.

---

## Usage

### Option 1: Custom Trainer

Create a custom trainer that overrides the architecture building method:

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.architecture.custom import DynamicKiUNet

class nnUNetTrainerKiUNet(nnUNetTrainer):
    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        """Override to use DynamicKiUNet instead of default architecture."""

        # Import required modules for arch_init_kwargs
        import pydoc
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        # Build DynamicKiUNet with configuration from plans
        return DynamicKiUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **architecture_kwargs
        )
```

Then train using:
```bash
nnUNetv2_train DATASET_ID CONFIGURATION FOLD -tr nnUNetTrainerKiUNet
```

### Option 2: Direct Reference in Plans

Modify your plans file to reference the architecture directly:

```python
plans['configurations']['3d_fullres']['architecture'] = {
    'class_name': 'nnunetv2.architecture.custom.kiunet.DynamicKiUNet',
    'kwargs': {
        'n_stages': 6,
        'features_per_stage': [32, 64, 128, 256, 320, 320],
        'conv_op': 'torch.nn.modules.conv.Conv3d',
        'kernel_sizes': [[3, 3, 3]] * 6,
        'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        'n_conv_per_stage': [2, 2, 2, 2, 2, 2],
        'n_conv_per_stage_decoder': [2, 2, 2, 2, 2],
        'conv_bias': True,
        'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None,
        'dropout_op_kwargs': None,
        'nonlin': 'torch.nn.LeakyReLU',
        'nonlin_kwargs': {'inplace': True},
        'pool_type': 'max',  # Use MaxPool to match original KiU-Net paper
    },
    'kwargs_req_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']
}
```

### Option 3: Programmatic Usage

Use the architecture directly in your own code:

```python
import torch
from nnunetv2.architecture.custom import DynamicKiUNet

model = DynamicKiUNet(
    input_channels=4,
    num_classes=3,
    n_stages=5,
    features_per_stage=[32, 64, 128, 256, 512],
    conv_op=torch.nn.Conv3d,
    kernel_sizes=[[3, 3, 3]] * 5,
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    n_conv_per_stage=[2, 2, 2, 2, 2],
    n_conv_per_stage_decoder=[2, 2, 2, 2],
    conv_bias=True,
    norm_op=torch.nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs={'inplace': True},
    pool_type='max',  # Use MaxPool to match original KiU-Net paper
    deep_supervision=True
)

# Forward pass
input_tensor = torch.randn(2, 4, 128, 128, 128)
output = model(input_tensor)

if model.deep_supervision:
    print(f"Output is list of {len(output)} tensors at different scales")
else:
    print(f"Output shape: {output.shape}")
```

## Configuration Guidelines

### Matching Original Paper

To replicate the original KiU-Net paper as closely as possible, use these settings:

```python
model = DynamicKiUNet(
    input_channels=4,
    num_classes=3,
    n_stages=5,
    features_per_stage=[32, 64, 128, 256, 512],
    conv_op=torch.nn.Conv3d,
    kernel_sizes=[[3, 3, 3]] * 5,
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    n_conv_per_stage=[2, 2, 2, 2, 2],
    n_conv_per_stage_decoder=[2, 2, 2, 2],
    conv_bias=True,
    norm_op=torch.nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs={'inplace': True},
    pool_type='max',  # ⚠️ Use MaxPool like original (not strided conv)
    deep_supervision=True
)
```

**Key parameter:** `pool_type='max'` ensures the U-Net branch uses MaxPool downsampling like the original implementation, rather than strided convolutions.

**Note:** The upsampling limit (issue #4) is still applied by default. See "Memory Optimization & Extension" section to remove it if you have sufficient GPU memory (>40GB).

---

### Pooling Type Parameter

**New in this implementation:** The `pool_type` parameter controls how the U-Net encoder performs downsampling:

- **`pool_type='conv'`** (default): Uses strided convolutions for downsampling (nnU-Net standard)
- **`pool_type='max'`**: Uses MaxPool2d/3d for downsampling (original KiU-Net paper)
- **`pool_type='avg'`**: Uses AvgPool2d/3d for downsampling (alternative)

**To match the original paper, use `pool_type='max'`:**
```python
pool_type='max'  # Original KiU-Net behavior
```

**Implementation details:**
- When using `pool_type='max'` or `'avg'`, pooling layers are inserted before convolution blocks at each stage (except the first)
- Convolution stride is automatically set to 1 when explicit pooling is used
- See lines 210-226 in `nnunetv2/architecture/custom/kiunet.py`

---

### Choosing the Number of Stages

The number of stages determines the depth of the network:

- **Small datasets / 2D images**: 4-5 stages
- **Large datasets / 3D volumes**: 5-7 stages
- **Limited GPU memory**: Reduce stages or features_per_stage

### Feature Channels per Stage

Balance between model capacity and memory:

- **Light**: `[32, 64, 128, 256]`
- **Standard**: `[32, 64, 128, 256, 512]`
- **Heavy**: `[32, 64, 128, 256, 320, 320]`

### 2D vs 3D

The architecture automatically adapts to 2D or 3D based on `conv_op`:

**2D Configuration:**
```python
conv_op = torch.nn.Conv2d
kernel_sizes = [[3, 3]] * n_stages
strides = [[1, 1], [2, 2], [2, 2], ...]
```

**3D Configuration:**
```python
conv_op = torch.nn.Conv3d
kernel_sizes = [[3, 3, 3]] * n_stages
strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], ...]
```

### Deep Supervision

Enable deep supervision for better gradient flow and multi-scale learning:

```python
deep_supervision = True
```

This outputs a list: `[final_output, intermediate_1, intermediate_2, ...]`

## Architecture Details

### Dual-Branch Design

**U-Net Branch (Standard Path):**
```
Input -> Conv -> Pool -> Conv -> Pool -> ... -> Conv (Bottleneck)
                  ↓        ↓                      ↑
               Skip1    Skip2                  Upsample + Concat
```

**Ki-Net Branch (Overcomplete Path):**
```
Input -> Conv -> Upsample -> Conv -> Upsample -> ... -> Conv
                    ↓           ↓                         ↑
                 Skip1       Skip2                   Downsample + Concat
```

### Cross-Refinement Block (CRFB)

At each stage, features are exchanged between branches:

```
U-Net Features --[1x1 Conv + Interpolate]--> Ki-Net Features
                                                      |
Ki-Net Features --[1x1 Conv + Interpolate]--> U-Net Features
```

This allows:
- U-Net to benefit from global context (from Ki-Net's large receptive field)
- Ki-Net to benefit from local details (from U-Net's detailed features)

### Output Fusion

Final predictions combine both branches:

```
Output = (U-Net Prediction + Ki-Net Prediction) / 2
```

## Performance Considerations

### Memory Usage

KiU-Net uses ~1.5-2x more memory than standard U-Net due to dual branches:

- Reduce `features_per_stage` if GPU memory is limited
- Use mixed precision training (`--compile` flag in nnU-Net)
- Reduce batch size or patch size

### Training Time

Expect ~1.3-1.5x longer training time compared to standard U-Net:

- Two parallel encoders
- CRFBs add computation
- Deep supervision helps convergence

### When to Use KiU-Net

**Good for:**
- Small objects that need both local and global context
- Medical imaging with varying scale targets
- Datasets where standard U-Net struggles with scale variation

**May not help:**
- Very simple segmentation tasks
- Extremely limited GPU memory
- When standard U-Net already performs well

## Adding New Custom Architectures

To add a new architecture:

1. Create `new_architecture.py` in this directory
2. Implement your architecture inheriting from `nn.Module`
3. Ensure it accepts nnU-Net's standard parameters:
   - `input_channels`
   - `num_classes`
   - `n_stages`
   - `features_per_stage`
   - `conv_op`, `norm_op`, `dropout_op`, `nonlin`
   - `deep_supervision`
4. Return list of outputs if `deep_supervision=True`, else single output
5. Export in `__init__.py`
6. Document usage here

## References

- [KiU-Net Paper](https://arxiv.org/abs/2006.04878)
- [Original Implementation](https://github.com/jeya-maria-jose/KiU-Net-pytorch)
- [nnU-Net Documentation](https://github.com/MIC-DKFZ/nnUNet)

## Troubleshooting

### "Module not found" error

Ensure you're importing from the correct path:
```python
from nnunetv2.architecture.custom import DynamicKiUNet
```

### CUDA Out of Memory

Reduce model size:
```python
features_per_stage = [24, 48, 96, 192, 384]  # Smaller features
n_stages = 4  # Fewer stages
```

### Poor convergence

Try:
- Enable deep supervision
- Adjust learning rate
- Use standard U-Net initialization (already done in implementation)
- Increase training iterations

### Dimension mismatch errors

Ensure your `conv_op`, `kernel_sizes`, and `strides` are consistent:
- 2D: Use `Conv2d` with `[H, W]` specifications
- 3D: Use `Conv3d` with `[H, W, D]` specifications
