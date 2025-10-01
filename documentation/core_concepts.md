# Core Concepts

This guide explains how nnU-Net works under the hood: its design philosophy, configuration strategy, and pipeline architecture.

## Design Philosophy

### The Problem

Medical image segmentation datasets are extremely diverse:
- **Dimensionality**: 2D (microscopy, X-rays) vs 3D (CT, MRI)
- **Modalities**: RGB, grayscale, multi-channel (T1/T2 MRI, multi-phase CT)
- **Resolutions**: Voxel spacings from 0.1mm to 10mm
- **Image sizes**: From 40×40 pixels to 1500×1500×1500 voxels
- **Class distributions**: Balanced to extremely imbalanced
- **Target structures**: Small lesions to large organs

Traditionally, each dataset requires manual pipeline design: choosing architecture, preprocessing, augmentation, training hyperparameters. This is:
- **Time-consuming**: Days to weeks of trial-and-error
- **Error-prone**: Easy to make suboptimal choices
- **Expertise-dependent**: Success varies with experimenter skill
- **Not scalable**: Can't easily apply to new problems

### The Solution

nnU-Net automates this entire process through **self-configuration**:
1. **Analyze** the dataset to extract key properties
2. **Configure** preprocessing, architecture, and training based on these properties
3. **Validate** multiple configurations to find the best one

The result: state-of-the-art performance with zero manual tuning.

## Three-Step Configuration Recipe

nnU-Net configures itself using three strategies:

### 1. Fixed Parameters

These are hard-coded defaults that work well universally. During development, these were extensively tested and found to be robust across diverse datasets.

**Examples**:
- **Loss function**: Dice + Cross-Entropy combination
- **Optimizer**: SGD with momentum (0.99) and Nesterov acceleration
- **Base learning rate**: 0.01
- **Data augmentation**: Extensive set including rotation, scaling, elastic deformation, brightness, contrast, noise
- **Deep supervision**: Enabled (auxiliary losses at multiple resolutions)

**Why fixed?**: These choices are robust and rarely need dataset-specific adjustment. Keeping them fixed reduces complexity.

### 2. Rule-Based Parameters

These adapt to dataset properties through hard-coded heuristics. The heuristics were derived from extensive empirical analysis.

**Examples**:

- **Network topology** (depth, pooling operations):
  - Determined by target patch size and image spacing
  - Ensures the receptive field matches the scale of anatomical structures

- **Patch size**:
  - As large as possible given GPU memory constraints
  - Balances context (larger patches) with batch size (smaller patches)

- **Batch size**:
  - Maximized to fit in available GPU memory
  - Adjusted based on patch size and network architecture

- **Target spacing** (resampling):
  - Typically median spacing across the dataset
  - Ensures isotropic or near-isotropic voxels for 3D convolutions

- **Normalization scheme**:
  - CT: Clip to foreground percentiles, then z-score
  - MRI: Z-score normalization per image
  - Chosen based on modality metadata

**Why rule-based?**: These properties strongly depend on dataset characteristics but can be determined through systematic rules rather than search.

### 3. Empirical Parameters

These require trial-and-error because they can't be reliably predicted from the dataset fingerprint.

**Examples**:

- **Best configuration selection**:
  - Train 2D, 3D full-resolution, and (if needed) 3D cascade
  - Compare performance on validation set
  - Select the best configuration

- **Postprocessing**:
  - Test whether removing small connected components improves performance
  - Apply only if beneficial

**Why empirical?**: No reliable heuristic exists for these choices. Cross-validation provides ground truth performance data.

## Dataset Fingerprint

The dataset fingerprint is a structured summary of dataset properties extracted during `nnUNetv2_plan_and_preprocess`.

### What's in the Fingerprint?

- **Image geometry**:
  - Shapes (min, max, median, mean)
  - Spacings (min, max, median, mean)
  - Dimensionality (2D, 3D, anisotropy)

- **Intensity statistics** (per channel):
  - Mean, std, min, max, percentiles
  - Distribution characteristics

- **Class information**:
  - Number of classes
  - Class frequencies (voxel counts)
  - Class imbalance ratios
  - Regions (connected vs. continuous structures)

- **Modality metadata**:
  - Channel names (e.g., "T1", "T2", "CT")
  - Inferred modality types

This fingerprint drives the automatic configuration.

## Configurations

nnU-Net can create multiple configurations for a dataset. Not all are always generated.

### 2D U-Net (`2d`)

**When**: Always created for any dataset

**How it works**:
- Treats 3D volumes as stacks of 2D slices
- Trains a 2D U-Net on individual slices
- Useful for:
  - Truly 2D data (microscopy, X-rays)
  - Very anisotropic 3D data (thick slices)
  - Datasets with limited training data

**Tradeoffs**:
- Ignores inter-slice context
- Faster to train than 3D
- Often competitive with 3D on anisotropic data

### 3D Full-Resolution U-Net (`3d_fullres`)

**When**: Created for 3D datasets

**How it works**:
- Operates on 3D patches at full resolution
- Network topology scales with patch size
- Typically the best choice for isotropic 3D data

**Tradeoffs**:
- Full context in all dimensions
- Higher memory requirement than 2D
- Slower training

### 3D Low-Resolution U-Net (`3d_lowres`)

**When**: Created only for large 3D datasets where full-resolution patches would be too small

**How it works**:
- Downsamples images to lower resolution
- Allows larger patches (more context)
- First stage of the cascade

**Tradeoffs**:
- Loses fine detail
- Faster training
- Only used as cascade input

### 3D Cascade Full-Resolution U-Net (`3d_cascade_fullres`)

**When**: Created for large 3D datasets (same condition as `3d_lowres`)

**How it works**:
- Takes low-resolution predictions as additional input
- Refines predictions at full resolution
- Two-stage pipeline: `3d_lowres` → `3d_cascade_fullres`

**Tradeoffs**:
- Best of both worlds: context + detail
- Requires training two models
- Slower inference

**When to use**: Automatically determined during planning. If median image size is very large (e.g., >512³), the cascade is created. Otherwise, only `3d_fullres` is used.

## Pipeline Stages

### 1. Experiment Planning (`nnUNetv2_plan_and_preprocess`)

**Steps**:
1. **Fingerprint extraction**:
   - Load all training images
   - Compute statistics (shapes, spacings, intensities)
   - Analyze class distributions
   - Save to `dataset_fingerprint.json`

2. **Plan generation**:
   - Determine which configurations to create (2d, 3d_fullres, cascade)
   - For each configuration:
     - Calculate target spacing
     - Determine patch size (based on GPU memory target)
     - Design network topology
     - Set batch size
   - Save to `nnUNetPlans.json`

3. **Preprocessing**:
   - For each configuration:
     - Resample images to target spacing
     - Apply normalization
     - Crop to nonzero regions
     - Save preprocessed data

**Output**: `$nnUNet_preprocessed/DatasetXXX_Name/`

### 2. Training (`nnUNetv2_train`)

**Steps**:
1. **Setup**:
   - Load plans and configuration
   - Build network architecture
   - Initialize optimizer and learning rate scheduler
   - Create data loaders with augmentation

2. **Training loop** (default 1000 epochs):
   - Sample random patches from preprocessed images
   - Apply on-the-fly data augmentation
   - Forward pass through network
   - Compute loss (Dice + CE with deep supervision)
   - Backward pass and optimizer step
   - Update learning rate (polynomial decay)

3. **Validation** (every 50 epochs):
   - Sliding window inference on validation cases
   - Compute metrics (Dice, etc.)
   - Save checkpoint if performance improves

4. **Checkpointing**:
   - Save every 50 epochs
   - Keep best model (`checkpoint_best.pth`)
   - Save final model (`checkpoint_final.pth`)

**Output**: `$nnUNet_results/DatasetXXX_Name/nnUNetTrainer__nnUNetPlans__CONFIG/fold_X/`

### 3. Inference (`nnUNetv2_predict`)

**Steps**:
1. **Load model(s)**:
   - Load checkpoint(s) from specified fold(s)
   - Initialize network architecture
   - Load weights

2. **Preprocessing**:
   - Apply same preprocessing as during training
   - Resample, normalize, crop

3. **Prediction**:
   - Sliding window inference (overlapping patches)
   - Aggregate overlapping predictions (Gaussian weighting)
   - Optional: Test-time augmentation (mirroring)
   - Optional: Ensembling (average multiple models)

4. **Postprocessing**:
   - Apply postprocessing if determined beneficial
   - Resample to original spacing
   - Save predictions

**Output**: Segmentation masks in specified output folder

## Data Augmentation

nnU-Net uses extensive on-the-fly augmentation during training:

**Spatial transforms**:
- Random rotation (±30° in-plane, ±15° out-of-plane for 3D)
- Random scaling (0.7 - 1.4×)
- Random elastic deformation
- Random mirroring

**Intensity transforms**:
- Brightness adjustment
- Contrast adjustment (gamma correction)
- Gaussian noise
- Gaussian blur
- Simulate low resolution

**Why so much?**: Medical imaging datasets are typically small (10s to 100s of cases). Heavy augmentation prevents overfitting and improves generalization.

## Loss Function

nnU-Net uses a combination of **Dice loss** and **Cross-Entropy (CE) loss**:

```python
loss = dice_loss + ce_loss
```

**Dice loss**:
- Directly optimizes the Dice score (primary metric)
- Handles class imbalance naturally
- Smooth and differentiable

**Cross-Entropy loss**:
- Provides strong gradients
- Improves training stability
- Complements Dice loss

**Deep supervision**:
- Additional losses at intermediate decoder layers
- Weighted: higher weight for full-resolution output
- Helps gradient flow in deep networks

## Network Architecture

### Base U-Net Structure

nnU-Net uses a dynamic U-Net architecture that adapts to the dataset:

```
Input → Encoder → Bottleneck → Decoder → Output
         ↓          ↓            ↑
         Skip connections --------
```

**Encoder**:
- Convolutional blocks with downsampling (strided conv or pooling)
- Depth determined by patch size
- Feature maps: 32, 64, 128, 256, ... (doubled at each stage)

**Bottleneck**:
- Deepest layer with highest feature count

**Decoder**:
- Upsampling (transposed conv) + convolutional blocks
- Skip connections from encoder
- Deep supervision outputs at multiple scales

**Convolutional block**:
- Conv → InstanceNorm → LeakyReLU
- Repeated twice per stage
- Kernel size: 3×3 (2D) or 3×3×3 (3D)

### Topology Adaptation

The network depth and pooling operations are automatically determined:

- **Patch size** → determines maximum depth
- **Anisotropy** → determines pooling strategy
  - Isotropic data: Pool equally in all dimensions
  - Anisotropic data: Pool less in the thin dimension

Example: For a patch size of 128×128×128, depth = 5 (128 → 64 → 32 → 16 → 8 → 4).

## When nnU-Net Works Well

**Excellent performance**:
- Medical imaging (CT, MRI, microscopy, ultrasound)
- 3D volumetric data
- Non-standard modalities or channel combinations
- Small to medium-sized datasets (10-1000 cases)
- Training from scratch required

**Why?**: nnU-Net was specifically designed for these scenarios.

## When nnU-Net May Not Be Optimal

**Foundation models may be better**:
- Natural images (RGB photos)
- Large benchmark datasets (Cityscapes, ADE20k)
- When large-scale pretraining data exists (ImageNet, etc.)

**Why?**: Foundation models leverage transfer learning from massive datasets. nnU-Net doesn't use pretraining, so it can't compete in these domains.

**nnU-Net's strength**: Non-standard problems where pretraining isn't applicable.

## Key Takeaways

1. **Self-configuration**: nnU-Net adapts to your data automatically
2. **Three-part strategy**: Fixed defaults + rule-based heuristics + empirical validation
3. **Dataset fingerprint**: Properties drive configuration decisions
4. **Multiple configurations**: Try 2D, 3D, cascade; pick the best
5. **Heavy augmentation**: Essential for small medical imaging datasets
6. **Modular design**: Easy to extend and customize

## Next Steps

- **[Getting Started](getting_started.md)**: Set up and run your first experiment
- **[Advanced Usage](advanced_usage.md)**: Customize configurations, add new architectures
- **[Reference](reference/)**: Detailed technical documentation
