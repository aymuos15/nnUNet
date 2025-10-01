# Experiment Planning

This module analyzes your dataset and automatically generates training plans. It's the first stage of nnU-Net's self-configuration pipeline.

## Overview

Experiment planning consists of three steps:

1. **Dataset Fingerprinting** - Extract statistical properties of your training data
2. **Configuration Planning** - Design optimal preprocessing and architecture based on the fingerprint
3. **Plan Persistence** - Save plans to disk for training and preprocessing

## Directory Structure

```
experiment_planning/
├── cli/                    # Command-line interface entry points
├── config/                 # Configuration data structures
├── core/                   # Core fingerprinting logic
├── dataset/                # Dataset analysis utilities
├── planners/               # Experiment planner implementations
├── planning/               # Planning algorithms and heuristics
├── pretraining/            # Plan transfer utilities for fine-tuning
├── resampling/             # Target spacing determination
└── utils/                  # Helper functions
```

## Key Components

### Dataset Fingerprinting (`core/`)

Extracts a comprehensive "fingerprint" of your dataset:

- **Image properties**: Sizes, spacings, modalities, file formats
- **Intensity statistics**: Mean, std, percentiles per modality
- **Foreground analysis**: Class frequencies, region sizes
- **Metadata**: Number of cases, dimensions (2D/3D)

**Output**: `dataset_fingerprint.json` containing all properties

### Experiment Planners (`planners/`)

Generate training plans based on fingerprints. The main planner is `ExperimentPlanner`, with specialized variants:

- **`ExperimentPlanner`** - Default planner for standard U-Net
- **`ResEncUNetPlanner`** - Planner for Residual Encoder U-Net architectures
- **Custom planners** - Subclass `ExperimentPlanner` to implement custom planning logic

Each planner creates multiple configurations (2D, 3D full-res, 3D low-res + cascade) and determines:

- Target spacing and resampling strategy
- Network topology (stages, kernels, strides, features per stage)
- Patch size and batch size
- Normalization scheme
- Data augmentation parameters

**Output**: `nnUNetPlans.json` containing all configurations

### Configuration Planning (`planning/`)

Modular utilities used by planners:

- **Architecture planning**: Determine network topology from fingerprint
- **Patch size planning**: Calculate optimal patch sizes given GPU memory
- **Batch size planning**: Determine batch size from patch size and network
- **Resampling planning**: Choose target spacing based on dataset properties

These utilities are composed by `ExperimentPlanner` to create the full plan.

### Plan Transfer (`pretraining/`)

Utilities for adapting plans from one dataset to another:

- Transfer architecture configuration for fine-tuning
- Adjust class counts while preserving topology
- Enable pretraining workflows

## Usage

### CLI Commands

**Full pipeline (fingerprint + plan + preprocess)**:
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID
```

**Fingerprint only**:
```bash
nnUNetv2_extract_fingerprint -d DATASET_ID
```

**Plan only** (requires fingerprint):
```bash
nnUNetv2_plan_experiment -d DATASET_ID
```

**Custom planner**:
```bash
nnUNetv2_plan_experiment -d DATASET_ID -pl ResEncUNetPlanner
```

### Programmatic API

```python
from nnunetv2.experiment_planning.planners import ExperimentPlanner

# Create planner instance
planner = ExperimentPlanner(
    dataset_id=1,
    gpu_memory_target_in_gb=24,
    preprocessor_name='DefaultPreprocessor',
    plans_name='nnUNetPlans',
    overwrite_target_spacing=None,
    suppress_transpose=False
)

# Run planning pipeline
planner.plan_experiment()
```

## Output Files

### Dataset Fingerprint (`dataset_fingerprint.json`)

Located in `nnUNet_preprocessed/DatasetXXX_Name/`

Contains:
- `spacing`: List of spacings for each training case
- `shapes_after_crop`: Image sizes after cropping to foreground
- `foreground_intensity_properties_per_channel`: Intensity statistics
- `class_locations`: Per-class foreground voxel counts
- `n_cases`: Number of training cases
- `dim`: Dimensionality (2D or 3D)

### Plans File (`nnUNetPlans.json`)

Located in `nnUNet_preprocessed/DatasetXXX_Name/`

Contains multiple configurations (e.g., `2d`, `3d_fullres`, `3d_lowres`, `3d_cascade_fullres`). Each configuration specifies:

- `data_identifier`: Preprocessing identifier
- `preprocessor_name`: Preprocessor class to use
- `batch_size`: Training batch size
- `patch_size`: Input patch dimensions
- `median_image_size_in_voxels`: Reference size after resampling
- `spacing`: Target voxel spacing
- `normalization_schemes`: Per-channel normalization
- `architecture`: Network topology parameters
  - `n_stages`, `features_per_stage`, `kernel_sizes`, `strides`
  - `n_conv_per_stage`, `n_conv_per_stage_decoder`
  - Convolution/normalization/activation operations

## Planning Heuristics

nnU-Net uses rule-based heuristics adapted from the dataset fingerprint:

### 1. Target Spacing

- Use median spacing across training set
- Optionally reduce spacing for anisotropic data (large spacing differences)
- Configurable via `--overwrite_target_spacing`

### 2. Network Topology

- **Number of stages**: Determined by median image size and desired min feature map size (typically 4x4x4 for 3D)
- **Kernel sizes**: Default 3x3x3 for 3D, 3x3 for 2D
- **Strides**: 2 for each downsampling stage (except first stage: stride 1)
- **Features per stage**: Starts at 32, doubles each stage (32→64→128→256→512→...)

### 3. Patch Size

- Maximize patch size given GPU memory constraint
- Ensure patch size covers representative regions of target structures
- Must be divisible by product of all strides

### 4. Batch Size

- Maximize batch size given GPU memory and patch size
- Minimum batch size of 2 (for batch normalization stability)

### 5. Normalization

- **CT**: Clip to [0.5, 99.5] percentiles, z-score normalize
- **MRI / Other**: Per-image z-score normalization (zero mean, unit std)
- Configurable per-channel based on modality properties

## Extending Experiment Planning

### Custom Planner

Create a subclass of `ExperimentPlanner`:

```python
from nnunetv2.experiment_planning.planners import ExperimentPlanner

class MyCustomPlanner(ExperimentPlanner):
    def generate_data_identifier(self, configuration_name: str) -> str:
        """Override to change preprocessing identifier."""
        return f"MyPreprocessing_{configuration_name}"
    
    def plan_experiment(self):
        """Override to implement custom planning logic."""
        super().plan_experiment()
        
        # Modify plans after default planning
        for config in self.plans['configurations'].values():
            config['batch_size'] = 4  # Force batch size
            config['patch_size'] = [128, 128, 128]  # Force patch size
```

Register and use:
```bash
nnUNetv2_plan_experiment -d 001 -pl MyCustomPlanner
```

### Manual Plan Editing

You can manually edit `nnUNetPlans.json` after generation:

1. Run planning: `nnUNetv2_plan_experiment -d 001`
2. Edit `nnUNet_preprocessed/Dataset001_Name/nnUNetPlans.json`
3. Proceed with preprocessing/training using modified plans

**Common edits**:
- Adjust `batch_size` or `patch_size` for memory constraints
- Change `spacing` to override resampling target
- Modify architecture topology (`n_stages`, `features_per_stage`)
- Add/remove configurations

### Custom Target Spacing

Override target spacing without custom planner:

```bash
nnUNetv2_plan_experiment -d 001 --overwrite_target_spacing 1.0 1.0 1.0
```

## Configuration Types

### 2D

- Treats each 2D slice as independent sample
- Useful for datasets with thick slices or truly 2D data (microscopy)
- Fast training, lower memory usage

### 3D Full Resolution

- Trains on full-resolution 3D patches
- Best for isotropic or mildly anisotropic data
- Higher memory usage

### 3D Low Resolution → Cascade

- **Stage 1 (3d_lowres)**: Train on downsampled data to capture context
- **Stage 2 (3d_cascade_fullres)**: Train at full resolution, conditioned on low-res prediction

Useful for:
- Very large images that don't fit in memory at full resolution
- Datasets where context matters (e.g., organ segmentation)

## Tips

- **GPU memory constraints**: Use `--gpu_memory_target` to specify available memory
- **Anisotropic data**: nnU-Net automatically handles via resampling and axis transposition
- **Small datasets**: Default plans work well; manual tuning rarely needed
- **Debugging plans**: Check `dataset_fingerprint.json` and `nnUNetPlans.json` for sanity
- **Pretraining**: Use `nnUNetv2_move_plans_between_datasets` to transfer plans for fine-tuning

## See Also

- [Plans File Reference](../documentation/reference/plans_file.md) - Detailed explanation of plan structure
- [Extending nnU-Net](../documentation/reference/extending_nnunet.md) - Custom components guide
- [Pretraining & Fine-Tuning](../documentation/reference/pretraining_and_finetuning.md) - Transfer learning workflows
