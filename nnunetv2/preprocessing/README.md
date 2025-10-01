# Preprocessing

This module handles all data preprocessing operations, transforming raw medical images into a format suitable for training.

## Overview

Preprocessing converts heterogeneous medical images (varying spacings, intensities, orientations) into normalized, consistently-formatted data. nnU-Net preprocessing is:

- **Automatic**: Configured by experiment planning based on dataset fingerprint
- **On-demand**: Can preprocess entire dataset upfront or during training
- **Reversible**: Stores parameters to reverse transformations during inference

## Directory Structure

```
preprocessing/
├── cropping/              # Foreground cropping utilities
├── normalization/         # Intensity normalization schemes
├── resampling/            # Spatial resampling operations
└── default_preprocessor.py  # Main preprocessing orchestrator
```

## Preprocessing Pipeline

The `DefaultPreprocessor` applies these operations in sequence:

1. **Cropping** - Remove background, crop to foreground bounding box
2. **Resampling** - Resample to target spacing (from plans)
3. **Normalization** - Intensity normalization per channel
4. **Serialization** - Save as `.npz` or `.npy` files

### 1. Cropping (`cropping/`)

**Purpose**: Remove empty background regions to reduce memory and focus on anatomy.

**Method**:
- Compute foreground mask (non-zero voxels in labels for training data)
- Find tight bounding box around foreground
- Crop image and labels to this bounding box
- Store crop parameters for reversal during inference

**Files**:
- `cropping/cropper.py` - Foreground detection and cropping logic

### 2. Resampling (`resampling/`)

**Purpose**: Standardize voxel spacing across all images.

**Method**:
- Resample image to target spacing (specified in plans)
- Use 3rd-order spline interpolation for images
- Use nearest-neighbor interpolation for labels
- Handle anisotropic data by resampling each axis appropriately

**Target spacing** is determined during experiment planning:
- Typically the median spacing of the training set
- May be reduced for highly anisotropic data
- Can be manually overridden via `--overwrite_target_spacing`

**Files**:
- `resampling/resamplers.py` - Resampling operations using scipy or SimpleITK

### 3. Normalization (`normalization/`)

**Purpose**: Standardize intensity distributions across images.

nnU-Net uses different schemes based on modality:

#### CT Normalization
- Clip intensities to global percentiles (0.5 and 99.5)
- Apply z-score normalization: `(x - mean) / std`
- Clipping removes extreme outliers (metal artifacts, etc.)
- Uses statistics computed across entire training set

#### MRI / Other Normalization
- Per-image z-score normalization: `(x - image_mean) / image_std`
- Computed only on foreground voxels (non-zero in foreground mask)
- Handles varying intensity scales across scanners/sequences

**Automatic detection**: Normalization scheme is automatically chosen during planning based on intensity properties in the dataset fingerprint.

**Files**:
- `normalization/normalizers.py` - Normalization scheme implementations

### 4. Serialization

Preprocessed data is saved in two formats:

- **NPZ (compressed)**: `.npz` files with `data` and `seg` (if labels available)
  - Space-efficient, slower to load
  - Used for training by default
  
- **NPY (uncompressed)**: Separate `.npy` files for `data` and `seg`
  - Faster to load, more disk space
  - Can be generated via `--npz` flag during preprocessing

## Usage

### CLI

Preprocessing is typically run automatically via:
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID
```

To preprocess only (after planning):
```bash
nnUNetv2_preprocess -d DATASET_ID -c CONFIG_NAME
```

**Configurations to preprocess**:
- `2d` - 2D configuration
- `3d_fullres` - 3D full resolution
- `3d_lowres` - 3D low resolution (for cascade)
- `3d_cascade_fullres` - 3D cascade full resolution

**Multiple configurations**:
```bash
nnUNetv2_preprocess -d 001 -c 2d 3d_fullres
```

**Number of processes**:
```bash
nnUNetv2_preprocess -d 001 -c 3d_fullres -np 8
```

### Programmatic API

```python
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

# Load plans
plans_manager = PlansManager('nnUNet_preprocessed/Dataset001_Name/nnUNetPlans.json')

# Create preprocessor
preprocessor = DefaultPreprocessor(verbose=True)

# Preprocess a configuration
preprocessor.run(
    dataset_id=1,
    configuration_name='3d_fullres',
    plans_manager=plans_manager,
    num_processes=8
)
```

### Lazy Preprocessing

nnU-Net supports lazy preprocessing where data is preprocessed on-the-fly during training:

**Advantages**:
- No upfront preprocessing time
- No additional disk space for preprocessed data

**Disadvantages**:
- Slower first epoch (preprocessing on demand)
- Repeated preprocessing across folds

**Enable lazy preprocessing**:
Set `nnUNet_preprocessed` to same location as `nnUNet_raw`:
```bash
export nnUNet_preprocessed=$nnUNet_raw
```

Training will automatically detect missing preprocessed files and generate them on-demand.

## Custom Preprocessing

### Custom Preprocessor

Create a custom preprocessor by subclassing `DefaultPreprocessor`:

```python
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
import numpy as np

class MyPreprocessor(DefaultPreprocessor):
    def run_case(self, image_files, seg_file, plans_manager, configuration_manager, dataset_json):
        """Override to implement custom preprocessing for a single case."""
        
        # Load image
        data, properties = self.load_image(image_files)
        
        # Load segmentation if available
        seg = self.load_segmentation(seg_file) if seg_file is not None else None
        
        # Custom preprocessing step
        data = self.my_custom_operation(data)
        
        # Apply standard cropping/resampling/normalization
        data, seg = self.crop_to_foreground(data, seg, properties)
        data, seg = self.resample(data, seg, properties, configuration_manager)
        data = self.normalize(data, properties, configuration_manager)
        
        return data, seg, properties
    
    def my_custom_operation(self, data):
        """Custom preprocessing logic."""
        # Example: apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        return np.stack([gaussian_filter(d, sigma=0.5) for d in data])
```

Register and use:
```python
preprocessor_name = 'MyPreprocessor'

# Update plans file to reference your preprocessor
plans['configurations']['3d_fullres']['preprocessor_name'] = preprocessor_name

# Run preprocessing
nnUNetv2_preprocess -d 001 -c 3d_fullres
```

### Custom Normalization Scheme

Add a new normalization scheme:

```python
from nnunetv2.preprocessing.normalization.normalizers import ImageNormalizer

class MyNormalizer(ImageNormalizer):
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """Implement custom normalization logic."""
        # Example: min-max normalization
        min_val = image.min()
        max_val = image.max()
        normalized = (image - min_val) / (max_val - min_val + 1e-8)
        return normalized
```

Reference in plans:
```python
plans['configurations']['3d_fullres']['normalization_schemes'] = ['MyNormalizer']
```

## Output Structure

Preprocessed data is stored in `nnUNet_preprocessed/DatasetXXX_Name/`:

```
nnUNet_preprocessed/
  Dataset001_Name/
    nnUNetPlans.json                 # Plans file
    dataset_fingerprint.json         # Dataset fingerprint
    gt_segmentations/                # Original labels (for validation)
      case_001.nii.gz
      case_002.nii.gz
      ...
    nnUNetPlans_3d_fullres/          # Preprocessed data for config
      case_001.npz                   # Preprocessed image + seg
      case_002.npz
      ...
    nnUNetPlans_2d/                  # Preprocessed data for 2D config
      case_001.npz
      ...
```

Each `.npz` file contains:
- **`data`**: Preprocessed image array, shape `[C, X, Y, Z]` or `[C, X, Y]`
- **`seg`**: Preprocessed segmentation, shape `[1, X, Y, Z]` or `[1, X, Y]` (only for training data)

## Preprocessing Properties

During preprocessing, metadata is stored for each case to enable reversibility:

**Properties saved** (in `.pkl` files alongside `.npz`):
- Original shape and spacing
- Cropping bounding box
- Resampling transformation parameters
- Normalization statistics (if global)

**Usage during inference**:
- Predictions are resampled back to original spacing
- Cropped regions are restored to original image size
- Ensures predictions align with input images

## Memory Considerations

Preprocessing large 3D volumes can be memory-intensive:

**Tips**:
- Use multiple processes (`-np`) to parallelize across cases (not within case)
- Reduce `num_processes` if running out of RAM
- Preprocessed files are often larger than raw data (after cropping, before compression)

**Typical memory usage**:
- ~2-4 GB RAM per preprocessing worker
- With 8 processes: ~16-32 GB RAM total

## Tips

- **Always run preprocessing** before training for faster iterations
- **Verify preprocessing** by checking a few `.npz` files:
  ```python
  import numpy as np
  data = np.load('case_001.npz')
  print(data['data'].shape, data['seg'].shape)
  ```
- **Debugging**: Set `verbose=True` in preprocessor to see detailed logs
- **Disk space**: Preprocessed data typically requires similar or more space than raw data
- **Multi-GPU**: Preprocessing is CPU-bound, can run independently of GPU training

## See Also

- [Intensity Normalization Reference](../documentation/reference/normalization.md) - Detailed normalization docs
- [Dataset Format](../documentation/reference/dataset_format.md) - Raw data format requirements
- [Experiment Planning](experiment_planning/) - How target spacing and normalization are determined
