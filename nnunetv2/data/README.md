# Data

This module handles dataset abstraction, data loading, and augmentation for training.

## Overview

The data module provides:

- **Dataset classes** - Abstractions for accessing training/validation data
- **Data loaders** - PyTorch DataLoader integration with batching
- **Transforms** - Data augmentation pipeline using `batchgenerators`
- **Dataset I/O** - Loading preprocessed data from disk

## Directory Structure

```
data/
├── dataset_io/           # Dataset I/O utilities
├── transforms/           # Augmentation transforms
├── dataset.py            # Dataset class implementations
├── loader.py             # DataLoader implementations
├── transform_builders.py # Augmentation pipeline builders
└── utils.py              # Data utilities
```

## Key Components

### Dataset Classes (`dataset.py`)

**nnUNetDataset**:

Main dataset class for training and validation:

```python
class nnUNetDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        case_identifiers: List[str],
        num_images_properties_loading_threshold: int = 0
    ):
        """
        Args:
            data_folder: Path to preprocessed data
            case_identifiers: List of case names to include
            num_images_properties_loading_threshold: Load all properties if 
                dataset size < threshold (speeds up access)
        """
```

**Features**:
- Lazy loading: Data loaded on-demand, not upfront
- Property caching: Stores image properties (spacing, shapes) for all cases
- Memory efficient: Only loads requested cases

**Usage**:
```python
from nnunetv2.data.dataset import nnUNetDataset

dataset = nnUNetDataset(
    data_folder='nnUNet_preprocessed/Dataset001_Name/nnUNetPlans_3d_fullres',
    case_identifiers=['case_001', 'case_002', 'case_003']
)

# Access single case
case_data = dataset[0]  # Returns dict with 'data', 'seg', 'properties'
```

**Output format** (per case):
```python
{
    'data': np.ndarray,        # Image, shape [C, X, Y, Z]
    'seg': np.ndarray,         # Segmentation, shape [1, X, Y, Z]
    'properties': dict,        # Metadata (spacing, original_shape, etc.)
    'keys': List[str]          # Case identifier
}
```

### Data Loaders (`loader.py`)

**nnUNetDataLoader**:

Custom DataLoader with:
- Multi-threaded data loading
- Background preprocessing workers
- Queue-based batching
- On-the-fly augmentation

```python
class nnUNetDataLoader:
    def __init__(
        self,
        dataset: nnUNetDataset,
        batch_size: int,
        patch_size: List[int],
        num_threads_in_multithreaded: int = 1,
        transforms=None,
        ...
    ):
```

**Features**:
- **Multi-processing**: Loads data in background threads
- **Patch extraction**: Randomly extracts patches from full images
- **Augmentation**: Applies transforms on-the-fly
- **Infinite iteration**: Continuously yields batches (useful for training)

**Usage**:
```python
from nnunetv2.data.loader import nnUNetDataLoader
from nnunetv2.data.transform_builders import get_training_transforms

# Build augmentation pipeline
transforms = get_training_transforms(
    patch_size=[128, 128, 128],
    rotation_for_DA=(-30, 30),
    deep_supervision_scales=[[1, 1, 1], [0.5, 0.5, 0.5]],
    ...
)

# Create data loader
dataloader = nnUNetDataLoader(
    dataset=train_dataset,
    batch_size=2,
    patch_size=[128, 128, 128],
    num_threads_in_multithreaded=8,
    transforms=transforms
)

# Iterate
for batch in dataloader:
    data = batch['data']    # [B, C, X, Y, Z]
    target = batch['target']  # [B, num_classes, X, Y, Z]
    # ... training step
```

### Transforms & Augmentation (`transforms/`, `transform_builders.py`)

nnU-Net uses `batchgenerators` library for augmentation.

#### Transform Pipeline Builder (`transform_builders.py`)

**`get_training_transforms()`**:

Creates the full augmentation pipeline:

```python
from nnunetv2.data.transform_builders import get_training_transforms

transforms = get_training_transforms(
    patch_size=[128, 128, 128],
    rotation_for_DA=(-30, 30),            # Rotation range in degrees
    deep_supervision_scales=None,         # Multi-scale outputs for deep supervision
    mirror_axes=(0, 1, 2),                # Axes to mirror
    do_dummy_2d_data_aug=False,           # 2D augmentation on 3D data
    use_mask_for_norm=True,               # Normalize only foreground
    ...
)
```

**Augmentation stages**:

1. **Spatial transforms**:
   - Random rotation (`-30°` to `+30°` by default)
   - Random scaling (`0.7x` to `1.4x` by default)
   - Random elastic deformation
   
2. **Intensity transforms**:
   - Brightness multiplication (`0.75` to `1.25`)
   - Contrast adjustment
   - Gamma correction (`0.7` to `1.5`)
   - Gaussian noise
   - Gaussian blur
   
3. **Mirroring**:
   - Random mirroring along spatial axes

4. **Downsampling** (for deep supervision):
   - Create multi-scale versions of targets

**Validation transforms** (`get_validation_transforms()`):

Minimal preprocessing (no augmentation):
- Only handles deep supervision downsampling
- No spatial or intensity augmentation

#### Custom Transforms (`transforms/`)

**Available transforms**:
- `SpatialTransform` - Rotation, scaling, elastic deformation
- `GaussianNoiseTransform` - Add Gaussian noise
- `GaussianBlurTransform` - Gaussian smoothing
- `BrightnessMultiplicativeTransform` - Brightness adjustment
- `ContrastAugmentationTransform` - Contrast adjustment
- `GammaTransform` - Gamma correction
- `MirrorTransform` - Random mirroring
- `MaskTransform` - Foreground masking
- `RemoveLabelTransform` - Ignore specific labels
- `RenameTransform` - Rename dict keys
- `NumpyToTensor` - Convert numpy arrays to PyTorch tensors
- `DownsampleSegForDSTransform` - Downsample segmentation for deep supervision

### Dataset I/O (`dataset_io/`)

Utilities for loading/saving preprocessed data:

**Load case**:
```python
from nnunetv2.data.dataset_io import load_case_from_npz

data, seg, properties = load_case_from_npz('case_001.npz')
```

**Load dataset properties**:
```python
from nnunetv2.data.dataset_io import load_dataset_json

dataset_json = load_dataset_json('dataset.json')
```

## Augmentation Parameters

Augmentation is automatically configured during experiment planning, but can be customized.

### Default Parameters

**Spatial**:
- **Rotation**: `-30°` to `+30°` (uniformly sampled)
- **Scaling**: `0.7x` to `1.4x` (per axis, independent)
- **Elastic deformation**: Control point spacing = `patch_size // 8`, deformation scale = `25` voxels

**Intensity**:
- **Brightness**: `0.75x` to `1.25x`
- **Contrast**: Per-channel contrast adjustment
- **Gamma**: `0.7` to `1.5` (apply with 30% probability)
- **Gaussian noise**: Small magnitude, apply with 15% probability
- **Gaussian blur**: Apply with 20% probability

**Mirroring**:
- 50% probability per axis

### Customizing Augmentation

**Via custom trainer**:

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerCustomAug(nnUNetTrainer):
    def get_training_transforms(self):
        """Override to customize augmentation."""
        from nnunetv2.data.transform_builders import get_training_transforms
        
        return get_training_transforms(
            patch_size=self.configuration_manager.patch_size,
            rotation_for_DA=(-45, 45),  # More aggressive rotation
            ...
        )
```

**Via manual transform composition**:

```python
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform

transforms = Compose([
    SpatialTransform(
        patch_size=[128, 128, 128],
        do_elastic_deform=True,
        do_rotation=True,
        angle_x=(-30 / 180 * np.pi, 30 / 180 * np.pi),
        ...
    ),
    BrightnessMultiplicativeTransform(
        multiplier_range=(0.75, 1.25),
        per_channel=True
    ),
    ...
])
```

## Training Data Flow

### Typical Training Loop

```
1. nnUNetDataLoader samples random case from dataset
2. Loads case from disk (lazy loading)
3. Extracts random patch of size patch_size
4. Applies augmentation transforms
5. Converts to PyTorch tensors
6. Yields batch to training loop
```

### Deep Supervision

When `deep_supervision=True`, the data loader:

1. Creates multi-scale versions of target segmentation
2. Downsamples to match network auxiliary output resolutions
3. Returns `target` as list: `[full_res, half_res, quarter_res, ...]`

**Downsampling**:
- Uses 3rd-order spline interpolation
- Thresholds to binary after downsampling

## Memory Considerations

### Dataset Loading

**Lazy loading** (default):
- Data loaded on-demand when accessed
- Lower memory usage
- Slightly slower (disk I/O overhead)

**Preloading** (optional):
- Load all data into RAM upfront
- Faster training (no disk I/O during training)
- Higher memory usage

**Enable preloading**:
```python
dataset.load_all_data_into_memory()  # Method not exposed by default
```

### Data Loader Workers

**`num_threads_in_multithreaded`**:
- Number of background workers for data loading
- More workers → faster, but more RAM

**Typical values**:
- CPU training: 4-8 workers
- GPU training: 8-12 workers
- RAM-constrained: 2-4 workers

**Memory usage**: ~2-4 GB per worker (depends on patch size)

## Advanced Usage

### Custom Dataset

Subclass `nnUNetDataset` for custom data sources:

```python
from nnunetv2.data.dataset import nnUNetDataset

class MyCustomDataset(nnUNetDataset):
    def load_case(self, case_identifier):
        """Override to load from custom source."""
        # Load from database, cloud storage, etc.
        data = my_custom_load_function(case_identifier)
        
        return {
            'data': data['image'],
            'seg': data['label'],
            'properties': data['metadata'],
            'keys': [case_identifier]
        }
```

### On-The-Fly Preprocessing

For lazy preprocessing (no preprocessed files):

```python
class LazyPreprocessDataset(nnUNetDataset):
    def load_case(self, case_identifier):
        # Load raw image
        raw_image = load_raw_image(case_identifier)
        
        # Apply preprocessing on-the-fly
        preprocessed = self.preprocess(raw_image)
        
        return preprocessed
```

### Multi-Modal Data

nnU-Net natively supports multi-channel inputs:

**Dataset format**:
```
imagesTr/
  case_001_0000.nii.gz  # Modality 0 (e.g., T1)
  case_001_0001.nii.gz  # Modality 1 (e.g., T2)
  case_001_0002.nii.gz  # Modality 2 (e.g., FLAIR)
```

**Loaded as**:
```python
data.shape  # [3, X, Y, Z] - 3 channels
```

**Normalization**: Applied per-channel independently.

## Tips

### Training Speed

**Faster training**:
1. Increase `num_threads_in_multithreaded` (8-12)
2. Use NPZ format (compressed, already default)
3. Ensure data on fast storage (SSD, not HDD)
4. Reduce augmentation complexity (fewer transforms)

### Debugging Data Loading

**Inspect batch**:
```python
for batch in dataloader:
    print("Data shape:", batch['data'].shape)
    print("Target shape:", batch['target'].shape)
    print("Keys:", batch['keys'])
    break  # Only check first batch
```

**Visualize augmentation**:
```python
import matplotlib.pyplot as plt

batch = next(iter(dataloader))
data = batch['data'][0, 0].cpu().numpy()  # First sample, first channel

plt.imshow(data[:, :, data.shape[2]//2])  # Mid-slice
plt.show()
```

### Handling Class Imbalance

nnU-Net handles class imbalance via:
1. **Dice loss**: Naturally handles imbalance (small classes get high weight)
2. **Oversampling foreground**: 33% of patches are forced to contain foreground

**Custom oversampling**:
```python
class nnUNetTrainerCustomSampling(nnUNetTrainer):
    def get_dataloaders(self):
        # Override to change oversampling ratio
        # Default: oversample_foreground_percent = 0.33
        ...
```

## See Also

- [Preprocessing](../preprocessing/) - How data is preprocessed before loading
- [Training](../training/) - How data loaders are used in training
- [Dataset Format Reference](../documentation/reference/dataset_format.md) - Raw dataset requirements
