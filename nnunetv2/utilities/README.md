# Utilities

This module contains helper functions and utilities used across nnU-Net.

## Overview

Provides common utilities for:

- **Core utilities** - File operations, JSON handling, multiprocessing helpers
- **Visualization** - Plotting, image overlay generation

## Directory Structure

```
utilities/
├── core/             # Core utility functions
└── visualization/    # Visualization tools
```

## Key Components

### Core Utilities (`core/`)

**File operations**:
- `maybe_mkdir_p()` - Create directory if it doesn't exist
- `subdirs()` - List subdirectories
- `subfiles()` - List files matching pattern
- `load_json()` - Load JSON file
- `save_json()` - Save JSON file
- `load_pickle()` - Load pickle file
- `write_pickle()` - Save pickle file

**Multiprocessing**:
- `multiprocessing_pool()` - Context manager for process pools
- `map_with_progress()` - Parallel map with progress bar

**Plans handling**:
- `PlansManager` - Load and access experiment plans
- `ConfigurationManager` - Access configuration-specific settings

**Usage**:
```python
from nnunetv2.utilities.file_path_utilities import maybe_mkdir_p, subfiles
from nnunetv2.utilities.json_export import load_json, save_json

# Create directory
maybe_mkdir_p('/path/to/directory')

# List files
nifti_files = subfiles('/path/to/folder', suffix='.nii.gz', join=True)

# Load/save JSON
data = load_json('file.json')
save_json(data, 'output.json')
```

### Visualization (`visualization/`)

**Overlay generation**:

Create PNG overlays of predictions on images for visual inspection.

**CLI**:
```bash
nnUNetv2_plot_overlay_pngs -i INPUT_FOLDER -o OUTPUT_FOLDER -s SEG_FOLDER
```

**Example**:
```bash
nnUNetv2_plot_overlay_pngs \
    -i nnUNet_raw/Dataset001_Name/imagesTs \
    -o overlay_pngs \
    -s predictions/Dataset001_Name
```

**Output**: PNG images with segmentation overlaid on input images.

## Common Utilities

### Plans Manager

Access experiment plans:

```python
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

plans_manager = PlansManager('nnUNet_preprocessed/Dataset001_Name/nnUNetPlans.json')

# Access configuration
config = plans_manager.get_configuration('3d_fullres')
print(config.patch_size)
print(config.batch_size)
```

### Configuration Manager

Access configuration-specific settings:

```python
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

config_manager = ConfigurationManager(
    plans=plans,
    configuration_name='3d_fullres',
    dataset_json=dataset_json
)

# Access settings
print(config_manager.patch_size)
print(config_manager.spacing)
print(config_manager.normalization_schemes)
```

### Multiprocessing Helpers

Parallel processing with progress bar:

```python
from nnunetv2.utilities.helpers import multiprocessing_pool

def process_case(case_id):
    # Your processing logic
    return result

with multiprocessing_pool(num_processes=8) as pool:
    results = pool.map(process_case, case_ids)
```

## See Also

- All other modules use utilities extensively
- Visualization utilities used for debugging and result inspection
