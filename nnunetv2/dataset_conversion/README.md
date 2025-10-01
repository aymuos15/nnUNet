# Dataset Conversion

This module contains utilities for converting datasets from other formats to nnU-Net format.

## Overview

Provides converters for:

- **Medical Segmentation Decathlon (MSD)** datasets
- **nnU-Net v1** datasets (for migration to v2)
- Custom dataset formats

## Usage

### Convert MSD Dataset

```bash
nnUNetv2_convert_MSD_dataset -i INPUT_FOLDER -o OUTPUT_FOLDER
```

### Convert nnU-Net v1 Dataset

```bash
nnUNetv2_convert_old_nnUNet_dataset -i OLD_DATASET_FOLDER -o NEW_DATASET_FOLDER
```

## Dataset Format

For target nnU-Net format, see:
- [Dataset Format Reference](../documentation/reference/dataset_format.md)

## Custom Converters

To convert from a custom format:

```python
import shutil
import json
from pathlib import Path

def convert_my_dataset(input_folder, output_folder, dataset_id, dataset_name):
    """
    Convert custom dataset to nnU-Net format.
    
    Args:
        input_folder: Source dataset folder
        output_folder: Target nnU-Net_raw folder
        dataset_id: Dataset ID (e.g., 001)
        dataset_name: Dataset name (e.g., 'MyDataset')
    """
    # Create output structure
    dataset_folder = Path(output_folder) / f"Dataset{dataset_id:03d}_{dataset_name}"
    (dataset_folder / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_folder / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dataset_folder / "imagesTs").mkdir(parents=True, exist_ok=True)
    
    # Copy and rename files
    # ... your conversion logic ...
    
    # Create dataset.json
    dataset_json = {
        "name": dataset_name,
        "description": "Description of dataset",
        "labels": {
            "background": 0,
            "class1": 1,
            "class2": 2
        },
        "channel_names": {
            "0": "CT",
        },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz"
    }
    
    with open(dataset_folder / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
```

## See Also

- [Dataset Format Reference](../documentation/reference/dataset_format.md) - Target format specification
- [Experiment Planning](../experiment_planning/) - Next step after conversion
