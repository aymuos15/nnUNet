# Model Sharing

This module provides utilities for exporting, importing, and downloading pretrained nnU-Net models.

## Overview

Model sharing enables:

- **Model export** - Package trained models into shareable ZIP files
- **Model import** - Install models from ZIP files
- **Model download** - Download pretrained models from URLs
- **Portability** - Share models across systems and users

## Directory Structure

```
model_sharing/
├── entry_points.py     # CLI entry points
├── model_export.py     # Export trained models to ZIP
├── model_import.py     # Import models from ZIP
└── model_download.py   # Download models from URLs
```

## Usage

### Export Trained Model

Package a trained model into a ZIP file:

**CLI**:
```bash
nnUNetv2_export_model_to_zip -i INPUT_FOLDER -o OUTPUT_FILE.zip
```

**Example**:
```bash
nnUNetv2_export_model_to_zip \
    -i nnUNet_results/Dataset001_BrainTumor/nnUNetTrainer__nnUNetPlans__3d_fullres \
    -o BrainTumor_3dfullres.zip
```

**What gets packaged**:
- All fold checkpoints (`checkpoint_best.pth`, `checkpoint_final.pth`)
- Plans file (`nnUNetPlans.json`)
- Dataset fingerprint (`dataset_fingerprint.json`)
- Dataset JSON (`dataset.json`)
- Training metadata

**ZIP structure**:
```
BrainTumor_3dfullres.zip
├── fold_0/
│   ├── checkpoint_best.pth
│   └── checkpoint_final.pth
├── fold_1/
│   └── ...
├── nnUNetPlans.json
├── dataset_fingerprint.json
└── dataset.json
```

### Import Pretrained Model

Install a pretrained model from ZIP:

**CLI**:
```bash
nnUNetv2_install_pretrained_model_from_zip MODEL.zip
```

**Example**:
```bash
nnUNetv2_install_pretrained_model_from_zip BrainTumor_3dfullres.zip
```

**What it does**:
1. Extracts ZIP to `nnUNet_results/`
2. Places model in appropriate directory structure
3. Makes model available for inference

**After installation**:
```bash
# Use the model for prediction
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c CONFIG -f 0
```

### Download Pretrained Model

Download and install from URL:

**CLI**:
```bash
nnUNetv2_download_pretrained_model_by_url URL
```

**Example**:
```bash
nnUNetv2_download_pretrained_model_by_url \
    https://example.com/models/BrainTumor_3dfullres.zip
```

**What it does**:
1. Downloads ZIP from URL
2. Automatically installs using import function
3. Cleans up temporary files

## Pretrained Model Usage

After installing a pretrained model:

### Inference

```bash
nnUNetv2_predict_from_modelfolder \
    -i INPUT_FOLDER \
    -o OUTPUT_FOLDER \
    -m nnUNet_results/DatasetXXX_Name/TRAINER__PLANS__CONFIG \
    -f 0 1 2 3 4
```

### Fine-Tuning

Use pretrained weights to initialize training on a new dataset:

```bash
# 1. Transfer plans
nnUNetv2_move_plans_between_datasets \
    -s SOURCE_DATASET_ID \
    -t TARGET_DATASET_ID \
    -sp nnUNetPlans \
    -tp nnUNetPlans_pretrained

# 2. Train with pretrained weights
nnUNetv2_train TARGET_DATASET_ID 3d_fullres 0 \
    -p nnUNetPlans_pretrained \
    -pretrained_weights nnUNet_results/DatasetSOURCE/.../checkpoint_final.pth
```

## Programmatic API

### Export Model

```python
from nnunetv2.model_sharing.model_export import export_model_to_zip

export_model_to_zip(
    input_folder='nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres',
    output_file='model.zip'
)
```

### Import Model

```python
from nnunetv2.model_sharing.model_import import install_model_from_zip

install_model_from_zip(
    zip_file='model.zip',
    install_location='nnUNet_results'
)
```

### Download Model

```python
from nnunetv2.model_sharing.model_download import download_and_install_from_url

download_and_install_from_url(
    url='https://example.com/model.zip',
    destination='nnUNet_results'
)
```

## Model Sharing Best Practices

### What to Share

**Minimum**:
- Best checkpoint (`checkpoint_best.pth`)
- Plans file (`nnUNetPlans.json`)
- Dataset JSON (`dataset.json`)

**Recommended**:
- All fold checkpoints (for ensemble)
- Final checkpoints (for fine-tuning)
- Dataset fingerprint
- Training log (for reference)

**Optional**:
- Validation predictions
- Debug info

### Model Metadata

Include in `dataset.json`:

```json
{
  "name": "BrainTumor",
  "description": "Trained on BraTS 2021 dataset",
  "labels": {
    "background": 0,
    "edema": 1,
    "non-enhancing": 2,
    "enhancing": 3
  },
  "modality": {
    "0": "T1",
    "1": "T1ce",
    "2": "T2",
    "3": "FLAIR"
  },
  "tensorImageSize": "4D",
  "reference": "https://www.med.upenn.edu/cbica/brats2021/",
  "licence": "CC-BY-SA 4.0",
  "release": "1.0"
}
```

### Sharing Platforms

**Options**:
- **Zenodo** - Academic datasets, citable DOI
- **GitHub Releases** - Version control, release notes
- **Google Drive / Dropbox** - Simple sharing
- **Institutional repositories** - For internal sharing

**Recommended**: Zenodo for public models (persistent, citable).

## Security Considerations

### Importing Models

**Risks**:
- Models are PyTorch checkpoints (can contain arbitrary code)
- Only import models from trusted sources

**Best practices**:
1. Verify source (official repository, known researcher)
2. Check file integrity (SHA256 hash if provided)
3. Inspect ZIP contents before importing
4. Use in isolated environment for first run

### Exporting Models

**Avoid including**:
- Absolute paths (use relative paths)
- Sensitive data (patient info, internal paths)
- Large validation predictions (strip before export)

## Troubleshooting

### Import Fails

**Missing dependencies**:
```
Error: Missing required files in ZIP
```
- Ensure ZIP contains all required files (plans, checkpoints, etc.)

**Path conflicts**:
```
Error: Model already exists at destination
```
- Remove existing model or use different dataset ID

### Export Fails

**Large file size**:
- ZIP files can be large (multiple GB for 5 folds)
- Consider sharing only best checkpoints, not final

**Disk space**:
- Export requires temporary space (2x model size)

### Download Fails

**Network issues**:
- Large files may timeout
- Use tools like `wget` or `curl` for large downloads, then import manually

```bash
wget https://example.com/model.zip
nnUNetv2_install_pretrained_model_from_zip model.zip
```

## Tips

### Reducing Model Size

**Share fewer folds**:
- Export only fold 0 (for single-model inference)
- Or folds 0-2 (for faster ensemble)

**Compress checkpoints**:
- ZIP already provides compression
- Further compression (tar.gz) provides minimal benefit

**Strip unnecessary data**:
- Remove optimizer state (only needed for resuming training)
- Keep only `network_weights` from checkpoint

### Versioning Models

Include version in filename:
```
BrainTumor_3dfullres_v1.0.zip
BrainTumor_3dfullres_v1.1.zip
```

Track changes:
- Training dataset version
- nnU-Net version used
- Architecture modifications

## See Also

- [Pretraining & Fine-Tuning Reference](../documentation/reference/pretraining_and_finetuning.md) - Transfer learning workflows
- [Inference](../inference/) - Using pretrained models for prediction
- [Training](../training/) - Training from pretrained weights
