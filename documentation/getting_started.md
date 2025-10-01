# Getting Started with nnU-Net

This guide walks you through installation, dataset preparation, and running your first training.

## Installation

### Requirements

- **OS**: Linux (Ubuntu, CentOS, RHEL), Windows, or macOS
- **Python**: 3.10 or newer
- **Hardware**: 
  - Training: GPU with 10+ GB VRAM (e.g., RTX 3090, RTX 4090)
  - Inference: GPU with 4+ GB VRAM (or CPU/Apple M1/M2)
  - CPU: 6+ cores (12 threads) minimum for training
  - RAM: 32GB minimum, 64GB recommended
  - Storage: SSD strongly recommended

### Step 1: Install PyTorch

First, install PyTorch for your hardware. Visit [pytorch.org](https://pytorch.org/get-started/locally/) and follow the instructions for your system.

```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Example for CPU
pip install torch torchvision

# Example for Apple Silicon
pip install torch torchvision
```

**Do not skip this step!** nnU-Net requires PyTorch to be properly installed first.

### Step 2: Install nnU-Net

Choose based on your use case:

**For standard usage** (baseline segmentation, pretrained models):
```bash
pip install nnunetv2
```

**For development** (modifying code, custom extensions):
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

### Step 3: Configure Paths

nnU-Net needs to know where to store data. Set these environment variables (add to `~/.bashrc` or `~/.zshrc` for persistence):

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

Create these directories:
```bash
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```

### Step 4: Verify Installation

```bash
nnUNetv2_train --help
```

If this shows the help message, you're ready to go!

### Optional: Tuning for Your Hardware

Set the number of data augmentation workers based on your CPU/GPU ratio:

```bash
export nnUNet_n_proc_DA=12  # For RTX 3090
export nnUNet_n_proc_DA=16  # For RTX 4090
export nnUNet_n_proc_DA=28  # For A100
```

## Dataset Preparation

nnU-Net expects a specific directory structure. Each dataset has a unique 3-digit ID.

### Directory Structure

```
nnUNet_raw/
  DatasetXXX_YourDatasetName/
    imagesTr/              # Training images
    labelsTr/              # Training labels (segmentation masks)
    imagesTs/              # Test images (optional)
    dataset.json           # Dataset metadata
```

### Image Naming Convention

Images and labels must follow this pattern:

```
# For single-channel images
imagesTr/case_identifier_0000.nii.gz
labelsTr/case_identifier.nii.gz

# For multi-channel images (e.g., 4 MRI sequences)
imagesTr/case_identifier_0000.nii.gz  # Channel 0
imagesTr/case_identifier_0001.nii.gz  # Channel 1
imagesTr/case_identifier_0002.nii.gz  # Channel 2
imagesTr/case_identifier_0003.nii.gz  # Channel 3
labelsTr/case_identifier.nii.gz       # Corresponding label
```

**Important**: 
- Case identifiers must be unique within a dataset
- Channel indices start at `0000` and increment (`0001`, `0002`, ...)
- Labels use the same identifier but no channel suffix

### Supported File Formats

- `.nii.gz` (NIfTI, most common for medical imaging)
- `.nii` (uncompressed NIfTI)
- `.png` (2D images)
- `.tif`, `.tiff` (2D/3D images)

### dataset.json

This file describes your dataset:

```json
{
  "channel_names": {
    "0": "T1",
    "1": "T2"
  },
  "labels": {
    "background": 0,
    "tumor": 1,
    "edema": 2
  },
  "numTraining": 120,
  "file_ending": ".nii.gz"
}
```

**Key fields**:
- `channel_names`: Map channel index to modality name
- `labels`: Map class name to integer label (0 is always background)
- `numTraining`: Number of training cases
- `file_ending`: File extension used

### Example: Creating a Dataset

Let's create a simple 2-class brain tumor dataset:

```bash
# Create dataset folder
mkdir -p $nnUNet_raw/Dataset001_BrainTumor/imagesTr
mkdir -p $nnUNet_raw/Dataset001_BrainTumor/labelsTr
mkdir -p $nnUNet_raw/Dataset001_BrainTumor/imagesTs

# Copy your data (adjust paths)
cp /your/data/patient001_t1.nii.gz $nnUNet_raw/Dataset001_BrainTumor/imagesTr/patient001_0000.nii.gz
cp /your/data/patient001_label.nii.gz $nnUNet_raw/Dataset001_BrainTumor/labelsTr/patient001.nii.gz
# ... repeat for all cases

# Create dataset.json
cat > $nnUNet_raw/Dataset001_BrainTumor/dataset.json << EOF
{
  "channel_names": {
    "0": "T1"
  },
  "labels": {
    "background": 0,
    "tumor": 1
  },
  "numTraining": 100,
  "file_ending": ".nii.gz"
}
EOF
```

## Your First Training Run

### Step 1: Plan and Preprocess

This analyzes your dataset and configures the pipeline:

```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

**What happens**:
1. Dataset integrity check (spacing, dimensions, labels)
2. Dataset fingerprint extraction (statistics, properties)
3. Plan generation (3-5 different U-Net configurations)
4. Preprocessing (resampling, normalization)

This takes 10 minutes to a few hours depending on dataset size.

Output goes to `$nnUNet_preprocessed/Dataset001_BrainTumor/`.

### Step 2: Train a Model

Train the 3D full-resolution U-Net on fold 0:

```bash
nnUNetv2_train 001 3d_fullres 0 --npz
```

**Arguments**:
- `001`: Dataset ID
- `3d_fullres`: Configuration (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres)
- `0`: Fold number (0-4 for 5-fold cross-validation)
- `--npz`: Save softmax outputs for later analysis (optional but recommended)

**Training time**: Varies widely (hours to days) depending on:
- Dataset size
- Image dimensions
- Number of classes
- GPU speed

The model checkpoints save to `$nnUNet_results/Dataset001_BrainTumor/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/`.

### Step 3: Run Inference

Apply your trained model to new images:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 001 -c 3d_fullres -f 0
```

**Arguments**:
- `-i`: Folder with test images (same naming convention as training)
- `-o`: Output folder for predictions
- `-d`: Dataset ID
- `-c`: Configuration used for training
- `-f`: Fold(s) to use (can specify multiple: `-f 0 1 2 3 4` for ensemble)

Predictions are saved as NIfTI files in `OUTPUT_FOLDER`.

## Cross-Validation and Ensembling

For best results, train all 5 folds:

```bash
for fold in 0 1 2 3 4; do
  nnUNetv2_train 001 3d_fullres $fold --npz
done
```

Then use all 5 for ensemble prediction:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 001 -c 3d_fullres -f 0 1 2 3 4
```

This averages predictions from all 5 models, typically improving performance.

## Choosing the Best Configuration

If you train multiple configurations (2d, 3d_fullres, etc.), let nnU-Net automatically determine the best:

```bash
# Train all configurations
nnUNetv2_train 001 2d all --npz
nnUNetv2_train 001 3d_fullres all --npz

# Find best
nnUNetv2_find_best_configuration 001 -c 2d 3d_fullres
```

This compares cross-validation performance and tells you which configuration to use for final predictions.

## Troubleshooting

### Out of Memory (OOM)

If training crashes with OOM:
1. Check GPU memory: `nvidia-smi`
2. Reduce batch size: Edit `$nnUNet_preprocessed/Dataset001_BrainTumor/nnUNetPlans.json`
3. Use a smaller configuration (try `2d` instead of `3d_fullres`)
4. Disable torch.compile: `export nnUNet_compile=false`

### Slow Training

- Increase `nnUNet_n_proc_DA` (data augmentation workers)
- Check CPU usage during training
- Ensure data is on SSD, not HDD
- Verify GPU utilization with `nvidia-smi`

### Dataset Errors

If `--verify_dataset_integrity` fails:
- Check file naming (channels start at `0000`)
- Ensure all cases have the same number of channels
- Verify labels contain only expected integer values
- Check that all images have compatible shapes

## Next Steps

- **[Core Concepts](core_concepts.md)**: Understand how nnU-Net works internally
- **[Advanced Usage](advanced_usage.md)**: Fine-tuning, custom architectures, manual configuration
- **[Reference](reference/)**: Detailed API documentation

## Quick Reference

```bash
# Full pipeline
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID CONFIG FOLD --npz
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c CONFIG -f FOLD

# Common configurations
CONFIG = 2d | 3d_fullres | 3d_lowres | 3d_cascade_fullres

# Train all folds at once
nnUNetv2_train DATASET_ID CONFIG all

# Ensemble prediction
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c CONFIG -f 0 1 2 3 4

# Continue interrupted training
nnUNetv2_train DATASET_ID CONFIG FOLD --c
```
