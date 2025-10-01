# Inference

This module handles prediction on new images using trained models. It includes sliding window inference, test-time augmentation, multi-model ensembling, and efficient memory management.

## Overview

The inference pipeline:

1. **Preprocessing** - Apply same transformations as training (crop, resample, normalize)
2. **Prediction** - Sliding window with optional test-time augmentation
3. **Postprocessing** - Reverse transformations, apply postprocessing strategies
4. **Export** - Save predictions in original image space

## Directory Structure

```
inference/
├── predictor/
│   ├── predict_from_raw_data.py       # Main predictor class
│   ├── sliding_window_prediction.py   # Sliding window implementation
│   ├── export_prediction.py           # Prediction export utilities
│   ├── utils.py                       # Inference utilities
│   └── ...
└── __init__.py
```

## Key Components

### Predictor (`predictor/predict_from_raw_data.py`)

The `nnUNetPredictor` class orchestrates the inference pipeline:

**Responsibilities**:
- Load trained model(s) and plans
- Preprocess input images
- Perform sliding window prediction
- Apply test-time augmentation (mirroring)
- Reverse preprocessing transformations
- Export predictions to disk

**Key methods**:
- `initialize_from_trained_model_folder()` - Load model from folder
- `predict_from_files()` - Predict on image files
- `predict_single_npy_array()` - Predict on numpy array
- `predict_sliding_window()` - Core sliding window logic

### Sliding Window Prediction (`predictor/sliding_window_prediction.py`)

Handles prediction on large images that don't fit in GPU memory:

**Strategy**:
1. Divide image into overlapping patches
2. Predict on each patch independently
3. Aggregate overlapping predictions (Gaussian weighting)
4. Return full-size prediction

**Parameters**:
- `tile_step_size` - Overlap between patches (0.5 = 50% overlap)
- `use_gaussian` - Weight patches by Gaussian (emphasizes center)
- `use_mirroring` - Test-time augmentation via mirroring

### Test-Time Augmentation

Improves predictions by averaging over augmented versions:

**Mirroring axes**:
- Flip image along spatial axes
- Predict on flipped version
- Flip prediction back
- Average with original prediction

**Supported mirrors**:
- 2D: Horizontal, vertical, both
- 3D: All 3 axes, all combinations (8 total)

**Enable**:
```bash
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -f 0 --disable_tta  # Disable (default: enabled)
```

### Multi-Model Ensembling

Not in this module (see `ensembling/`), but predictor supports loading multiple models:

```python
predictor.initialize_from_trained_model_folder(
    model_folder,
    use_folds=(0, 1, 2, 3, 4),  # Ensemble all 5 folds
    checkpoint_name='checkpoint_best.pth'
)
```

Predictions from all folds are averaged before argmax.

## Usage

### CLI Prediction

**Basic prediction**:
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIG -f FOLD
```

**Examples**:

```bash
# Single fold
nnUNetv2_predict -i /data/test/images -o /data/test/predictions -d 001 -c 3d_fullres -f 0

# Ensemble all folds
nnUNetv2_predict -i /data/test/images -o /data/test/predictions -d 001 -c 3d_fullres -f all

# Specific folds
nnUNetv2_predict -i /data/test/images -o /data/test/predictions -d 001 -c 3d_fullres -f 0 1 2

# Disable test-time augmentation (faster)
nnUNetv2_predict -i /data/test/images -o /data/test/predictions -d 001 -c 3d_fullres -f 0 --disable_tta

# Use specific checkpoint
nnUNetv2_predict -i /data/test/images -o /data/test/predictions -d 001 -c 3d_fullres -f 0 -chk checkpoint_best

# Save softmax probabilities
nnUNetv2_predict -i /data/test/images -o /data/test/predictions -d 001 -c 3d_fullres -f 0 --save_probabilities
```

**Flags**:
- `-i` / `--input_folder` - Input images folder
- `-o` / `--output_folder` - Output predictions folder
- `-d` / `--dataset` - Dataset ID or name
- `-c` / `--configuration` - Configuration (2d, 3d_fullres, etc.)
- `-f` / `--folds` - Fold(s) to use (0-4, or 'all')
- `-tr` / `--trainer` - Trainer name (if using custom trainer)
- `-p` / `--plans_identifier` - Plans identifier
- `-chk` / `--checkpoint_name` - Checkpoint name (default: checkpoint_final.pth)
- `--disable_tta` - Disable test-time augmentation
- `--save_probabilities` - Save softmax probabilities (not just argmax)
- `--step_size` - Tile step size for sliding window (default: 0.5)

### Predict from Model Folder

Directly from a model folder (without dataset ID):

```bash
nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER -f FOLD
```

**Use case**: Pretrained models, shared models, custom training locations.

### Programmatic API

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Initialize predictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

# Load model
predictor.initialize_from_trained_model_folder(
    model_training_output_dir='nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres',
    use_folds=(0,),
    checkpoint_name='checkpoint_final.pth'
)

# Predict on files
predictor.predict_from_files(
    list_of_lists_or_source_folder='/data/test/images',
    output_folder_or_list_of_truncated_output_files='/data/test/predictions',
    save_probabilities=False,
    overwrite=True,
    num_processes_preprocessing=4,
    num_processes_segmentation_export=4
)
```

### Predict on Numpy Array

For in-memory prediction:

```python
# Load image as numpy array
import numpy as np
image = np.load('image.npy')  # Shape: [C, X, Y, Z]

# Predict
properties = {'spacing': [1.0, 1.0, 1.0]}  # Must provide spacing
prediction = predictor.predict_single_npy_array(
    input_image=image,
    image_properties=properties,
    segmentation_previous_stage=None,
    output_file_truncated=None,
    save_or_return_probabilities=False
)

# prediction shape: [X, Y, Z] (argmax of class probabilities)
```

## Input Format

### File-Based Prediction

**Input folder structure**:
```
input_folder/
  case_001_0000.nii.gz    # Channel 0
  case_001_0001.nii.gz    # Channel 1 (if multi-channel)
  case_002_0000.nii.gz
  case_002_0001.nii.gz
  ...
```

**Channel naming**:
- Single channel: `case_XXX_0000.ext`
- Multi-channel: `case_XXX_0000.ext`, `case_XXX_0001.ext`, ...

**Supported formats**: Any format supported by `imageio` module (NIfTI, TIFF, PNG, etc.)

### List of Files

Pass explicit file lists:

```python
predictor.predict_from_files(
    list_of_lists_or_source_folder=[
        ['case_001_0000.nii.gz', 'case_001_0001.nii.gz'],  # Case 1
        ['case_002_0000.nii.gz', 'case_002_0001.nii.gz'],  # Case 2
    ],
    output_folder_or_list_of_truncated_output_files=[
        'output/case_001',  # Will add .nii.gz extension
        'output/case_002'
    ]
)
```

## Output Format

### Segmentation Maps

**Default output**: Argmax segmentation (integer labels)

```
output_folder/
  case_001.nii.gz    # Label map: 0 = background, 1 = class 1, etc.
  case_002.nii.gz
  ...
```

**Spacing**: Predictions are resampled to original input spacing.

**Shape**: Predictions match original input shape (before preprocessing).

### Probability Maps

**Enable with** `--save_probabilities`:

```
output_folder/
  case_001.npz       # Contains 'probabilities' array [num_classes, X, Y, Z]
  case_002.npz
  ...
```

**Contents**:
```python
data = np.load('case_001.npz')
probabilities = data['probabilities']  # Shape: [num_classes, X, Y, Z]
# Softmax probabilities for each class
```

## Memory Management

### Tile Size (Patch Size)

**Determined by**:
- `patch_size` from plans file
- `tile_step_size` (overlap)

**GPU memory usage**:
- Larger patches → more memory
- Test-time augmentation → 2-8x memory (multiple forward passes)

**Out of memory**:
1. Disable TTA: `--disable_tta`
2. Reduce patch size in plans (requires retraining)
3. Use smaller model

### Tile Step Size

**Controls overlap** between sliding window patches:

- `tile_step_size=0.5` - 50% overlap (default)
- `tile_step_size=1.0` - No overlap (faster, lower quality)
- `tile_step_size=0.25` - 75% overlap (slower, higher quality)

**Trade-off**:
- More overlap → better predictions (smoother), slower
- Less overlap → faster, potential artifacts at boundaries

## Prediction Tips

### Speed Optimization

**Faster prediction**:
1. Disable TTA: `--disable_tta` (2-8x speedup)
2. Increase tile step size: `--step_size 1.0` (2-4x speedup, lower quality)
3. Use single fold instead of ensemble
4. Use smaller model (2D instead of 3D, fewer stages)

**Typical prediction time** (3D, single case):
- With TTA: 30-60 seconds
- Without TTA: 5-15 seconds

### Quality Optimization

**Best quality**:
1. Enable TTA (default)
2. Ensemble multiple folds: `-f all`
3. Use `checkpoint_best.pth` instead of `checkpoint_final.pth`
4. Lower tile step size: `--step_size 0.5`

### Multi-GPU Prediction

Not directly supported, but can parallelize across cases:

```bash
# GPU 0: Cases 0-49
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i input -o output -d 001 -c 3d_fullres -f 0 &

# GPU 1: Cases 50-99  
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i input2 -o output2 -d 001 -c 3d_fullres -f 0 &

wait
```

Or use case-level parallelization via `num_processes_segmentation_export`.

### Cascade Prediction

For 3D cascade configurations:

**Step 1**: Predict low-res
```bash
nnUNetv2_predict -i input -o output_lowres -d 001 -c 3d_lowres -f all
```

**Step 2**: Predict full-res (using low-res as input)
```bash
nnUNetv2_predict -i input -o output_fullres -d 001 -c 3d_cascade_fullres -f all -prev_stage_predictions output_lowres
```

The cascade uses low-res predictions as an additional input channel to the full-res model.

## Advanced Usage

### Custom Postprocessing

Apply custom postprocessing after prediction:

```python
predictor.predict_from_files(
    list_of_lists_or_source_folder=input_folder,
    output_folder_or_list_of_truncated_output_files=output_folder
)

# Then apply your custom postprocessing
from nnunetv2.postprocessing.custom_pp import my_postprocess

for file in output_folder:
    seg = load_segmentation(file)
    seg_pp = my_postprocess(seg)
    save_segmentation(seg_pp, file)
```

See `postprocessing/` module.

### Predict with Uncertainty

Using dropout at test time:

Requires custom trainer with dropout enabled at test time:

```python
class UncertaintyPredictor(nnUNetPredictor):
    def predict_logits_from_preprocessed_data(self, data):
        """Enable dropout during inference."""
        self.network.train()  # Enable dropout
        
        # Multiple forward passes
        predictions = []
        for _ in range(10):
            predictions.append(self.network(data))
        
        # Mean prediction
        return torch.stack(predictions).mean(dim=0)
```

### Export to Different Format

Predictions are saved in same format as input by default. To change:

```python
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# After prediction, convert
for file in output_files:
    seg, props = SimpleITKIO().read_seg(file)
    
    # Save as PNG, TIFF, etc.
    save_as_png(seg, file.replace('.nii.gz', '.png'))
```

## See Also

- [Ensembling](../ensembling/) - Multi-model ensemble utilities
- [Postprocessing](../postprocessing/) - Postprocessing strategies
- [Model Sharing](../model_sharing/) - Pretrained model usage
- [Inference Format Reference](../documentation/reference/inference_format.md) - Input dataset format
