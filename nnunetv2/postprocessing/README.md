# Postprocessing

This module provides postprocessing strategies to refine segmentation predictions after inference.

## Overview

Postprocessing applies rule-based refinements to improve segmentation quality:

- **Connected component analysis** - Remove small false positive regions
- **Automatic determination** - Find optimal postprocessing on validation set
- **Per-class strategies** - Different postprocessing for each class

## Directory Structure

```
postprocessing/
└── remove_connected_components.py  # Connected component filtering
```

## Key Component

### Remove Connected Components (`remove_connected_components.py`)

Removes small connected components (isolated regions) from predictions.

**Rationale**:
- Predictions may contain small false positive regions
- Removing isolated components below a size threshold can improve precision
- Trade-off: May remove small true positives

## Usage

### Determine Optimal Postprocessing

Automatically finds best postprocessing strategy on validation data:

**CLI**:
```bash
nnUNetv2_determine_postprocessing -tr TRAINER -c CONFIG -d DATASET_ID
```

**Example**:
```bash
nnUNetv2_determine_postprocessing \
    -tr nnUNetTrainer \
    -c 3d_fullres \
    -d 001
```

**What it does**:
1. Loads cross-validation predictions
2. Tests different component size thresholds per class
3. Computes metrics (Dice) for each threshold
4. Selects threshold that maximizes Dice
5. Saves postprocessing plan to `postprocessing.pkl`

**Output**:
```
nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres/
  └── postprocessing.pkl
```

**Postprocessing plan** contains:
```python
{
    0: None,      # Background: no postprocessing
    1: 50,        # Class 1: remove components < 50 voxels
    2: None,      # Class 2: no postprocessing
    3: 100        # Class 3: remove components < 100 voxels
}
```

### Apply Postprocessing

**Automatic** (during prediction):

If `postprocessing.pkl` exists in model folder, it's automatically applied:

```bash
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -f 0
# Postprocessing applied automatically
```

**Manual** (on existing predictions):

```bash
nnUNetv2_apply_postprocessing \
    -i PREDICTIONS_FOLDER \
    -o OUTPUT_FOLDER \
    -pp_pkl_file postprocessing.pkl
```

### Disable Postprocessing

To predict without postprocessing:

```bash
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -f 0 --disable_postprocessing
```

Or remove `postprocessing.pkl` from model folder.

## How It Works

### Connected Component Filtering

**Algorithm**:
1. For each class in prediction:
   - Find connected components (26-connectivity for 3D, 8-connectivity for 2D)
   - Compute size of each component (voxel count)
   - Remove components smaller than threshold
2. Combine filtered classes into final prediction

**Example**:

```
Before postprocessing:
Class 1 components: [5000 voxels, 30 voxels, 20 voxels]
Threshold: 50 voxels

After postprocessing:
Class 1 components: [5000 voxels]
(30 and 20 voxel components removed)
```

### Threshold Determination

**Strategy**:
- Test thresholds: 0 (no filtering), 10, 20, 50, 100, 200, 500, 1000, 2000, 5000 voxels
- For each threshold, apply to validation predictions
- Compute Dice score
- Select threshold with highest Dice

**Per-class thresholds**:
- Each class has independent threshold
- Some classes may benefit from aggressive filtering (many small false positives)
- Others may not (small true positives would be removed)

## Programmatic API

### Determine Postprocessing

```python
from nnunetv2.postprocessing.remove_connected_components import determine_postprocessing

postprocessing_plan = determine_postprocessing(
    training_output_dir='nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres',
    dataset_json_file='nnUNet_raw/Dataset001_Name/dataset.json',
    folds=(0, 1, 2, 3, 4)
)

# Save plan
import pickle
with open('postprocessing.pkl', 'wb') as f:
    pickle.dump(postprocessing_plan, f)
```

### Apply Postprocessing

```python
from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
import pickle

# Load plan
with open('postprocessing.pkl', 'rb') as f:
    postprocessing_plan = pickle.load(f)

# Apply to folder
apply_postprocessing_to_folder(
    input_folder='predictions',
    output_folder='predictions_postprocessed',
    postprocessing_plan=postprocessing_plan,
    num_processes=8
)
```

### Apply to Single Prediction

```python
from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing
import numpy as np

# Load prediction
prediction = np.load('prediction.npy')

# Apply postprocessing
postprocessed = apply_postprocessing(
    segmentation=prediction,
    postprocessing_plan={1: 50, 2: 100}  # Class 1: >50 voxels, Class 2: >100 voxels
)
```

## When Postprocessing Helps

### Good Candidates

Postprocessing is beneficial when:
- **Many small false positives** - Network predicts scattered small regions
- **High precision already** - Won't hurt by removing small true positives
- **Large target structures** - Small components are likely false positives

**Example scenarios**:
- Organ segmentation (liver, kidney) - large targets, small FPs removed
- Lesion detection - remove noise while keeping lesions

### Poor Candidates

Postprocessing may hurt when:
- **Small targets** - True positives may be below threshold
- **Scattered targets** - Multiple small instances (e.g., cells, nuclei)
- **High recall required** - Can't afford to miss any true positives

**Example scenarios**:
- Cell counting - small components are the target
- Metastasis detection - small lesions are critical

## Custom Postprocessing

For more complex postprocessing beyond component filtering:

```python
from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing

def my_custom_postprocessing(segmentation):
    """Custom postprocessing logic."""
    
    # Apply standard component filtering
    seg = apply_postprocessing(segmentation, postprocessing_plan={1: 50})
    
    # Custom refinement: fill holes
    from scipy.ndimage import binary_fill_holes
    for class_id in [1, 2, 3]:
        class_mask = seg == class_id
        filled = binary_fill_holes(class_mask)
        seg[filled] = class_id
    
    # Custom refinement: morphological closing
    from scipy.ndimage import binary_closing
    for class_id in [1, 2]:
        class_mask = seg == class_id
        closed = binary_closing(class_mask, structure=np.ones((3, 3, 3)))
        seg[closed] = class_id
    
    return seg
```

## Advanced Topics

### Multi-Stage Postprocessing

Apply different strategies sequentially:

```python
# Stage 1: Component filtering
seg = apply_postprocessing(seg, {1: 50, 2: 100})

# Stage 2: Hole filling
seg = fill_holes(seg)

# Stage 3: Boundary smoothing
seg = smooth_boundaries(seg)
```

### Conditional Postprocessing

Apply different postprocessing based on image properties:

```python
if image_size > 1000**3:  # Large image
    postprocessing_plan = {1: 500, 2: 1000}  # Aggressive filtering
else:  # Small image
    postprocessing_plan = {1: 50, 2: 100}  # Conservative filtering

seg = apply_postprocessing(seg, postprocessing_plan)
```

### Region-Based Postprocessing

Apply postprocessing only in specific regions:

```python
# Remove small components only in periphery, not in central region
mask_periphery = get_periphery_mask(image)
seg_periphery = seg * mask_periphery
seg_periphery = apply_postprocessing(seg_periphery, {1: 100})

# Combine
seg = seg * (1 - mask_periphery) + seg_periphery
```

## Tips

### When to Determine Postprocessing

**Timing**:
- After all folds are trained
- After cross-validation predictions are generated

**Typical workflow**:
```bash
# 1. Train all folds
nnUNetv2_train 001 3d_fullres all

# 2. Accumulate cross-validation results
nnUNetv2_accumulate_crossval_results \
    nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres

# 3. Determine postprocessing
nnUNetv2_determine_postprocessing -tr nnUNetTrainer -c 3d_fullres -d 001
```

### Debugging Postprocessing

**Visualize effect**:
```python
import matplotlib.pyplot as plt

# Before
plt.subplot(1, 2, 1)
plt.imshow(prediction[prediction.shape[0]//2])
plt.title('Before')

# After
plt.subplot(1, 2, 2)
plt.imshow(postprocessed[postprocessed.shape[0]//2])
plt.title('After')

plt.show()
```

**Compare metrics**:
```bash
# Evaluate without postprocessing
nnUNetv2_evaluate_folder GT_FOLDER PRED_FOLDER

# Evaluate with postprocessing
nnUNetv2_evaluate_folder GT_FOLDER PRED_POSTPROCESSED_FOLDER

# Compare Dice scores
```

### Computational Cost

**Component analysis**:
- Fast for 2D images
- Can be slow for large 3D volumes (seconds per case)

**Optimization**:
- Use `num_processes` for parallel processing
- Only apply to classes that benefit (check postprocessing.pkl)

## See Also

- [Inference](../inference/) - Prediction pipeline where postprocessing is applied
- [Evaluation](../evaluation/) - Evaluating postprocessing effectiveness
