# Ensembling

This module provides utilities for combining predictions from multiple models to improve segmentation quality.

## Overview

Ensembling combines predictions from:

- **Multiple folds** - Average predictions across cross-validation folds
- **Multiple configurations** - Combine 2D, 3D full-res, cascade predictions
- **Multiple architectures** - Ensemble U-Net, KiU-Net, etc.
- **Multiple trainers** - Different training strategies

**Benefits**:
- Improved accuracy (typically +1-3% Dice)
- Reduced variance
- More robust predictions

## Directory Structure

```
ensembling/
└── ensemble.py  # Ensemble prediction implementation
```

## How Ensembling Works

### Soft Ensemble (Default)

1. Each model outputs class probabilities (softmax)
2. Average probabilities across all models
3. Take argmax to get final segmentation

**Formula**:
```
P_ensemble(class) = mean([P_model1(class), P_model2(class), ...])
segmentation = argmax(P_ensemble)
```

**Advantages**:
- Optimal for combining diverse models
- Handles confidence differences between models

### Hard Ensemble (Alternative)

1. Each model outputs segmentation (argmax)
2. Majority vote across models
3. Ties broken arbitrarily

**Formula**:
```
segmentation = mode([seg_model1, seg_model2, ...])
```

**Disadvantages**:
- Less optimal than soft ensemble
- Loses probability information

**Note**: nnU-Net uses soft ensemble by default.

## Usage

### CLI Ensemble

**Basic ensembling**:
```bash
nnUNetv2_ensemble -i PRED_FOLDER1 PRED_FOLDER2 ... -o OUTPUT_FOLDER
```

**Examples**:

**Ensemble multiple folds**:
```bash
nnUNetv2_ensemble \
    -i nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation_raw \
       nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation_raw \
       nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/validation_raw \
    -o ensemble_output
```

**Ensemble different configurations**:
```bash
nnUNetv2_ensemble \
    -i predictions_2d \
       predictions_3d_fullres \
    -o ensemble_2d_3d
```

**Ensemble with custom weights**:
```bash
# Not supported in CLI by default - use programmatic API
```

**Requirements**:
- All prediction folders must contain same cases
- Predictions must have same shape (or be in probability format for resampling)

### Programmatic API

```python
from nnunetv2.ensembling.ensemble import ensemble_folders

ensemble_folders(
    input_folders=[
        'predictions_fold_0',
        'predictions_fold_1',
        'predictions_fold_2'
    ],
    output_folder='ensemble_output',
    num_processes=8,
    save_npz=True  # Save probability maps
)
```

**Ensemble with custom weights**:

```python
import numpy as np

# Load predictions
pred1 = np.load('pred1.npz')['probabilities']  # [C, X, Y, Z]
pred2 = np.load('pred2.npz')['probabilities']
pred3 = np.load('pred3.npz')['probabilities']

# Weighted average (e.g., based on validation performance)
weights = [0.5, 0.3, 0.2]  # Model 1 has highest weight
ensemble_probs = weights[0] * pred1 + weights[1] * pred2 + weights[2] * pred3

# Argmax
ensemble_seg = np.argmax(ensemble_probs, axis=0)
```

## Ensemble Strategies

### Multi-Fold Ensemble

**Use case**: Reduce variance, improve robustness

**Benefit**: ~0.5-1% Dice improvement

**How**:
```bash
# Option 1: During prediction (automatic)
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -f all

# Option 2: Manual ensemble
nnUNetv2_ensemble -i fold_0/predictions fold_1/predictions ... -o ensemble
```

### Multi-Configuration Ensemble

**Use case**: Combine complementary perspectives (2D + 3D)

**Benefit**: ~1-2% Dice improvement

**Common combinations**:
- **2D + 3D full-res** - 2D captures in-plane details, 3D captures 3D context
- **3D full-res + 3D cascade** - Cascade adds multi-scale reasoning

**How**:
```bash
# 1. Train both configurations
nnUNetv2_train 001 2d all
nnUNetv2_train 001 3d_fullres all

# 2. Predict with both
nnUNetv2_predict -i INPUT -o pred_2d -d 001 -c 2d -f all
nnUNetv2_predict -i INPUT -o pred_3d -d 001 -c 3d_fullres -f all

# 3. Ensemble
nnUNetv2_ensemble -i pred_2d pred_3d -o ensemble_2d_3d
```

**Recommended by nnU-Net**: Use `nnUNetv2_find_best_configuration` to automatically find best ensemble.

### Multi-Architecture Ensemble

**Use case**: Combine different network architectures

**Benefit**: Variable (depends on architecture diversity)

**Example**: U-Net + KiU-Net

**How**:
```bash
# 1. Train both architectures
nnUNetv2_train 001 3d_fullres all  # Standard U-Net
nnUNetv2_train 001 3d_fullres all -tr kiunet  # KiU-Net

# 2. Predict with both
nnUNetv2_predict -i INPUT -o pred_unet -d 001 -c 3d_fullres -f all
nnUNetv2_predict -i INPUT -o pred_kiunet -d 001 -c 3d_fullres -f all -tr kiunet

# 3. Ensemble
nnUNetv2_ensemble -i pred_unet pred_kiunet -o ensemble_architectures
```

## Requirements & Compatibility

### Probability vs Segmentation Files

**Soft ensemble** (recommended):
- Requires probability maps (`.npz` files with `probabilities` key)
- Enable with `--save_probabilities` during prediction

**Example**:
```bash
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -f 0 --save_probabilities
```

**Hard ensemble**:
- Works with segmentation files (`.nii.gz`, `.npy`)
- Less optimal than soft ensemble

### Shape Compatibility

**Same shape**:
- All predictions must have same spatial dimensions
- Automatic if using same configuration and preprocessing

**Different shapes**:
- Ensemble on probabilities (resampled if needed)
- Or resample all to common space before ensembling

## Performance Considerations

### Computational Cost

**Ensemble overhead**:
- Soft ensemble: Fast (just averaging)
- Hard ensemble: Very fast (mode operation)

**Total inference time**:
- Multiply by number of models (e.g., 5 folds = 5x inference time)

**Optimization**:
- Parallelize inference across models (different GPUs)
- Use fewer folds (e.g., 3 instead of 5)

### Diminishing Returns

**Typical improvements**:
- 2 models: +1-2% Dice
- 3 models: +1.5-2.5% Dice
- 5 models: +2-3% Dice
- >5 models: Marginal gains (<0.5%)

**Trade-off**:
- More models → better accuracy, slower inference
- Decide based on application needs

## Advanced Usage

### Weighted Ensemble

Weight models by validation performance:

```python
import numpy as np

# Validation Dice scores
dice_scores = [0.85, 0.87, 0.83]

# Compute weights (proportional to Dice)
weights = np.array(dice_scores)
weights = weights / weights.sum()  # Normalize to sum to 1

# Weighted ensemble
ensemble_probs = sum(w * prob for w, prob in zip(weights, predictions))
```

### Ensemble with Uncertainty

Use prediction variance as uncertainty estimate:

```python
import numpy as np

# Ensemble and compute variance
ensemble_probs = np.mean(predictions, axis=0)
variance = np.var(predictions, axis=0)

# High variance → low confidence regions
uncertainty = variance.max(axis=0)  # Max variance across classes
```

### Selective Ensemble

Ensemble only on low-confidence regions:

```python
import numpy as np

# Get best single model prediction
best_pred = predictions[0]
best_probs = probabilities[0]

# Identify low-confidence regions
confidence = best_probs.max(axis=0)
low_confidence_mask = confidence < 0.8

# Ensemble only in low-confidence regions
ensemble_probs = np.mean(probabilities, axis=0)
final_probs = np.where(low_confidence_mask, ensemble_probs, best_probs)
final_seg = np.argmax(final_probs, axis=0)
```

## Tips

### When to Ensemble

**Always ensemble**:
- Challenge submissions (maximize performance)
- Critical applications (medical diagnosis)

**Consider not ensembling**:
- Real-time applications (latency constraints)
- Resource-limited deployment
- Single model already sufficient

### Best Ensemble Practices

1. **Train diverse models**: Different configurations, augmentations, architectures
2. **Use validation to weight**: Weight by cross-validation Dice
3. **Test ensemble benefit**: Compare ensemble vs best single model
4. **Use find_best_configuration**: Automatic ensemble selection

### Debugging Ensemble

**Check alignment**:
```python
# Ensure all predictions have same cases
import os

folders = ['pred1', 'pred2', 'pred3']
files = [set(os.listdir(f)) for f in folders]

if len(set.intersection(*files)) == 0:
    print("ERROR: No common files across folders")
else:
    print(f"Common files: {len(set.intersection(*files))}")
```

**Visualize ensemble effect**:
```python
import matplotlib.pyplot as plt

# Mid-slice visualization
slice_idx = pred1.shape[2] // 2

plt.subplot(1, 4, 1)
plt.imshow(pred1[:, :, slice_idx])
plt.title('Model 1')

plt.subplot(1, 4, 2)
plt.imshow(pred2[:, :, slice_idx])
plt.title('Model 2')

plt.subplot(1, 4, 3)
plt.imshow(ensemble[:, :, slice_idx])
plt.title('Ensemble')

plt.subplot(1, 4, 4)
plt.imshow(ground_truth[:, :, slice_idx])
plt.title('Ground Truth')

plt.show()
```

## See Also

- [Inference](../inference/) - Generating predictions for ensembling
- [Evaluation](../evaluation/) - Comparing ensemble vs single model performance
- [Configuration Comparison](../evaluation/) - `nnUNetv2_find_best_configuration` for automatic ensemble selection
