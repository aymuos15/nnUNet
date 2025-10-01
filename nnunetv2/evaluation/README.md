# Evaluation

This module provides tools for evaluating segmentation predictions, comparing configurations, and aggregating cross-validation results.

## Overview

The evaluation module handles:

- **Prediction evaluation** - Compute metrics (Dice, Hausdorff, etc.) between predictions and ground truth
- **Cross-validation aggregation** - Summarize results across folds
- **Configuration comparison** - Find best configuration (2D, 3D, cascade)

## Directory Structure

```
evaluation/
├── evaluate_predictions.py      # Evaluate predictions against ground truth
├── accumulate_cv_results.py     # Aggregate cross-validation results
└── find_best_configuration.py   # Compare and select best configuration
```

## Key Components

### Evaluate Predictions (`evaluate_predictions.py`)

Computes segmentation metrics on predicted vs ground truth labels.

**CLI**:
```bash
nnUNetv2_evaluate_folder GROUND_TRUTH_FOLDER PREDICTIONS_FOLDER
```

**Example**:
```bash
nnUNetv2_evaluate_folder \
    nnUNet_raw/Dataset001_Name/labelsTs \
    predictions/Dataset001_Name/fold_0
```

**Output**:
- `summary.json` - Per-case and aggregate metrics
- Console output with mean metrics per class

**Metrics computed**:
- **Dice Coefficient** - Overlap between prediction and ground truth
- **Hausdorff Distance (95th percentile)** - Surface distance metric
- **Surface Dice** - Dice computed on surfaces (1mm tolerance)
- **Precision** - True positives / (true positives + false positives)
- **Recall** - True positives / (true positives + false negatives)

**Programmatic API**:
```python
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder

metrics = compute_metrics_on_folder(
    folder_ref='path/to/ground_truth',
    folder_pred='path/to/predictions',
    labels={"background": 0, "tumor": 1, "edema": 2},
    num_processes=8
)
```

**Output format** (`summary.json`):
```json
{
  "case_001": {
    "Dice": [1.0, 0.85, 0.78],  // Per-class Dice
    "Hausdorff95": [0.0, 2.3, 3.1],
    ...
  },
  "mean": {
    "Dice": [1.0, 0.83, 0.75],  // Mean across all cases
    ...
  }
}
```

### Accumulate Cross-Validation Results (`accumulate_cv_results.py`)

Aggregates validation results across all folds to provide overall cross-validation performance.

**CLI**:
```bash
nnUNetv2_accumulate_crossval_results TRAINED_MODEL_FOLDER
```

**Example**:
```bash
nnUNetv2_accumulate_crossval_results \
    nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres
```

**What it does**:
1. Loads validation predictions from each fold
2. Combines into single set (each case predicted by fold it wasn't trained on)
3. Evaluates combined predictions against ground truth
4. Saves aggregated metrics

**Output**:
- `cv_niftis_raw/` - Combined validation predictions from all folds
- `summary.json` - Cross-validation metrics

**Programmatic API**:
```python
from nnunetv2.evaluation.accumulate_cv_results import accumulate_crossval_results

accumulate_crossval_results(
    trained_model_folder='nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres',
    dataset_json_file='nnUNet_raw/Dataset001_Name/dataset.json'
)
```

### Find Best Configuration (`find_best_configuration.py`)

Compares different configurations (2D, 3D full-res, 3D low-res, cascade) to identify the best performing one.

**CLI**:
```bash
nnUNetv2_find_best_configuration DATASET_ID -c CONFIG1 CONFIG2 ...
```

**Example**:
```bash
# Compare all available configurations
nnUNetv2_find_best_configuration 001 -c 2d 3d_fullres 3d_lowres 3d_cascade_fullres

# Specific configurations
nnUNetv2_find_best_configuration 001 -c 2d 3d_fullres
```

**What it does**:
1. Accumulates cross-validation results for each configuration
2. Compares mean Dice scores
3. Recommends best configuration
4. Optionally tests ensembles (2D+3D, 3D+cascade, etc.)

**Output**:
```
Configuration Performance:
  2d:                 Mean Dice = 0.812
  3d_fullres:         Mean Dice = 0.847
  3d_cascade_fullres: Mean Dice = 0.851

Ensemble Performance:
  2d + 3d_fullres:    Mean Dice = 0.856
  all configurations: Mean Dice = 0.859

Best configuration: 3d_cascade_fullres (Dice = 0.851)
Best ensemble: all configurations (Dice = 0.859)
```

**Programmatic API**:
```python
from nnunetv2.evaluation.find_best_configuration import find_best_configuration

best_config, results = find_best_configuration(
    dataset_id=1,
    configurations=['2d', '3d_fullres'],
    trainer='nnUNetTrainer',
    plans_identifier='nnUNetPlans'
)

print(f"Best configuration: {best_config}")
```

## Metrics Explained

### Dice Coefficient

Measures overlap between prediction and ground truth:

```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```

- **Range**: 0 (no overlap) to 1 (perfect overlap)
- **Per-class**: Computed independently for each class
- **Mean Dice**: Average across all classes (excluding background)

**Interpretation**:
- `Dice > 0.9` - Excellent
- `Dice > 0.8` - Good
- `Dice > 0.7` - Acceptable
- `Dice < 0.7` - Poor

### Hausdorff Distance (95th percentile)

Measures maximum surface distance between prediction and ground truth:

```
HD95 = 95th percentile of max(d(A, B), d(B, A))
```

- **Range**: 0 (perfect alignment) to ∞ (large misalignment)
- **Units**: mm (millimeters) or voxels
- **95th percentile**: Robust to outliers (ignores worst 5% of distances)

**Interpretation**:
- `HD95 < 1mm` - Excellent boundary alignment
- `HD95 < 5mm` - Good
- `HD95 > 10mm` - Poor boundary alignment

### Surface Dice

Dice coefficient computed on surfaces with tolerance:

```
Surface Dice = |A_surf ∩ B_surf_tol| / (|A_surf| + |B_surf|)
```

- Tolerance typically 1mm
- More sensitive to boundary accuracy than volumetric Dice

### Precision and Recall

**Precision** (Positive Predictive Value):
```
Precision = TP / (TP + FP)
```
- Fraction of predicted voxels that are correct
- Low precision → many false positives

**Recall** (Sensitivity):
```
Recall = TP / (TP + FN)
```
- Fraction of ground truth voxels that are detected
- Low recall → many false negatives (missed regions)

## Usage Examples

### Basic Evaluation

Evaluate a single configuration:

```bash
# Run prediction (if not done yet)
nnUNetv2_predict -i INPUT -o OUTPUT -d 001 -c 3d_fullres -f 0

# Evaluate
nnUNetv2_evaluate_folder \
    nnUNet_raw/Dataset001_Name/labelsTs \
    OUTPUT
```

### Cross-Validation Evaluation

After training all folds:

```bash
# Train all folds (0-4)
nnUNetv2_train 001 3d_fullres all

# Aggregate cross-validation results
nnUNetv2_accumulate_crossval_results \
    nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres
```

### Configuration Comparison

Compare 2D vs 3D:

```bash
# Train both configurations
nnUNetv2_train 001 2d all
nnUNetv2_train 001 3d_fullres all

# Find best configuration
nnUNetv2_find_best_configuration 001 -c 2d 3d_fullres
```

### Ensemble Evaluation

Evaluate ensemble of multiple configurations:

```bash
# Train configurations
nnUNetv2_train 001 2d all
nnUNetv2_train 001 3d_fullres all

# Create ensemble predictions
nnUNetv2_ensemble -i \
    nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__2d/cv_niftis_raw \
    nnUNet_results/Dataset001_Name/nnUNetTrainer__nnUNetPlans__3d_fullres/cv_niftis_raw \
    -o ensemble_predictions

# Evaluate ensemble
nnUNetv2_evaluate_folder \
    nnUNet_raw/Dataset001_Name/labelsTr \
    ensemble_predictions
```

## Advanced Usage

### Custom Metrics

Add custom metrics by extending evaluation functions:

```python
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
import numpy as np

def my_custom_metric(pred, gt):
    """Example: Compute volume difference."""
    return np.abs(pred.sum() - gt.sum()) / gt.sum()

# Modify evaluation to include custom metric
# (requires editing evaluate_predictions.py or writing custom script)
```

### Per-Class Evaluation

Analyze performance per class:

```python
import json

# Load results
with open('summary.json', 'r') as f:
    results = json.load(f)

# Per-class mean Dice
mean_dice = results['mean']['Dice']
print(f"Class 0 (background): {mean_dice[0]:.3f}")
print(f"Class 1 (tumor): {mean_dice[1]:.3f}")
print(f"Class 2 (edema): {mean_dice[2]:.3f}")
```

### Statistical Significance Testing

Compare two configurations statistically:

```python
from scipy.stats import wilcoxon
import json

# Load results from both configurations
with open('config1/summary.json', 'r') as f:
    results1 = json.load(f)

with open('config2/summary.json', 'r') as f:
    results2 = json.load(f)

# Extract per-case Dice for class 1
dice1 = [results1[case]['Dice'][1] for case in results1 if case != 'mean']
dice2 = [results2[case]['Dice'][1] for case in results2 if case != 'mean']

# Wilcoxon signed-rank test
stat, p_value = wilcoxon(dice1, dice2)
print(f"p-value: {p_value:.4f}")
```

## Evaluation Tips

### Validation vs Test Evaluation

**Validation** (during training):
- Used to select best checkpoint
- Cross-validation provides unbiased estimate
- Use `nnUNetv2_accumulate_crossval_results`

**Test** (after training):
- Independent test set (never seen during training)
- Use `nnUNetv2_predict` + `nnUNetv2_evaluate_folder`

### Ensemble Best Practices

**When to ensemble**:
- Multiple configurations perform similarly
- Marginal gain acceptable (slower inference)

**Common ensembles**:
- 2D + 3D full-res (different perspectives)
- All 5 folds of same configuration (reduces variance)
- Multiple architectures (e.g., U-Net + KiU-Net)

**How to ensemble**:
```bash
nnUNetv2_ensemble -i PRED_FOLDER1 PRED_FOLDER2 ... -o OUTPUT
```

Averaging is done on softmax probabilities, then argmax.

### Handling Missing Ground Truth

If some test cases lack ground truth:

```bash
# Evaluate only on cases with ground truth
nnUNetv2_evaluate_folder GT_FOLDER PRED_FOLDER
# Will automatically skip cases not in GT_FOLDER
```

### Multi-Class vs Binary Evaluation

**Multi-class**:
- Metrics computed per class, then averaged
- Background (class 0) typically excluded from mean

**Binary** (single foreground class):
- Only one class evaluated (class 1)

## Output Files

### summary.json

Contains comprehensive metrics:

```json
{
  "case_001": {
    "Dice": [1.0, 0.85, 0.78],
    "Hausdorff95": [0.0, 2.3, 3.1],
    "SurfaceDice": [1.0, 0.82, 0.74],
    "Precision": [1.0, 0.87, 0.81],
    "Recall": [1.0, 0.83, 0.75]
  },
  ...
  "mean": {
    "Dice": [1.0, 0.83, 0.75],
    ...
  }
}
```

### Cross-Validation Predictions

Located in `cv_niftis_raw/`:
- Contains predictions for all training cases
- Each case predicted by fold it wasn't trained on
- Can be used for further analysis or postprocessing determination

## See Also

- [Metrics Module](../metrics/) - Metric computation implementations
- [Inference](../inference/) - How to generate predictions
- [Postprocessing](../postprocessing/) - Postprocessing evaluation and determination
- [Ensembling](../ensembling/) - Ensemble prediction utilities
