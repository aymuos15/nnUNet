# Metrics

This module provides implementations of segmentation metrics for evaluating prediction quality.

## Overview

Metrics quantify segmentation performance by comparing predictions against ground truth labels.

## Directory Structure

```
metrics/
├── dice.py      # Dice metric implementations
└── __init__.py  # Public API exports
```

## Available Metrics

### Dice Coefficient

Measures overlap between prediction and ground truth:

```
Dice = 2 * |A ∩ B| / (|A| + |B|)
```

- **Range**: 0 (no overlap) to 1 (perfect overlap)
- **Use**: Primary metric in medical image segmentation
- **Interpretation**: 
  - >0.9 = Excellent
  - >0.8 = Good  
  - >0.7 = Acceptable

### Hausdorff Distance (95th percentile)

Measures maximum surface distance:

```
HD95 = 95th percentile of surface distances
```

- **Range**: 0 (perfect) to ∞
- **Units**: mm or voxels
- **Use**: Boundary accuracy
- **Interpretation**: Lower is better

### Surface Dice

Dice computed on surfaces with tolerance:

- **Tolerance**: Typically 1mm
- **Use**: Boundary-focused metric
- **Interpretation**: More sensitive to boundary errors than volumetric Dice

### Precision & Recall

**Precision** (Positive Predictive Value):
```
Precision = TP / (TP + FP)
```

**Recall** (Sensitivity):
```
Recall = TP / (TP + FN)
```

- **Use**: Understand type of errors (FP vs FN)

## Usage

Metrics are primarily used via the evaluation module:

```bash
nnUNetv2_evaluate_folder GROUND_TRUTH_FOLDER PREDICTIONS_FOLDER
```

See [Evaluation Module](../evaluation/) for details.

## Programmatic API

```python
from nnunetv2.metrics import get_tp_fp_fn_tn
import torch

# Compute TP, FP, FN, TN for Dice calculation
net_output = torch.softmax(model_output, dim=1)  # Your model output
ground_truth = labels  # Your ground truth labels

tp, fp, fn, tn = get_tp_fp_fn_tn(net_output, ground_truth)

# Calculate Dice from TP, FP, FN
dice = (2 * tp) / (2 * tp + fp + fn)
```

## Custom Metrics

To add a custom metric:

```python
import numpy as np

def my_custom_metric(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute custom metric.
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        
    Returns:
        Metric value (float)
    """
    # Your implementation
    return metric_value
```

Then integrate into evaluation pipeline by modifying `evaluation/evaluate_predictions.py`.

## See Also

- [Evaluation Module](../evaluation/) - Using metrics to evaluate predictions
- Training module uses Dice loss (differentiable version of Dice metric)
