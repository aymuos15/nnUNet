# Import from new modular structure
from nnunetv2.experiment_planning.core.fingerprint_extraction import extract_fingerprints, extract_fingerprint_dataset
from nnunetv2.experiment_planning.core.experiment_planning import plan_experiments, plan_experiment_dataset
from nnunetv2.experiment_planning.core.preprocessing_coordination import preprocess, preprocess_dataset


# All functions are now imported from the new modular structure
# This file serves as a backward-compatible API layer

# Re-export all the functions for backward compatibility
__all__ = [
    'extract_fingerprints',
    'extract_fingerprint_dataset',
    'plan_experiments',
    'plan_experiment_dataset',
    'preprocess',
    'preprocess_dataset'
]
