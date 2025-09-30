"""
Default configuration values for experiment planning.
"""

import os
from nnunetv2.training.runtime_utils.default_n_proc_DA import get_allowed_n_proc_DA

# Default number of processes for different operations
DEFAULT_NUM_PROCESSES = 8 if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])
DEFAULT_FINGERPRINT_PROCESSES = DEFAULT_NUM_PROCESSES
DEFAULT_PREPROCESSING_PROCESSES = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
DEFAULT_DATA_AUGMENTATION_PROCESSES = get_allowed_n_proc_DA()

# Default class names
DEFAULT_FINGERPRINT_EXTRACTOR = 'DatasetFingerprintExtractor'
DEFAULT_EXPERIMENT_PLANNER = 'ExperimentPlanner'
DEFAULT_PREPROCESSOR = 'DefaultPreprocessor'
DEFAULT_PLANS_NAME = 'nnUNetPlans'

# Default configurations for preprocessing
DEFAULT_CONFIGURATIONS = ['2d', '3d_fullres', '3d_lowres']

# Network architecture defaults
DEFAULT_UNET_BASE_NUM_FEATURES = 32
DEFAULT_MAX_NUM_FEATURES = 320
DEFAULT_BATCH_SIZE = 2

# Dataset fingerprint defaults
DEFAULT_NUM_FOREGROUND_VOXELS_FOR_INTENSITYSTATS = 10e7
DEFAULT_INTENSITY_SAMPLES_PER_CASE = 10000

# Anisotropy threshold - determines when a sample is considered anisotropic
# (3 means that the spacing in the low resolution axis must be 3x as large as the next largest spacing)
DEFAULT_ANISO_THRESHOLD = 3

# GPU memory defaults
DEFAULT_GPU_MEMORY_TARGET_GB = 8

# Preprocessing defaults
DEFAULT_SPACING_PERCENTILE = 50.0
DEFAULT_TARGET_SPACING_PERCENTILE = 50.0