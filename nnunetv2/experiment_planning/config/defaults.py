"""
Default configuration values for experiment planning.
"""

from nnunetv2.configuration import default_num_processes, ANISO_THRESHOLD

# Default number of processes for different operations
DEFAULT_FINGERPRINT_PROCESSES = default_num_processes
DEFAULT_PREPROCESSING_PROCESSES = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}

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

# Anisotropy threshold
DEFAULT_ANISO_THRESHOLD = ANISO_THRESHOLD

# GPU memory defaults
DEFAULT_GPU_MEMORY_TARGET_GB = 8

# Preprocessing defaults
DEFAULT_SPACING_PERCENTILE = 50.0
DEFAULT_TARGET_SPACING_PERCENTILE = 50.0