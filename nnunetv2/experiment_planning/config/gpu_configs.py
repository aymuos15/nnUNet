"""
GPU memory configuration presets for different hardware setups.
"""

# Standard GPU memory configurations (in GB)
GPU_MEMORY_CONFIGS = {
    'low': 4.0,      # For older or smaller GPUs
    'medium': 8.0,   # Standard configuration
    'high': 11.0,    # For RTX 2080 Ti, RTX 3080
    'very_high': 16.0,  # For RTX 3090, RTX 4090
    'ultra': 24.0,   # For RTX 3090, A6000, etc.
    'workstation': 48.0,  # For A100, H100
}

# Recommended batch sizes for different memory configurations
BATCH_SIZE_RECOMMENDATIONS = {
    'low': {'2d': 4, '3d_fullres': 1, '3d_lowres': 2},
    'medium': {'2d': 8, '3d_fullres': 2, '3d_lowres': 4},
    'high': {'2d': 12, '3d_fullres': 2, '3d_lowres': 6},
    'very_high': {'2d': 16, '3d_fullres': 4, '3d_lowres': 8},
    'ultra': {'2d': 20, '3d_fullres': 6, '3d_lowres': 12},
    'workstation': {'2d': 24, '3d_fullres': 8, '3d_lowres': 16},
}

# Patch size recommendations for different memory configurations
PATCH_SIZE_RECOMMENDATIONS = {
    'low': {'2d': (256, 256), '3d_fullres': (64, 64, 64), '3d_lowres': (96, 96, 96)},
    'medium': {'2d': (320, 320), '3d_fullres': (96, 96, 96), '3d_lowres': (128, 128, 128)},
    'high': {'2d': (384, 384), '3d_fullres': (128, 128, 128), '3d_lowres': (160, 160, 160)},
    'very_high': {'2d': (448, 448), '3d_fullres': (160, 160, 160), '3d_lowres': (192, 192, 192)},
    'ultra': {'2d': (512, 512), '3d_fullres': (192, 192, 192), '3d_lowres': (224, 224, 224)},
    'workstation': {'2d': (576, 576), '3d_fullres': (224, 224, 224), '3d_lowres': (256, 256, 256)},
}


def get_gpu_config(memory_gb: float) -> str:
    """
    Get the appropriate GPU configuration name for given memory.

    Args:
        memory_gb: Available GPU memory in GB

    Returns:
        Configuration name
    """
    if memory_gb <= 4:
        return 'low'
    elif memory_gb <= 8:
        return 'medium'
    elif memory_gb <= 11:
        return 'high'
    elif memory_gb <= 16:
        return 'very_high'
    elif memory_gb <= 24:
        return 'ultra'
    else:
        return 'workstation'


def get_recommended_batch_size(memory_gb: float, configuration: str) -> int:
    """
    Get recommended batch size for given memory and configuration.

    Args:
        memory_gb: Available GPU memory in GB
        configuration: Training configuration ('2d', '3d_fullres', '3d_lowres')

    Returns:
        Recommended batch size
    """
    config_name = get_gpu_config(memory_gb)
    return BATCH_SIZE_RECOMMENDATIONS[config_name].get(configuration, 2)


def get_recommended_patch_size(memory_gb: float, configuration: str) -> tuple:
    """
    Get recommended patch size for given memory and configuration.

    Args:
        memory_gb: Available GPU memory in GB
        configuration: Training configuration ('2d', '3d_fullres', '3d_lowres')

    Returns:
        Recommended patch size as tuple
    """
    config_name = get_gpu_config(memory_gb)
    return PATCH_SIZE_RECOMMENDATIONS[config_name].get(configuration, (128, 128, 128))