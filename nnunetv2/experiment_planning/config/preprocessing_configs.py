"""
Preprocessing configuration presets and utilities.
"""

from typing import Dict, List, Union, Tuple

# Standard preprocessing configurations
PREPROCESSING_CONFIGS = {
    '2d': {
        'description': '2D U-Net configuration',
        'spacing_strategy': 'median',
        'normalization': 'instance',
        'data_augmentation': True,
        'default_num_processes': 8,
    },
    '3d_fullres': {
        'description': '3D full resolution U-Net configuration',
        'spacing_strategy': 'median',
        'normalization': 'instance',
        'data_augmentation': True,
        'default_num_processes': 4,
    },
    '3d_lowres': {
        'description': '3D low resolution U-Net configuration (for cascade)',
        'spacing_strategy': 'median_with_downsampling',
        'normalization': 'instance',
        'data_augmentation': True,
        'default_num_processes': 8,
    },
    '3d_cascade_fullres': {
        'description': '3D cascade full resolution (uses 3d_fullres data)',
        'uses_data_from': '3d_fullres',
        'normalization': 'instance',
        'data_augmentation': True,
        'default_num_processes': 4,
    }
}

# Spacing strategies
SPACING_STRATEGIES = {
    'median': 'Use median spacing across all training cases',
    'mean': 'Use mean spacing across all training cases',
    'percentile_10': 'Use 10th percentile spacing',
    'percentile_50': 'Use 50th percentile spacing (same as median)',
    'percentile_90': 'Use 90th percentile spacing',
    'original': 'Keep original spacing (no resampling)',
    'custom': 'Use custom specified spacing',
}

# Normalization schemes
NORMALIZATION_SCHEMES = {
    'instance': 'Instance normalization (recommended)',
    'batch': 'Batch normalization',
    'group': 'Group normalization',
    'layer': 'Layer normalization',
    'none': 'No normalization',
}

# Data augmentation presets
DATA_AUGMENTATION_PRESETS = {
    'standard': {
        'rotation': True,
        'scaling': True,
        'elastic_deformation': True,
        'gaussian_noise': True,
        'gaussian_blur': True,
        'brightness': True,
        'contrast': True,
        'simulation_of_low_resolution': True,
        'gamma': True,
        'mirroring': True,
    },
    'light': {
        'rotation': True,
        'scaling': True,
        'mirroring': True,
        'gaussian_noise': False,
        'gaussian_blur': False,
        'brightness': True,
        'contrast': True,
        'elastic_deformation': False,
        'simulation_of_low_resolution': False,
        'gamma': False,
    },
    'heavy': {
        'rotation': True,
        'scaling': True,
        'elastic_deformation': True,
        'gaussian_noise': True,
        'gaussian_blur': True,
        'brightness': True,
        'contrast': True,
        'simulation_of_low_resolution': True,
        'gamma': True,
        'mirroring': True,
        'cutout': True,
        'color_jitter': True,
    },
    'none': {}
}


def get_configuration_info(config_name: str) -> Dict:
    """
    Get information about a preprocessing configuration.

    Args:
        config_name: Name of the configuration

    Returns:
        Dictionary with configuration information
    """
    return PREPROCESSING_CONFIGS.get(config_name, {})


def get_available_configurations() -> List[str]:
    """Get list of available preprocessing configurations."""
    return list(PREPROCESSING_CONFIGS.keys())


def get_configurations_for_dimension(dimension: str) -> List[str]:
    """
    Get configurations that match the specified dimension.

    Args:
        dimension: '2d' or '3d'

    Returns:
        List of matching configuration names
    """
    if dimension == '2d':
        return ['2d']
    elif dimension == '3d':
        return ['3d_fullres', '3d_lowres', '3d_cascade_fullres']
    else:
        return []


def validate_configuration_combination(configs: List[str]) -> bool:
    """
    Validate that a combination of configurations makes sense.

    Args:
        configs: List of configuration names

    Returns:
        True if combination is valid, False otherwise
    """
    # Check if all configurations exist
    available = get_available_configurations()
    if not all(config in available for config in configs):
        return False

    # Check if cascade is used with appropriate fullres config
    if '3d_cascade_fullres' in configs and '3d_fullres' not in configs:
        return False

    return True