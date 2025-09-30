from .base import (
    TrainerConfig,
    register_config,
    get_config,
    list_configs,
    BASE_CONFIG
)

# Import presets to register all configs
from . import presets

__all__ = [
    'TrainerConfig',
    'register_config',
    'get_config',
    'list_configs',
    'BASE_CONFIG'
]