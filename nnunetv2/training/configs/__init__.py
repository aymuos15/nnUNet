# Import base classes first to avoid circular imports
from .base import TrainerConfig, register_config, get_config

# Import all preset configs to register them
from . import mirroring
from . import epochs
from . import optimizers
from . import losses
from . import data_augmentation
from . import lr_schedulers

__all__ = [
    'TrainerConfig',
    'register_config',
    'get_config',
    'mirroring',
    'epochs',
    'optimizers',
    'losses',
    'data_augmentation',
    'lr_schedulers'
]