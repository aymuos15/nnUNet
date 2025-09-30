# Import all preset configs to register them
from . import mirroring
from . import epochs
from . import optimizers
from . import losses
from . import data_augmentation
from . import lr_schedulers

__all__ = [
    'mirroring',
    'epochs',
    'optimizers',
    'losses',
    'data_augmentation',
    'lr_schedulers'
]