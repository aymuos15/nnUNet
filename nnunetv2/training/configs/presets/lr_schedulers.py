"""Learning rate scheduler configuration presets."""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.configs import TrainerConfig, register_config


def cosine_annealing_optimizer_builder(trainer_instance):
    """Build SGD optimizer with CosineAnnealingLR scheduler."""
    optimizer = torch.optim.SGD(
        trainer_instance.network.parameters(),
        trainer_instance.initial_lr,
        weight_decay=trainer_instance.weight_decay,
        momentum=0.99,
        nesterov=True
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=trainer_instance.num_epochs)
    return optimizer, lr_scheduler


# Cosine annealing LR scheduler
COSINE_ANNEALING_CONFIG = TrainerConfig(
    name="cosine_annealing",
    description="Use cosine annealing LR scheduler instead of PolyLR",
    optimizer_builder=cosine_annealing_optimizer_builder
)
register_config(COSINE_ANNEALING_CONFIG)