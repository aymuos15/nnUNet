"""Optimizer configuration presets."""

import torch
from torch.optim import Adam, AdamW

from nnunetv2.training.configs import TrainerConfig, register_config
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


def adam_optimizer_builder(trainer_instance):
    """Build AdamW optimizer with PolyLR scheduler."""
    optimizer = AdamW(
        trainer_instance.network.parameters(),
        lr=trainer_instance.initial_lr,
        weight_decay=trainer_instance.weight_decay,
        amsgrad=True
    )
    lr_scheduler = PolyLRScheduler(optimizer, trainer_instance.initial_lr, trainer_instance.num_epochs)
    return optimizer, lr_scheduler


def vanilla_adam_optimizer_builder(trainer_instance):
    """Build vanilla Adam optimizer with PolyLR scheduler."""
    optimizer = Adam(
        trainer_instance.network.parameters(),
        lr=trainer_instance.initial_lr,
        weight_decay=trainer_instance.weight_decay
    )
    lr_scheduler = PolyLRScheduler(optimizer, trainer_instance.initial_lr, trainer_instance.num_epochs)
    return optimizer, lr_scheduler


# AdamW optimizer
ADAM_CONFIG = TrainerConfig(
    name="adam",
    description="Use AdamW optimizer instead of SGD",
    optimizer_builder=adam_optimizer_builder
)
register_config(ADAM_CONFIG)


# Vanilla Adam optimizer
VANILLA_ADAM_CONFIG = TrainerConfig(
    name="vanilla_adam",
    description="Use vanilla Adam optimizer instead of SGD",
    optimizer_builder=vanilla_adam_optimizer_builder
)
register_config(VANILLA_ADAM_CONFIG)


# Adam with 1e-3 learning rate
ADAM_1E3_CONFIG = TrainerConfig(
    name="adam_1e3",
    description="Use AdamW optimizer with 1e-3 learning rate",
    initial_lr=1e-3,
    optimizer_builder=adam_optimizer_builder
)
register_config(ADAM_1E3_CONFIG)


# Adam with 3e-4 learning rate (Karpathy's constant)
ADAM_3E4_CONFIG = TrainerConfig(
    name="adam_3e4",
    description="Use AdamW optimizer with 3e-4 learning rate",
    initial_lr=3e-4,
    optimizer_builder=adam_optimizer_builder
)
register_config(ADAM_3E4_CONFIG)


# Vanilla Adam with 1e-3 learning rate
VANILLA_ADAM_1E3_CONFIG = TrainerConfig(
    name="vanilla_adam_1e3",
    description="Use vanilla Adam optimizer with 1e-3 learning rate",
    initial_lr=1e-3,
    optimizer_builder=vanilla_adam_optimizer_builder
)
register_config(VANILLA_ADAM_1E3_CONFIG)


# Vanilla Adam with 3e-4 learning rate
VANILLA_ADAM_3E4_CONFIG = TrainerConfig(
    name="vanilla_adam_3e4",
    description="Use vanilla Adam optimizer with 3e-4 learning rate",
    initial_lr=3e-4,
    optimizer_builder=vanilla_adam_optimizer_builder
)
register_config(VANILLA_ADAM_3E4_CONFIG)