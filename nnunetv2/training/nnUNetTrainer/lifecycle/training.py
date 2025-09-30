import torch

from .hooks import (on_train_start, on_train_end, on_epoch_start, on_epoch_end,
                    on_train_epoch_start, on_validation_epoch_start)
from ..training.step import train_step, on_train_epoch_end, on_validation_epoch_end
from ..validation.step import validation_step


def run_training(trainer_instance):
    """
    Main training orchestration loop.

    Handles the complete training lifecycle:
    - Training start setup
    - Epoch-by-epoch training and validation
    - Training completion cleanup
    """
    on_train_start(trainer_instance)

    for epoch in range(trainer_instance.current_epoch, trainer_instance.num_epochs):
        on_epoch_start(trainer_instance)

        on_train_epoch_start(trainer_instance)
        train_outputs = []
        for batch_id in range(trainer_instance.num_iterations_per_epoch):
            train_outputs.append(train_step(trainer_instance, next(trainer_instance.dataloader_train)))
        on_train_epoch_end(trainer_instance, train_outputs)

        with torch.no_grad():
            on_validation_epoch_start(trainer_instance)
            val_outputs = []
            for batch_id in range(trainer_instance.num_val_iterations_per_epoch):
                val_outputs.append(validation_step(trainer_instance, next(trainer_instance.dataloader_val)))
            on_validation_epoch_end(trainer_instance, val_outputs)

        on_epoch_end(trainer_instance)

    on_train_end(trainer_instance)