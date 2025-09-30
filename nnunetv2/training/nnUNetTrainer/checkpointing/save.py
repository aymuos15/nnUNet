import os
from typing import Union

import torch
from torch._dynamo import OptimizedModule


def save_checkpoint(trainer, filename: str) -> None:
    """
    Save a checkpoint containing the current state of training.

    Args:
        trainer: The nnUNetTrainer instance
        filename: Path where to save the checkpoint
    """
    if trainer.local_rank == 0:
        if not trainer.disable_checkpointing:
            if trainer.is_ddp:
                mod = trainer.network.module
            else:
                mod = trainer.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod

            checkpoint = {
                'network_weights': mod.state_dict(),
                'optimizer_state': trainer.optimizer.state_dict(),
                'grad_scaler_state': trainer.grad_scaler.state_dict() if trainer.grad_scaler is not None else None,
                'logging': trainer.logger.get_checkpoint(),
                '_best_ema': trainer._best_ema,
                'current_epoch': trainer.current_epoch + 1,
                'init_args': trainer.my_init_kwargs,
                'trainer_name': trainer.__class__.__name__,
                'inference_allowed_mirroring_axes': trainer.inference_allowed_mirroring_axes,
            }
            torch.save(checkpoint, filename)
        else:
            trainer.print_to_log_file('No checkpoint written, checkpointing is disabled')