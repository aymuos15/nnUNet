"""
Checkpoint saving and loading for training state management.

This module handles:
- Saving checkpoints with full training state (network weights, optimizer, metrics)
- Loading checkpoints and restoring training state
- Handling DDP and compiled models correctly
"""

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

            # Prepare init_args for checkpointing - exclude trainer_config object as it may contain unpicklable callables
            init_args = {}
            for k, v in trainer.my_init_kwargs.items():
                # Convert device to string for pickling
                if k == 'device' and isinstance(v, torch.device):
                    init_args[k] = str(v)
                # Skip trainer_config as it may contain unpicklable callables
                elif k == 'trainer_config':
                    continue
                else:
                    init_args[k] = v

            trainer_config_name = None
            # Check both init_args and trainer.trainer_config for the config
            if 'trainer_config' in trainer.my_init_kwargs and trainer.my_init_kwargs['trainer_config'] is not None:
                trainer_config_name = trainer.my_init_kwargs['trainer_config'].name
            elif hasattr(trainer, 'trainer_config') and trainer.trainer_config is not None:
                trainer_config_name = trainer.trainer_config.name

            checkpoint = {
                'network_weights': mod.state_dict(),
                'optimizer_state': trainer.optimizer.state_dict(),
                'grad_scaler_state': trainer.grad_scaler.state_dict() if trainer.grad_scaler is not None else None,
                'logging': trainer.logger.get_checkpoint(),
                '_best_ema': trainer._best_ema,
                'current_epoch': trainer.current_epoch + 1,
                'init_args': init_args,
                'trainer_config_name': trainer_config_name,
                'trainer_name': trainer.__class__.__name__,
                'inference_allowed_mirroring_axes': trainer.inference_allowed_mirroring_axes,
            }
            torch.save(checkpoint, filename)
        else:
            trainer.print_to_log_file('No checkpoint written, checkpointing is disabled')


def load_checkpoint(trainer, filename_or_checkpoint: Union[dict, str]) -> None:
    """
    Load a checkpoint and restore the training state.

    Args:
        trainer: The nnUNetTrainer instance
        filename_or_checkpoint: Path to checkpoint file or checkpoint dictionary
    """
    if not trainer.was_initialized:
        trainer.initialize()

    if isinstance(filename_or_checkpoint, str):
        checkpoint = torch.load(filename_or_checkpoint, map_location=trainer.device, weights_only=False)
    else:
        checkpoint = filename_or_checkpoint

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    new_state_dict = {}
    for k, value in checkpoint['network_weights'].items():
        key = k
        if key not in trainer.network.state_dict().keys() and key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    trainer.my_init_kwargs = checkpoint['init_args']

    # Convert device string back to torch.device if needed
    if 'device' in trainer.my_init_kwargs and isinstance(trainer.my_init_kwargs['device'], str):
        trainer.my_init_kwargs['device'] = torch.device(trainer.my_init_kwargs['device'])

    # Restore trainer_config from name if it was saved
    if 'trainer_config_name' in checkpoint and checkpoint['trainer_config_name'] is not None:
        from nnunetv2.training.configs import get_config
        trainer.my_init_kwargs['trainer_config'] = get_config(checkpoint['trainer_config_name'])

    trainer.current_epoch = checkpoint['current_epoch']
    trainer.logger.load_checkpoint(checkpoint['logging'])
    trainer._best_ema = checkpoint['_best_ema']
    trainer.inference_allowed_mirroring_axes = checkpoint[
        'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else trainer.inference_allowed_mirroring_axes

    # messing with state dict naming schemes. Facepalm.
    if trainer.is_ddp:
        if isinstance(trainer.network.module, OptimizedModule):
            trainer.network.module._orig_mod.load_state_dict(new_state_dict)
        else:
            trainer.network.module.load_state_dict(new_state_dict)
    else:
        if isinstance(trainer.network, OptimizedModule):
            trainer.network._orig_mod.load_state_dict(new_state_dict)
        else:
            trainer.network.load_state_dict(new_state_dict)
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
    if trainer.grad_scaler is not None:
        if checkpoint['grad_scaler_state'] is not None:
            trainer.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])