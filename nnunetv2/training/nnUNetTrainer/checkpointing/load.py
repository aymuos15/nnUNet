from typing import Union

import torch
from torch._dynamo import OptimizedModule


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