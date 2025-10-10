"""
Training and validation step execution.

This module handles the core step logic that is executed repeatedly during training:
- train_step: Single training step (forward, backward, optimization)
- validation_step: Single validation step (forward, metrics computation)
"""

import numpy as np
import torch
from torch import autocast

from nnunetv2.utilities.core.helpers import dummy_context
from nnunetv2.metrics.dice import get_tp_fp_fn_tn


def train_step(trainer_instance, batch: dict) -> dict:
    """
    Perform a single training step.

    Args:
        trainer_instance: The nnUNetTrainer instance
        batch: Input batch containing 'data' and 'target'

    Returns:
        Dictionary containing loss value
    """
    data = batch['data']
    target = batch['target']

    data = data.to(trainer_instance.device, non_blocking=True)
    if isinstance(target, list):
        target = [i.to(trainer_instance.device, non_blocking=True) for i in target]
    else:
        target = target.to(trainer_instance.device, non_blocking=True)

    trainer_instance.optimizer.zero_grad(set_to_none=True)
    # Autocast can be annoying
    # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
    # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
    # So autocast will only be active if we have a cuda device.
    with autocast(trainer_instance.device.type, enabled=True) if trainer_instance.device.type == 'cuda' else dummy_context():
        output = trainer_instance.network(data)
        # del data
        l = trainer_instance.loss(output, target)

    if trainer_instance.grad_scaler is not None:
        trainer_instance.grad_scaler.scale(l).backward()
        trainer_instance.grad_scaler.unscale_(trainer_instance.optimizer)
        torch.nn.utils.clip_grad_norm_(trainer_instance.network.parameters(), 12)
        trainer_instance.grad_scaler.step(trainer_instance.optimizer)
        trainer_instance.grad_scaler.update()
    else:
        l.backward()
        torch.nn.utils.clip_grad_norm_(trainer_instance.network.parameters(), 12)
        trainer_instance.optimizer.step()
    return {'loss': l.detach().cpu().numpy()}


def validation_step(trainer_instance, batch: dict) -> dict:
    """
    Perform a single validation step.

    Args:
        trainer_instance: The nnUNetTrainer instance
        batch: Dictionary containing 'data' and 'target' keys

    Returns:
        Dictionary with validation metrics including loss, tp_hard, fp_hard, fn_hard
    """
    data = batch['data']
    target = batch['target']

    data = data.to(trainer_instance.device, non_blocking=True)
    if isinstance(target, list):
        target = [i.to(trainer_instance.device, non_blocking=True) for i in target]
    else:
        target = target.to(trainer_instance.device, non_blocking=True)

    # Autocast can be annoying
    # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
    # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
    # So autocast will only be active if we have a cuda device.
    with autocast(trainer_instance.device.type, enabled=True) if trainer_instance.device.type == 'cuda' else dummy_context():
        output = trainer_instance.network(data)
        del data
        l = trainer_instance.loss(output, target)

    # we only need the output with the highest output resolution (if DS enabled)
    if trainer_instance.enable_deep_supervision:
        output = output[0]
        target = target[0]

    # the following is needed for online evaluation. Fake dice (green line)
    axes = [0] + list(range(2, output.ndim))

    if trainer_instance.label_manager.has_regions:
        predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
    else:
        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

    if trainer_instance.label_manager.has_ignore_label:
        if not trainer_instance.label_manager.has_regions:
            mask = (target != trainer_instance.label_manager.ignore_label).float()
            # CAREFUL that you don't rely on target after this line!
            target[target == trainer_instance.label_manager.ignore_label] = 0
        else:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = 1 - target[:, -1:]
            # CAREFUL that you don't rely on target after this line!
            target = target[:, :-1]
    else:
        mask = None

    tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

    tp_hard = tp.detach().cpu().numpy()
    fp_hard = fp.detach().cpu().numpy()
    fn_hard = fn.detach().cpu().numpy()
    if not trainer_instance.label_manager.has_regions:
        # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        # (softmax training) there needs tobe one output for the background. We are not interested in the
        # background Dice
        # [1:] in order to remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

    return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
