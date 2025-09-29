import numpy as np
import torch
from torch import autocast, distributed as dist

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs


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


def on_train_epoch_end(trainer_instance, train_outputs):
    """
    Handle end of training epoch.

    Args:
        trainer_instance: The nnUNetTrainer instance
        train_outputs: List of training step outputs
    """
    outputs = collate_outputs(train_outputs)

    if trainer_instance.is_ddp:
        losses_tr = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(losses_tr, outputs['loss'])
        loss_here = np.vstack(losses_tr).mean()
    else:
        loss_here = np.mean(outputs['loss'])

    trainer_instance.logger.log('train_losses', loss_here, trainer_instance.current_epoch)


def on_validation_epoch_end(trainer_instance, val_outputs):
    """
    Handle end of validation epoch.

    Args:
        trainer_instance: The nnUNetTrainer instance
        val_outputs: List of validation step outputs
    """
    outputs_collated = collate_outputs(val_outputs)
    tp = np.sum(outputs_collated['tp_hard'], 0)
    fp = np.sum(outputs_collated['fp_hard'], 0)
    fn = np.sum(outputs_collated['fn_hard'], 0)

    if trainer_instance.is_ddp:
        world_size = dist.get_world_size()

        tps = [None for _ in range(world_size)]
        dist.all_gather_object(tps, tp)
        tp = np.vstack([i[None] for i in tps]).sum(0)

        fps = [None for _ in range(world_size)]
        dist.all_gather_object(fps, fp)
        fp = np.vstack([i[None] for i in fps]).sum(0)

        fns = [None for _ in range(world_size)]
        dist.all_gather_object(fns, fn)
        fn = np.vstack([i[None] for i in fns]).sum(0)

        losses_val = [None for _ in range(world_size)]
        dist.all_gather_object(losses_val, outputs_collated['loss'])
        loss_here = np.vstack(losses_val).mean()
    else:
        loss_here = np.mean(outputs_collated['loss'])

    global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
    mean_fg_dice = np.nanmean(global_dc_per_class)
    trainer_instance.logger.log('mean_fg_dice', mean_fg_dice, trainer_instance.current_epoch)
    trainer_instance.logger.log('dice_per_class_or_region', global_dc_per_class, trainer_instance.current_epoch)
    trainer_instance.logger.log('val_losses', loss_here, trainer_instance.current_epoch)