"""
Validation step methods extracted from nnUNetTrainer.py
Contains validation_step, on_validation_epoch_start, and on_validation_epoch_end methods.
"""

from typing import List
import numpy as np
import torch
from torch import autocast, distributed as dist

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context


def on_validation_epoch_start(self):
    """
    Set the network to evaluation mode.
    """
    self.network.eval()


def validation_step(self, batch: dict) -> dict:
    """
    Perform a single validation step.

    Args:
        batch: Dictionary containing 'data' and 'target' keys

    Returns:
        Dictionary with validation metrics including loss, tp_hard, fp_hard, fn_hard
    """
    data = batch['data']
    target = batch['target']

    data = data.to(self.device, non_blocking=True)
    if isinstance(target, list):
        target = [i.to(self.device, non_blocking=True) for i in target]
    else:
        target = target.to(self.device, non_blocking=True)

    # Autocast can be annoying
    # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
    # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
    # So autocast will only be active if we have a cuda device.
    with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
        output = self.network(data)
        del data
        l = self.loss(output, target)

    # we only need the output with the highest output resolution (if DS enabled)
    if self.enable_deep_supervision:
        output = output[0]
        target = target[0]

    # the following is needed for online evaluation. Fake dice (green line)
    axes = [0] + list(range(2, output.ndim))

    if self.label_manager.has_regions:
        predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
    else:
        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

    if self.label_manager.has_ignore_label:
        if not self.label_manager.has_regions:
            mask = (target != self.label_manager.ignore_label).float()
            # CAREFUL that you don't rely on target after this line!
            target[target == self.label_manager.ignore_label] = 0
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
    if not self.label_manager.has_regions:
        # if we train with regions all segmentation heads predict some kind of foreground. In conventional
        # (softmax training) there needs tobe one output for the background. We are not interested in the
        # background Dice
        # [1:] in order to remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

    return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


def on_validation_epoch_end(self, val_outputs: List[dict]):
    """
    Process validation outputs at the end of an epoch.
    Computes metrics and logs them.

    Args:
        val_outputs: List of validation step outputs
    """
    outputs_collated = collate_outputs(val_outputs)
    tp = np.sum(outputs_collated['tp_hard'], 0)
    fp = np.sum(outputs_collated['fp_hard'], 0)
    fn = np.sum(outputs_collated['fn_hard'], 0)

    if self.is_ddp:
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
    self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
    self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
    self.logger.log('val_losses', loss_here, self.current_epoch)