import os
import sys
from time import time

import numpy as np
from torch import distributed as dist
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, isfile

from nnunetv2.utilities.training_runtime.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.core.helpers import empty_cache
from ..data.loaders import get_dataloaders
from ..initialization.config import ensure_output_folder_exists, copy_plans_and_dataset_json
from ..initialization.network import set_deep_supervision_enabled, plot_network_architecture


def on_train_start(trainer_instance):
    """
    Called at the beginning of training.

    Handles:
    - Trainer initialization
    - Data loader setup
    - Output folder preparation
    - Dataset unpacking
    - Network architecture plotting
    - Debug information saving
    """
    if not trainer_instance.was_initialized:
        trainer_instance.initialize()

    # dataloaders must be instantiated here (instead of __init__) because they need access to the training data
    # which may not be present when doing inference
    trainer_instance.dataloader_train, trainer_instance.dataloader_val = get_dataloaders(trainer_instance)

    ensure_output_folder_exists(trainer_instance)

    # make sure deep supervision is on in the network
    set_deep_supervision_enabled(trainer_instance, trainer_instance.enable_deep_supervision)

    trainer_instance.print_plans()
    empty_cache(trainer_instance.device)

    # maybe unpack
    if trainer_instance.local_rank == 0:
        trainer_instance.dataset_class.unpack_dataset(
            trainer_instance.preprocessed_dataset_folder,
            overwrite_existing=False,
            num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
            verify=True)

    if trainer_instance.is_ddp:
        dist.barrier()

    # copy plans and dataset.json so that they can be used for restoring everything we need for inference
    copy_plans_and_dataset_json(trainer_instance)

    # produces a pdf in output folder
    plot_network_architecture(trainer_instance)

    trainer_instance._save_debug_information()


def on_train_end(trainer_instance):
    """
    Called at the end of training.

    Handles:
    - Final checkpoint saving
    - Cleanup of temporary files
    - Data loader shutdown
    - Memory cleanup
    """
    # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
    # This will lead to the wrong current epoch to be stored
    trainer_instance.current_epoch -= 1
    trainer_instance.save_checkpoint(join(trainer_instance.output_folder, "checkpoint_final.pth"))
    trainer_instance.current_epoch += 1

    # now we can delete latest
    if trainer_instance.local_rank == 0 and isfile(join(trainer_instance.output_folder, "checkpoint_latest.pth")):
        os.remove(join(trainer_instance.output_folder, "checkpoint_latest.pth"))

    # shut down dataloaders
    old_stdout = sys.stdout
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        if (trainer_instance.dataloader_train is not None and
                isinstance(trainer_instance.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter))):
            trainer_instance.dataloader_train._finish()
        if (trainer_instance.dataloader_val is not None and
                isinstance(trainer_instance.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter))):
            trainer_instance.dataloader_val._finish()
        sys.stdout = old_stdout

    empty_cache(trainer_instance.device)
    trainer_instance.print_to_log_file("Training done.")


def on_train_epoch_start(trainer_instance):
    """
    Called at the beginning of each training epoch.

    Handles:
    - Setting network to training mode
    - Learning rate scheduling
    - Epoch logging
    """
    trainer_instance.network.train()
    trainer_instance.lr_scheduler.step(trainer_instance.current_epoch)
    trainer_instance.print_to_log_file('')
    trainer_instance.print_to_log_file(f'Epoch {trainer_instance.current_epoch}')
    trainer_instance.print_to_log_file(
        f"Current learning rate: {np.round(trainer_instance.optimizer.param_groups[0]['lr'], decimals=5)}")
    # lrs are the same for all workers so we don't need to gather them in case of DDP training
    trainer_instance.logger.log('lrs', trainer_instance.optimizer.param_groups[0]['lr'], trainer_instance.current_epoch)


def on_validation_epoch_start(trainer_instance):
    """
    Called at the beginning of each validation epoch.

    Handles:
    - Setting network to evaluation mode
    """
    trainer_instance.network.eval()


def on_epoch_start(trainer_instance):
    """
    Called at the beginning of each epoch (before training and validation).

    Handles:
    - Epoch timestamp logging
    """
    trainer_instance.logger.log('epoch_start_timestamps', time(), trainer_instance.current_epoch)


def on_epoch_end(trainer_instance):
    """
    Called at the end of each epoch (after training and validation).

    Handles:
    - Epoch timestamp logging
    - Loss and metrics logging
    - Checkpoint saving (periodic and best)
    - Progress plotting
    - Epoch counter increment
    """
    trainer_instance.logger.log('epoch_end_timestamps', time(), trainer_instance.current_epoch)

    trainer_instance.print_to_log_file('train_loss',
                                       np.round(trainer_instance.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
    trainer_instance.print_to_log_file('val_loss',
                                       np.round(trainer_instance.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
    trainer_instance.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                                       trainer_instance.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
    trainer_instance.print_to_log_file(
        f"Epoch time: {np.round(trainer_instance.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - trainer_instance.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

    # handling periodic checkpointing
    current_epoch = trainer_instance.current_epoch
    if (current_epoch + 1) % trainer_instance.save_every == 0 and current_epoch != (trainer_instance.num_epochs - 1):
        trainer_instance.save_checkpoint(join(trainer_instance.output_folder, 'checkpoint_latest.pth'))

    # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
    if (trainer_instance._best_ema is None or
        trainer_instance.logger.my_fantastic_logging['ema_fg_dice'][-1] > trainer_instance._best_ema):
        trainer_instance._best_ema = trainer_instance.logger.my_fantastic_logging['ema_fg_dice'][-1]
        trainer_instance.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(trainer_instance._best_ema, decimals=4)}")
        trainer_instance.save_checkpoint(join(trainer_instance.output_folder, 'checkpoint_best.pth'))

    if trainer_instance.local_rank == 0:
        trainer_instance.logger.plot_progress_png(trainer_instance.output_folder)

    trainer_instance.current_epoch += 1