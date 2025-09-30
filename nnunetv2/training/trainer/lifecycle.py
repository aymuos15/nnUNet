"""
Trainer lifecycle and initialization management.

This module handles:
- Configuration initialization (batch size, folders, data augmentation params)
- Network architecture building and compilation
- Training lifecycle hooks (train/validation/epoch start/end)
- Main training orchestration loop
"""

import os
import sys
import shutil
from time import time
from typing import Union, List, Tuple

import numpy as np
import torch
from torch import nn, distributed as dist
from torch._dynamo import OptimizedModule

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p, isfile

from nnunetv2.experiment_planning.config.defaults import DEFAULT_ANISO_THRESHOLD
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data.transforms.compute_initial_patch_size import get_patch_size
from nnunetv2.utilities.planning.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.training_runtime.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.core.helpers import empty_cache

from .training import train_step, on_train_epoch_end, on_validation_epoch_end
from .validation import validation_step
# get_dataloaders imported locally within on_train_start to avoid circular import


# ============================================================================
# Configuration and Initialization Functions
# ============================================================================

def _set_batch_size_and_oversample(trainer_instance):
    """
    Set batch size and oversample percentage for training, handling DDP distribution.

    Args:
        trainer_instance: The nnUNetTrainer instance
    """
    if not trainer_instance.is_ddp:
        # set batch size to what the plan says, leave oversample untouched
        trainer_instance.batch_size = trainer_instance.configuration_manager.batch_size
    else:
        # batch size is distributed over DDP workers and we need to change oversample_percent for each worker

        world_size = dist.get_world_size()
        my_rank = dist.get_rank()

        global_batch_size = trainer_instance.configuration_manager.batch_size
        assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of ' \
                                                'GPUs... Duh.'

        batch_size_per_GPU = [global_batch_size // world_size] * world_size
        batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                              if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                              else batch_size_per_GPU[i]
                              for i in range(len(batch_size_per_GPU))]
        assert sum(batch_size_per_GPU) == global_batch_size

        sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
        sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])

        # This is how oversampling is determined in DataLoader
        # round(self.batch_size * (1 - self.oversample_foreground_percent))
        # We need to use the same scheme here because an oversample of 0.33 with a batch size of 2 will be rounded
        # to an oversample of 0.5 (1 sample random, one oversampled). This may get lost if we just numerically
        # compute oversample
        oversample = [True if not i < round(global_batch_size * (1 - trainer_instance.oversample_foreground_percent)) else False
                      for i in range(global_batch_size)]

        if sample_id_high / global_batch_size < (1 - trainer_instance.oversample_foreground_percent):
            oversample_percent = 0.0
        elif sample_id_low / global_batch_size > (1 - trainer_instance.oversample_foreground_percent):
            oversample_percent = 1.0
        else:
            oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]

        print("worker", my_rank, "oversample", oversample_percent)
        print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])

        trainer_instance.batch_size = batch_size_per_GPU[my_rank]
        trainer_instance.oversample_foreground_percent = oversample_percent


def configure_rotation_dummyDA_mirroring_and_inital_patch_size(trainer_instance):
    """
    This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.

    Args:
        trainer_instance: The nnUNetTrainer instance

    Returns:
        Tuple containing: rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
    """
    patch_size = trainer_instance.configuration_manager.patch_size
    dim = len(patch_size)
    # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
    if dim == 2:
        do_dummy_2d_data_aug = False
        # todo revisit this parametrization
        if max(patch_size) / min(patch_size) > 1.5:
            rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
        else:
            rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        mirror_axes = (0, 1)
    elif dim == 3:
        # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
        # order of the axes is determined by spacing, not image size
        do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > DEFAULT_ANISO_THRESHOLD
        if do_dummy_2d_data_aug:
            # why do we rotate 180 deg here all the time? We should also restrict it
            rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        else:
            rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        mirror_axes = (0, 1, 2)
    else:
        raise RuntimeError()

    # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
    #  old nnunet for now)
    initial_patch_size = get_patch_size(patch_size[-dim:],
                                        rotation_for_DA,
                                        rotation_for_DA,
                                        rotation_for_DA,
                                        (0.85, 1.25))
    if do_dummy_2d_data_aug:
        initial_patch_size[0] = patch_size[0]

    trainer_instance.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
    trainer_instance.inference_allowed_mirroring_axes = mirror_axes

    # Apply config overrides if provided
    if hasattr(trainer_instance, 'trainer_config') and trainer_instance.trainer_config is not None:
        config = trainer_instance.trainer_config
        if config.mirror_axes is not None:
            mirror_axes = config.mirror_axes
        if config.inference_allowed_mirroring_axes is not None:
            trainer_instance.inference_allowed_mirroring_axes = config.inference_allowed_mirroring_axes
        if config.do_dummy_2d_data_aug is not None:
            do_dummy_2d_data_aug = config.do_dummy_2d_data_aug
        if config.rotation_for_DA is not None:
            rotation_for_DA = config.rotation_for_DA

    return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


def setup_output_folders(trainer_instance):
    """
    Set up all output folder paths for the trainer.

    Args:
        trainer_instance: The nnUNetTrainer instance
    """
    # Setting all the folder names. We need to make sure things don't crash in case we are just running
    # inference and some of the folders may not be defined!
    trainer_instance.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, trainer_instance.plans_manager.dataset_name) \
        if nnUNet_preprocessed is not None else None
    trainer_instance.output_folder_base = join(nnUNet_results, trainer_instance.plans_manager.dataset_name,
                                   trainer_instance.__class__.__name__ + '__' + trainer_instance.plans_manager.plans_name + "__" + trainer_instance.configuration_name) \
        if nnUNet_results is not None else None
    trainer_instance.output_folder = join(trainer_instance.output_folder_base, f'fold_{trainer_instance.fold}')

    trainer_instance.preprocessed_dataset_folder = join(trainer_instance.preprocessed_dataset_folder_base,
                                            trainer_instance.configuration_manager.data_identifier)


def setup_cascaded_folders(trainer_instance):
    """
    Set up folders for cascaded training if applicable.

    Args:
        trainer_instance: The nnUNetTrainer instance
    """
    # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
    # be a different configuration in the same plans
    # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
    # "previous_stage" and "next_stage"). Otherwise it won't work!
    trainer_instance.is_cascaded = trainer_instance.configuration_manager.previous_stage_name is not None
    trainer_instance.folder_with_segs_from_previous_stage = \
        join(nnUNet_results, trainer_instance.plans_manager.dataset_name,
             trainer_instance.__class__.__name__ + '__' + trainer_instance.plans_manager.plans_name + "__" +
             trainer_instance.configuration_manager.previous_stage_name, 'predicted_next_stage', trainer_instance.configuration_name) \
            if trainer_instance.is_cascaded else None


def copy_plans_and_dataset_json(trainer_instance):
    """
    Copy plans.json and dataset.json to output folder for inference reproducibility.

    Args:
        trainer_instance: The nnUNetTrainer instance
    """
    # copy plans and dataset.json so that they can be used for restoring everything we need for inference
    save_json(trainer_instance.plans_manager.plans, join(trainer_instance.output_folder_base, 'plans.json'), sort_keys=False)
    save_json(trainer_instance.dataset_json, join(trainer_instance.output_folder_base, 'dataset.json'), sort_keys=False)

    # we don't really need the fingerprint but its still handy to have it with the others
    shutil.copy(join(trainer_instance.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                join(trainer_instance.output_folder_base, 'dataset_fingerprint.json'))


def ensure_output_folder_exists(trainer_instance):
    """
    Create output folder if it doesn't exist.

    Args:
        trainer_instance: The nnUNetTrainer instance
    """
    maybe_mkdir_p(trainer_instance.output_folder)


# ============================================================================
# Network Architecture Functions
# ============================================================================

def build_network_architecture(architecture_class_name: str,
                               arch_init_kwargs: dict,
                               arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                               num_input_channels: int,
                               num_output_channels: int,
                               enable_deep_supervision: bool = True) -> nn.Module:
    """
    This is where you build the architecture according to the plans. There is no obligation to use
    get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
    you want. Even ignore the plans and just return something static (as long as it can process the requested
    patch size)
    but don't bug us with your bugs arising from fiddling with this :-P
    This is the function that is called in inference as well! This is needed so that all network architecture
    variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
    training, so if you change the network architecture during training by deriving a new trainer class then
    inference will know about it).

    If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
    > label_manager = plans_manager.get_label_manager(dataset_json)
    > label_manager.num_segmentation_heads
    (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
    the number of outputs is != the number of classes. Also there is the ignore label for which no output
    should be generated. label_manager takes care of all that for you.)

    """
    return get_network_from_plans(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=True,
        deep_supervision=enable_deep_supervision)


def _do_i_compile(trainer_instance):
    """
    Determine whether to compile the network using torch.compile.

    Args:
        trainer_instance: The nnUNetTrainer instance

    Returns:
        bool: Whether to compile the network
    """
    # new default: compile is enabled!

    # compile does not work on mps
    if trainer_instance.device == torch.device('mps'):
        if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
            trainer_instance.print_to_log_file("INFO: torch.compile disabled because of unsupported mps device")
        return False

    # CPU compile crashes for 2D models. Not sure if we even want to support CPU compile!? Better disable
    if trainer_instance.device == torch.device('cpu'):
        if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
            trainer_instance.print_to_log_file("INFO: torch.compile disabled because device is CPU")
        return False

    # default torch.compile doesn't work on windows because there are apparently no triton wheels for it
    # https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2
    if os.name == 'nt':
        if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
            trainer_instance.print_to_log_file("INFO: torch.compile disabled because Windows is not natively supported. If "
                                   "you know what you are doing, check https://discuss.pytorch.org/t/windows-support-timeline-for-torch-compile/182268/2")
        return False

    if 'nnUNet_compile' not in os.environ.keys():
        return True
    else:
        return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')


def plot_network_architecture(trainer_instance):
    """
    Plot the network architecture and save it as a PDF.

    Args:
        trainer_instance: The nnUNetTrainer instance
    """
    if _do_i_compile(trainer_instance):
        trainer_instance.print_to_log_file("Unable to plot network architecture: nnUNet_compile is enabled!")
        return

    if trainer_instance.local_rank == 0:
        try:
            # raise NotImplementedError('hiddenlayer no longer works and we do not have a viable alternative :-(')
            # pip install git+https://github.com/saugatkandel/hiddenlayer.git

            # from torchviz import make_dot
            # # not viable.
            # make_dot(tuple(self.network(torch.rand((1, self.num_input_channels,
            #                                         *self.configuration_manager.patch_size),
            #                                        device=self.device)))).render(
            #     join(self.output_folder, "network_architecture.pdf"), format='pdf')
            # self.optimizer.zero_grad()

            # broken.

            import hiddenlayer as hl
            g = hl.build_graph(trainer_instance.network,
                               torch.rand((1, trainer_instance.num_input_channels,
                                           *trainer_instance.configuration_manager.patch_size),
                                          device=trainer_instance.device),
                               transforms=None)
            g.save(join(trainer_instance.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            trainer_instance.print_to_log_file("Unable to plot network architecture:")
            trainer_instance.print_to_log_file(e)

            # self.print_to_log_file("\nprinting the network instead:\n")
            # self.print_to_log_file(self.network)
            # self.print_to_log_file("\n")
        finally:
            empty_cache(trainer_instance.device)


def set_deep_supervision_enabled(trainer_instance, enabled: bool):
    """
    Enable or disable deep supervision in the network.
    This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
    chances you need to change this as well!

    Args:
        trainer_instance: The nnUNetTrainer instance
        enabled: Whether to enable deep supervision
    """
    if trainer_instance.is_ddp:
        mod = trainer_instance.network.module
    else:
        mod = trainer_instance.network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    mod.decoder.deep_supervision = enabled


# ============================================================================
# Training Lifecycle Hooks
# ============================================================================

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
    from nnunetv2.training.data.loader import get_dataloaders  # local import to avoid circular dependency
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


# ============================================================================
# Training Orchestration
# ============================================================================

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