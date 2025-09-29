import os
import shutil
from typing import Tuple, Union, List
import numpy as np
from torch import distributed as dist
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size


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
        do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
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