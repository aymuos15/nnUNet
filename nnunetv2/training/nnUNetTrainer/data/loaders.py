from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.training_runtime.default_n_proc_DA import get_allowed_n_proc_DA
from .datasets import get_tr_and_val_datasets
from .transforms import get_training_transforms, get_validation_transforms
from ..initialization.config import configure_rotation_dummyDA_mirroring_and_inital_patch_size


def get_dataloaders(trainer_instance):
    """
    Get training and validation data loaders for the trainer.

    Args:
        trainer_instance: The nnUNetTrainer instance containing configuration

    Returns:
        Tuple of (training_dataloader, validation_dataloader)
    """
    if trainer_instance.dataset_class is None:
        trainer_instance.dataset_class = infer_dataset_class(trainer_instance.preprocessed_dataset_folder)

    # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
    # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
    patch_size = trainer_instance.configuration_manager.patch_size

    # needed for deep supervision: how much do we need to downscale the segmentation targets for the different outputs?
    deep_supervision_scales = trainer_instance._get_deep_supervision_scales()

    (
        rotation_for_DA,
        do_dummy_2d_data_aug,
        initial_patch_size,
        mirror_axes,
    ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(trainer_instance)

    # training pipeline
    tr_transforms = get_training_transforms(
        patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        use_mask_for_norm=trainer_instance.configuration_manager.use_mask_for_norm,
        is_cascaded=trainer_instance.is_cascaded,
        foreground_labels=trainer_instance.label_manager.foreground_labels,
        regions=trainer_instance.label_manager.foreground_regions if trainer_instance.label_manager.has_regions else None,
        ignore_label=trainer_instance.label_manager.ignore_label)

    # validation pipeline
    val_transforms = get_validation_transforms(deep_supervision_scales,
                                                    is_cascaded=trainer_instance.is_cascaded,
                                                    foreground_labels=trainer_instance.label_manager.foreground_labels,
                                                    regions=trainer_instance.label_manager.foreground_regions if
                                                    trainer_instance.label_manager.has_regions else None,
                                                    ignore_label=trainer_instance.label_manager.ignore_label)

    dataset_tr, dataset_val = get_tr_and_val_datasets(trainer_instance)

    dl_tr = nnUNetDataLoader(dataset_tr, trainer_instance.batch_size,
                             initial_patch_size,
                             trainer_instance.configuration_manager.patch_size,
                             trainer_instance.label_manager,
                             oversample_foreground_percent=trainer_instance.oversample_foreground_percent,
                             sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                             probabilistic_oversampling=trainer_instance.probabilistic_oversampling)
    dl_val = nnUNetDataLoader(dataset_val, trainer_instance.batch_size,
                              trainer_instance.configuration_manager.patch_size,
                              trainer_instance.configuration_manager.patch_size,
                              trainer_instance.label_manager,
                              oversample_foreground_percent=trainer_instance.oversample_foreground_percent,
                              sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                              probabilistic_oversampling=trainer_instance.probabilistic_oversampling)

    allowed_num_processes = get_allowed_n_proc_DA()
    if allowed_num_processes == 0:
        mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)
    else:
        mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                    num_processes=allowed_num_processes,
                                                    num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                    pin_memory=trainer_instance.device.type == 'cuda', wait_time=0.002)
        mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                  transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                  num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                  pin_memory=trainer_instance.device.type == 'cuda',
                                                  wait_time=0.002)
    # # let's get this party started
    _ = next(mt_gen_train)
    _ = next(mt_gen_val)
    return mt_gen_train, mt_gen_val




