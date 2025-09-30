import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.training_runtime.crossval_split import generate_crossval_split


def do_split(trainer_instance):
    """
    Perform dataset split for training and validation.

    The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
    so always the same) and save it as splits_final.json file in the preprocessed data directory.
    Sometimes you may want to create your own split for various reasons. For this you will need to create your own
    splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
    it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
    and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
    use a random 80:20 data split.

    Args:
        trainer_instance: The nnUNetTrainer instance containing configuration

    Returns:
        Tuple of (training_keys, validation_keys)
    """
    if trainer_instance.dataset_class is None:
        trainer_instance.dataset_class = infer_dataset_class(trainer_instance.preprocessed_dataset_folder)

    if trainer_instance.fold == "all":
        # if fold==all then we use all images for training and validation
        case_identifiers = trainer_instance.dataset_class.get_identifiers(trainer_instance.preprocessed_dataset_folder)
        tr_keys = case_identifiers
        val_keys = tr_keys
    else:
        splits_file = join(trainer_instance.preprocessed_dataset_folder_base, "splits_final.json")
        dataset = trainer_instance.dataset_class(trainer_instance.preprocessed_dataset_folder,
                                     identifiers=None,
                                     folder_with_segs_from_previous_stage=trainer_instance.folder_with_segs_from_previous_stage)
        # if the split file does not exist we need to create it
        if not isfile(splits_file):
            trainer_instance.print_to_log_file("Creating new 5-fold cross-validation split...")
            all_keys_sorted = list(np.sort(list(dataset.identifiers)))
            splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
            save_json(splits, splits_file)

        else:
            trainer_instance.print_to_log_file("Using splits from existing split file:", splits_file)
            splits = load_json(splits_file)
            trainer_instance.print_to_log_file(f"The split file contains {len(splits)} splits.")

        trainer_instance.print_to_log_file("Desired fold for training: %d" % trainer_instance.fold)
        if trainer_instance.fold < len(splits):
            tr_keys = splits[trainer_instance.fold]['train']
            val_keys = splits[trainer_instance.fold]['val']
            trainer_instance.print_to_log_file("This split has %d training and %d validation cases."
                                   % (len(tr_keys), len(val_keys)))
        else:
            trainer_instance.print_to_log_file("INFO: You requested fold %d for training but splits "
                                   "contain only %d folds. I am now creating a "
                                   "random (but seeded) 80:20 split!" % (trainer_instance.fold, len(splits)))
            # if we request a fold that is not in the split file, create a random 80:20 split
            rnd = np.random.RandomState(seed=12345 + trainer_instance.fold)
            keys = np.sort(list(dataset.identifiers))
            idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
            idx_val = [i for i in range(len(keys)) if i not in idx_tr]
            tr_keys = [keys[i] for i in idx_tr]
            val_keys = [keys[i] for i in idx_val]
            trainer_instance.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                   % (len(tr_keys), len(val_keys)))
        if any([i in val_keys for i in tr_keys]):
            trainer_instance.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                   'splits.json or ignore if this is intentional.')
    return tr_keys, val_keys


def get_tr_and_val_datasets(trainer_instance):
    """
    Get training and validation datasets based on the split.

    Args:
        trainer_instance: The nnUNetTrainer instance containing configuration

    Returns:
        Tuple of (training_dataset, validation_dataset)
    """
    # create dataset split
    tr_keys, val_keys = do_split(trainer_instance)

    # load the datasets for training and validation. Note that we always draw random samples so we really don't
    # care about distributing training cases across GPUs.
    dataset_tr = trainer_instance.dataset_class(trainer_instance.preprocessed_dataset_folder, tr_keys,
                                    folder_with_segs_from_previous_stage=trainer_instance.folder_with_segs_from_previous_stage)
    dataset_val = trainer_instance.dataset_class(trainer_instance.preprocessed_dataset_folder, val_keys,
                                     folder_with_segs_from_previous_stage=trainer_instance.folder_with_segs_from_previous_stage)
    return dataset_tr, dataset_val