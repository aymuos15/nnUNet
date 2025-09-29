import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.crossval_split import generate_crossval_split


def do_split(trainer_instance):
    """Wrapper function to call the DataSplitter with trainer instance."""
    splitter = DataSplitter(
        trainer_instance.preprocessed_dataset_folder,
        trainer_instance.preprocessed_dataset_folder_base,
        trainer_instance.folder_with_segs_from_previous_stage,
        trainer_instance.fold,
        trainer_instance.dataset_class,
        trainer_instance.print_to_log_file
    )
    return splitter.do_split()


class DataSplitter:
    """Class containing the data splitting logic for nnUNet training."""

    def __init__(self, preprocessed_dataset_folder, preprocessed_dataset_folder_base,
                 folder_with_segs_from_previous_stage, fold, dataset_class=None, print_to_log_file=print):
        """
        Initialize DataSplitter.

        Args:
            preprocessed_dataset_folder: Path to preprocessed dataset folder
            preprocessed_dataset_folder_base: Base path to preprocessed dataset folder
            folder_with_segs_from_previous_stage: Path to previous stage segmentations
            fold: Fold number or "all" for using all data
            dataset_class: Dataset class to use (if None, will be inferred)
            print_to_log_file: Function to use for logging (default: print)
        """
        self.preprocessed_dataset_folder = preprocessed_dataset_folder
        self.preprocessed_dataset_folder_base = preprocessed_dataset_folder_base
        self.folder_with_segs_from_previous_stage = folder_with_segs_from_previous_stage
        self.fold = fold
        self.dataset_class = dataset_class
        self.print_to_log_file = print_to_log_file

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                         identifiers=None,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys