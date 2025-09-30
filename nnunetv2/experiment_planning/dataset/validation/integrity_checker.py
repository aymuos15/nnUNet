import multiprocessing
from typing import List
from os.path import dirname

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.utilities.planning.label_handling import LabelManager
from nnunetv2.utilities.planning.plans_handler import PlansManager
from nnunetv2.utilities.dataset_io.utils import get_filenames_of_train_images_and_targets
from .label_validators import verify_labels, validate_labels_consecutive
from .image_validators import check_cases


def validate_dataset_structure(folder: str, dataset_json: dict) -> None:
    """
    Validate basic dataset folder structure and required files.

    Args:
        folder: Dataset folder path
        dataset_json: Parsed dataset.json content

    Raises:
        AssertionError: If validation fails
    """
    assert isfile(join(folder, "dataset.json")), f"There needs to be a dataset.json file in folder, folder={folder}"

    if 'dataset' not in dataset_json.keys():
        assert isdir(join(folder, "imagesTr")), f"There needs to be a imagesTr subfolder in folder, folder={folder}"
        assert isdir(join(folder, "labelsTr")), f"There needs to be a labelsTr subfolder in folder, folder={folder}"


def validate_dataset_json_keys(dataset_json: dict) -> None:
    """
    Validate that dataset.json contains all required keys.

    Args:
        dataset_json: Parsed dataset.json content

    Raises:
        AssertionError: If validation fails
    """
    dataset_keys = list(dataset_json.keys())
    required_keys = ['labels', "channel_names", "numTraining", "file_ending"]
    missing_keys = [i for i in required_keys if i not in dataset_keys]
    unused_keys = [i for i in dataset_keys if i not in required_keys]

    assert all([i in dataset_keys for i in required_keys]), \
        'not all required keys are present in dataset.json.' \
        f'\n\nRequired: \n{str(required_keys)}' \
        f'\n\nPresent: \n{str(dataset_keys)}' \
        f'\n\nMissing: \n{str(missing_keys)}' \
        f'\n\nUnused by nnU-Net:\n{str(unused_keys)}'


def validate_training_cases_count(dataset: dict, expected_num_training: int) -> None:
    """
    Validate that the expected number of training cases is present.

    Args:
        dataset: Dataset dictionary with case information
        expected_num_training: Expected number of training cases

    Raises:
        AssertionError: If validation fails
    """
    assert len(dataset) == expected_num_training, \
        f'Did not find the expected number of training cases ({expected_num_training}). ' \
        f'Found {len(dataset)} instead.\nExamples: {list(dataset.keys())[:5]}'


def validate_files_exist(dataset: dict, dataset_json: dict) -> None:
    """
    Validate that all referenced files actually exist.

    Args:
        dataset: Dataset dictionary with case information
        dataset_json: Parsed dataset.json content

    Raises:
        FileNotFoundError: If validation fails
    """
    if 'dataset' in dataset_json.keys():
        # Check if everything is there
        missing_images = []
        missing_labels = []
        for k in dataset:
            for i in dataset[k]['images']:
                if not isfile(i):
                    missing_images.append(i)
            if not isfile(dataset[k]['label']):
                missing_labels.append(dataset[k]['label'])

        if missing_images or missing_labels:
            raise FileNotFoundError(
                f"Some expected files were missing. Make sure you are properly referencing them "
                f"in the dataset.json. Or use imagesTr & labelsTr folders!\nMissing images:"
                f"\n{missing_images}\n\nMissing labels:\n{missing_labels}")
    else:
        # Old code that uses imagestr and labelstr folders
        folder = dirname(list(dataset.values())[0]['images'][0])  # Get folder from first image path
        folder = dirname(folder)  # Go up one level to get dataset folder
        file_ending = dataset_json['file_ending']
        labelfiles = subfiles(join(folder, 'labelsTr'), suffix=file_ending, join=False)
        label_identifiers = [i[:-len(file_ending)] for i in labelfiles]
        labels_present = [i in label_identifiers for i in dataset.keys()]
        missing = [i for j, i in enumerate(dataset.keys()) if not labels_present[j]]
        assert all(labels_present), \
            f'not all training cases have a label file in labelsTr. Fix that. Missing: {missing}'


def verify_dataset_integrity(folder: str, num_processes: int = 8) -> None:
    """
    Verify the integrity of a dataset.

    folder needs the imagesTr, imagesTs and labelsTr subfolders. There also needs to be a dataset.json
    checks if the expected number of training cases and labels are present
    for each case, if possible, checks whether the pixel grids are aligned
    checks whether the labels really only contain values they should

    Args:
        folder: Path to dataset folder
        num_processes: Number of processes for parallel validation

    Raises:
        Various exceptions if validation fails
    """
    dataset_json = load_json(join(folder, "dataset.json"))

    # Validate basic structure
    validate_dataset_structure(folder, dataset_json)
    validate_dataset_json_keys(dataset_json)

    # Extract dataset information
    expected_num_training = dataset_json['numTraining']
    num_modalities = len(dataset_json['channel_names'].keys()
                         if 'channel_names' in dataset_json.keys()
                         else dataset_json['modality'].keys())
    file_ending = dataset_json['file_ending']

    dataset = get_filenames_of_train_images_and_targets(folder, dataset_json)

    # Validate dataset content
    validate_training_cases_count(dataset, expected_num_training)
    validate_files_exist(dataset, dataset_json)

    # Validate labels
    labelfiles = [v['label'] for v in dataset.values()]
    image_files = [v['images'] for v in dataset.values()]

    # Set up label validation
    label_manager = LabelManager(dataset_json['labels'], regions_class_order=dataset_json.get('regions_class_order'))
    expected_labels = label_manager.all_labels
    if label_manager.has_ignore_label:
        expected_labels.append(label_manager.ignore_label)

    # Validate label consecutiveness
    if not validate_labels_consecutive(expected_labels):
        raise AssertionError('Labels must be in consecutive order')

    # determine reader/writer class
    reader_writer_class = determine_reader_writer_from_dataset_json(
        dataset_json,
        dataset[list(dataset.keys())[0]]['images'][0]
    )

    # Parallel validation
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        # Check labels
        result = p.starmap(
            verify_labels,
            zip(labelfiles, [reader_writer_class] * len(labelfiles), [expected_labels] * len(labelfiles))
        )
        if not all(result):
            raise RuntimeError(
                'Some segmentation images contained unexpected labels. Please check text output above to see which one(s).')

        # Check shapes and spacings
        result = p.starmap(
            check_cases,
            zip(image_files, labelfiles, [num_modalities] * expected_num_training,
                [reader_writer_class] * expected_num_training)
        )
        if not all(result):
            raise RuntimeError(
                'Some images have errors. Please check text output above to see which one(s) and what\'s going on.')

    print('\n####################')
    print('verify_dataset_integrity Done. \nIf you didn\'t see any error messages then your dataset is most likely OK!')
    print('####################\n')