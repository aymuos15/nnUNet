"""
Utility functions for working with datasets.
"""

from typing import Dict, List, Tuple

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.data.dataset_io.dataset_name_id_conversion import convert_id_to_dataset_name, maybe_convert_to_dataset_name
from nnunetv2.data.dataset_io.utils import get_filenames_of_train_images_and_targets


def get_dataset_info(dataset_id: int) -> Dict:
    """
    Get basic information about a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        Dictionary with dataset information
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    dataset_folder = join(nnUNet_raw, dataset_name)
    dataset_json = load_json(join(dataset_folder, 'dataset.json'))

    return {
        'dataset_id': dataset_id,
        'dataset_name': dataset_name,
        'dataset_folder': dataset_folder,
        'num_training': dataset_json.get('numTraining', 0),
        'num_test': dataset_json.get('numTest', 0),
        'modalities': dataset_json.get('channel_names', dataset_json.get('modality', {})),
        'labels': dataset_json.get('labels', {}),
        'file_ending': dataset_json.get('file_ending', '.nii.gz'),
        'description': dataset_json.get('description', ''),
    }


def get_dataset_modalities(dataset_id: int) -> Dict:
    """
    Get the modalities (channels) for a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        Dictionary mapping modality indices to names
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    dataset_folder = join(nnUNet_raw, dataset_name)
    dataset_json = load_json(join(dataset_folder, 'dataset.json'))

    return dataset_json.get('channel_names', dataset_json.get('modality', {}))


def get_dataset_labels(dataset_id: int) -> Dict:
    """
    Get the labels for a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        Dictionary mapping label values to names
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    dataset_folder = join(nnUNet_raw, dataset_name)
    dataset_json = load_json(join(dataset_folder, 'dataset.json'))

    return dataset_json.get('labels', {})


def get_num_channels(dataset_id: int) -> int:
    """
    Get the number of input channels for a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        Number of input channels
    """
    modalities = get_dataset_modalities(dataset_id)
    return len(modalities)


def get_num_classes(dataset_id: int) -> int:
    """
    Get the number of classes (including background) for a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        Number of classes
    """
    labels = get_dataset_labels(dataset_id)
    return len(labels)


def get_training_cases(dataset_id: int) -> Dict:
    """
    Get information about training cases for a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        Dictionary with training case information
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    dataset_folder = join(nnUNet_raw, dataset_name)
    dataset_json = load_json(join(dataset_folder, 'dataset.json'))

    return get_filenames_of_train_images_and_targets(dataset_folder, dataset_json)


def validate_dataset_structure(dataset_id: int) -> Tuple[bool, List[str]]:
    """
    Validate the basic structure of a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    try:
        dataset_name = convert_id_to_dataset_name(dataset_id)
        dataset_folder = join(nnUNet_raw, dataset_name)

        # Check if dataset.json exists
        try:
            dataset_json = load_json(join(dataset_folder, 'dataset.json'))
        except Exception as e:
            errors.append(f"Could not load dataset.json: {str(e)}")
            return False, errors

        # Check required keys
        required_keys = ['labels', 'numTraining', 'file_ending']
        for key in required_keys:
            if key not in dataset_json:
                errors.append(f"Missing required key in dataset.json: {key}")

        # Check modality keys
        if 'channel_names' not in dataset_json and 'modality' not in dataset_json:
            errors.append("Missing both 'channel_names' and 'modality' keys in dataset.json")

        # Check if training cases exist
        try:
            training_cases = get_training_cases(dataset_id)
            expected_num = dataset_json.get('numTraining', 0)
            if len(training_cases) != expected_num:
                errors.append(f"Expected {expected_num} training cases, found {len(training_cases)}")
        except Exception as e:
            errors.append(f"Error loading training cases: {str(e)}")

    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")

    return len(errors) == 0, errors


def get_dataset_summary(dataset_ids: List[int]) -> Dict:
    """
    Get a summary of multiple datasets.

    Args:
        dataset_ids: List of dataset IDs

    Returns:
        Dictionary with summary information
    """
    summary = {
        'total_datasets': len(dataset_ids),
        'datasets': {},
        'total_training_cases': 0,
        'unique_modalities': set(),
        'file_endings': set(),
    }

    for dataset_id in dataset_ids:
        try:
            info = get_dataset_info(dataset_id)
            summary['datasets'][dataset_id] = info
            summary['total_training_cases'] += info['num_training']
            summary['unique_modalities'].update(info['modalities'].values())
            summary['file_endings'].add(info['file_ending'])
        except Exception as e:
            summary['datasets'][dataset_id] = {'error': str(e)}

    # Convert sets to lists for JSON serialization
    summary['unique_modalities'] = list(summary['unique_modalities'])
    summary['file_endings'] = list(summary['file_endings'])

    return summary