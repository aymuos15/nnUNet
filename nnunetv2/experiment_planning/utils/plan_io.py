"""
Utilities for handling plans file I/O operations.
"""

import os
from typing import Dict, Optional

from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, isfile, maybe_mkdir_p
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_io.dataset_name_id_conversion import convert_id_to_dataset_name


def get_plans_file_path(dataset_id: int, plans_name: str = 'nnUNetPlans') -> str:
    """
    Get the path to a plans file for a dataset.

    Args:
        dataset_id: Dataset ID
        plans_name: Name of the plans file (without .json extension)

    Returns:
        Full path to the plans file
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    return join(nnUNet_preprocessed, dataset_name, plans_name + '.json')


def load_plans_file(dataset_id: int, plans_name: str = 'nnUNetPlans') -> Optional[Dict]:
    """
    Load a plans file for a dataset.

    Args:
        dataset_id: Dataset ID
        plans_name: Name of the plans file

    Returns:
        Plans dictionary if file exists, None otherwise
    """
    plans_file = get_plans_file_path(dataset_id, plans_name)
    if isfile(plans_file):
        return load_json(plans_file)
    return None


def save_plans_file(dataset_id: int, plans: Dict, plans_name: str = 'nnUNetPlans') -> str:
    """
    Save a plans file for a dataset.

    Args:
        dataset_id: Dataset ID
        plans: Plans dictionary to save
        plans_name: Name of the plans file

    Returns:
        Path to the saved plans file

    Raises:
        Exception: If saving fails
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    output_folder = join(nnUNet_preprocessed, dataset_name)
    maybe_mkdir_p(output_folder)

    plans_file = join(output_folder, plans_name + '.json')

    try:
        save_json(plans, plans_file)
        return plans_file
    except Exception as e:
        if isfile(plans_file):
            os.remove(plans_file)
        raise e


def plans_file_exists(dataset_id: int, plans_name: str = 'nnUNetPlans') -> bool:
    """
    Check if a plans file exists for a dataset.

    Args:
        dataset_id: Dataset ID
        plans_name: Name of the plans file

    Returns:
        True if plans file exists, False otherwise
    """
    plans_file = get_plans_file_path(dataset_id, plans_name)
    return isfile(plans_file)


def get_fingerprint_file_path(dataset_id: int) -> str:
    """
    Get the path to a dataset fingerprint file.

    Args:
        dataset_id: Dataset ID

    Returns:
        Full path to the fingerprint file
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    return join(nnUNet_preprocessed, dataset_name, 'dataset_fingerprint.json')


def load_fingerprint_file(dataset_id: int) -> Optional[Dict]:
    """
    Load a dataset fingerprint file.

    Args:
        dataset_id: Dataset ID

    Returns:
        Fingerprint dictionary if file exists, None otherwise
    """
    fingerprint_file = get_fingerprint_file_path(dataset_id)
    if isfile(fingerprint_file):
        return load_json(fingerprint_file)
    return None


def fingerprint_file_exists(dataset_id: int) -> bool:
    """
    Check if a fingerprint file exists for a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        True if fingerprint file exists, False otherwise
    """
    fingerprint_file = get_fingerprint_file_path(dataset_id)
    return isfile(fingerprint_file)


def create_backup_plans_name(base_name: str, suffix: str = None) -> str:
    """
    Create a backup plans name with optional suffix.

    Args:
        base_name: Base plans name
        suffix: Optional suffix to add

    Returns:
        Backup plans name
    """
    if suffix:
        return f"{base_name}_{suffix}_backup"
    else:
        return f"{base_name}_backup"


def list_available_plans(dataset_id: int) -> list:
    """
    List all available plans files for a dataset.

    Args:
        dataset_id: Dataset ID

    Returns:
        List of available plans file names (without .json extension)
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    folder = join(nnUNet_preprocessed, dataset_name)

    if not os.path.isdir(folder):
        return []

    plans_files = []
    for file in os.listdir(folder):
        if file.endswith('.json') and file != 'dataset_fingerprint.json':
            plans_files.append(file[:-5])  # Remove .json extension

    return sorted(plans_files)