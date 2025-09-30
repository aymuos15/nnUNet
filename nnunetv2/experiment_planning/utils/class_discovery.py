"""
Utilities for dynamically discovering and loading classes.
"""

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.utilities.core.find_class_by_name import recursive_find_python_class


def find_fingerprint_extractor_class(class_name: str):
    """
    Find and return a fingerprint extractor class by name.

    Args:
        class_name: Name of the fingerprint extractor class

    Returns:
        The fingerprint extractor class
    """
    return recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                      class_name,
                                      current_module="nnunetv2.experiment_planning")


def find_experiment_planner_class(class_name: str):
    """
    Find and return an experiment planner class by name.

    Args:
        class_name: Name of the experiment planner class

    Returns:
        The experiment planner class
    """
    return recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                      class_name,
                                      current_module="nnunetv2.experiment_planning")


def find_preprocessor_class(class_name: str):
    """
    Find and return a preprocessor class by name.

    Args:
        class_name: Name of the preprocessor class

    Returns:
        The preprocessor class
    """
    return recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing"),
                                      class_name,
                                      current_module="nnunetv2.preprocessing")


def validate_class_exists(class_name: str, search_path: str, module_name: str) -> bool:
    """
    Check if a class exists without loading it.

    Args:
        class_name: Name of the class to check
        search_path: Path to search for the class
        module_name: Module name for the class

    Returns:
        True if class exists, False otherwise
    """
    try:
        recursive_find_python_class(search_path, class_name, current_module=module_name)
        return True
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False


def get_available_fingerprint_extractors() -> list:
    """
    Get a list of available fingerprint extractor classes.

    Returns:
        List of available fingerprint extractor class names
    """
    # This is a simplified implementation - in a full implementation,
    # you might scan the filesystem to discover available classes
    return [
        'DatasetFingerprintExtractor',
        # Add other extractors as they become available
    ]


def get_available_experiment_planners() -> list:
    """
    Get a list of available experiment planner classes.

    Returns:
        List of available experiment planner class names
    """
    return [
        'ExperimentPlanner',
        'ResEncUNetPlanner',
        'nnUNetPlannerResEncL',
        'nnUNetPlannerResEncM',
        'nnUNetPlannerResEncS',
        'nnUNetPlannerResEncL_noResampling',
        # Add other planners as they become available
    ]


def get_available_preprocessors() -> list:
    """
    Get a list of available preprocessor classes.

    Returns:
        List of available preprocessor class names
    """
    return [
        'DefaultPreprocessor',
        # Add other preprocessors as they become available
    ]