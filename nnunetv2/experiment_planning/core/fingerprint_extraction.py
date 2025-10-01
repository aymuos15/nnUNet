from typing import List, Type

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.experiment_planning.config.defaults import DEFAULT_NUM_PROCESSES
from nnunetv2.experiment_planning.dataset.fingerprint.extractor import DatasetFingerprintExtractor
from nnunetv2.experiment_planning.dataset.validation.integrity_checker import verify_dataset_integrity
from nnunetv2.paths import nnUNet_raw
from nnunetv2.data.dataset_io.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.core.find_class_by_name import recursive_find_python_class


def extract_fingerprint_dataset(dataset_id: int,
                                fingerprint_extractor_class: Type[
                                    DatasetFingerprintExtractor] = DatasetFingerprintExtractor,
                                num_processes: int = DEFAULT_NUM_PROCESSES, check_dataset_integrity: bool = False,
                                clean: bool = True, verbose: bool = True):
    """
    Extract dataset fingerprint used for automatic configuration.

    Analyzes all training cases to compute statistics: image shapes, spacings,
    intensity distributions, class frequencies, and foreground regions. This
    fingerprint drives the automatic configuration process.

    Args:
        dataset_id: Dataset ID (from DatasetXXX_Name)
        fingerprint_extractor_class: Extractor class to use
        num_processes: Number of parallel processes for analysis
        check_dataset_integrity: Run integrity checks before extraction
        clean: If False, skip extraction if fingerprint file exists
        verbose: Enable verbose progress output

    Returns:
        Dictionary containing the dataset fingerprint (also saved to disk)

    Side Effects:
        Creates dataset_fingerprint.json in $nnUNet_preprocessed/DatasetXXX_Name/
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(dataset_name)

    if check_dataset_integrity:
        verify_dataset_integrity(join(nnUNet_raw, dataset_name), num_processes)

    fpe = fingerprint_extractor_class(dataset_id, num_processes, verbose=verbose)
    return fpe.run(overwrite_existing=clean)


def extract_fingerprints(dataset_ids: List[int], fingerprint_extractor_class_name: str = 'DatasetFingerprintExtractor',
                         num_processes: int = DEFAULT_NUM_PROCESSES, check_dataset_integrity: bool = False,
                         clean: bool = True, verbose: bool = True):
    """
    Extract fingerprints for multiple datasets.

    Batch processing version of extract_fingerprint_dataset. Dynamically loads
    the specified extractor class and applies it to each dataset.

    Args:
        dataset_ids: List of dataset IDs to process
        fingerprint_extractor_class_name: Name of extractor class (must be findable in experiment_planning module)
        num_processes: Number of parallel processes per dataset
        check_dataset_integrity: Run integrity checks before extraction
        clean: If False, skip datasets with existing fingerprints
        verbose: Enable verbose progress output

    Note:
        Setting clean=False is useful for nnUNetv2_plan_and_preprocess to avoid
        redundant fingerprint extraction on repeated runs.
    """
    fingerprint_extractor_class = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                              fingerprint_extractor_class_name,
                                                              current_module="nnunetv2.experiment_planning")
    for d in dataset_ids:
        extract_fingerprint_dataset(d, fingerprint_extractor_class, num_processes, check_dataset_integrity, clean,
                                    verbose)