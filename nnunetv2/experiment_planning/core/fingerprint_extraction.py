from typing import List, Type

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.dataset.fingerprint.extractor import DatasetFingerprintExtractor
from nnunetv2.experiment_planning.dataset.validation.integrity_checker import verify_dataset_integrity
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class


def extract_fingerprint_dataset(dataset_id: int,
                                fingerprint_extractor_class: Type[
                                    DatasetFingerprintExtractor] = DatasetFingerprintExtractor,
                                num_processes: int = default_num_processes, check_dataset_integrity: bool = False,
                                clean: bool = True, verbose: bool = True):
    """
    Returns the fingerprint as a dictionary (additionally to saving it)
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(dataset_name)

    if check_dataset_integrity:
        verify_dataset_integrity(join(nnUNet_raw, dataset_name), num_processes)

    fpe = fingerprint_extractor_class(dataset_id, num_processes, verbose=verbose)
    return fpe.run(overwrite_existing=clean)


def extract_fingerprints(dataset_ids: List[int], fingerprint_extractor_class_name: str = 'DatasetFingerprintExtractor',
                         num_processes: int = default_num_processes, check_dataset_integrity: bool = False,
                         clean: bool = True, verbose: bool = True):
    """
    clean = False will not actually run this. This is just a switch for use with nnUNetv2_plan_and_preprocess where
    we don't want to rerun fingerprint extraction every time.
    """
    fingerprint_extractor_class = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                              fingerprint_extractor_class_name,
                                                              current_module="nnunetv2.experiment_planning")
    for d in dataset_ids:
        extract_fingerprint_dataset(d, fingerprint_extractor_class, num_processes, check_dataset_integrity, clean,
                                    verbose)