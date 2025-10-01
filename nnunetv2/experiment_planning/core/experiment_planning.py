from typing import List, Type, Optional, Tuple, Union

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.experiment_planning.planners.standard.default_planner import ExperimentPlanner
from nnunetv2.utilities.core.find_class_by_name import recursive_find_python_class


def plan_experiment_dataset(dataset_id: int,
                            experiment_planner_class: Type[ExperimentPlanner] = ExperimentPlanner,
                            gpu_memory_target_in_gb: float = None, preprocess_class_name: str = 'DefaultPreprocessor',
                            overwrite_target_spacing: Optional[Tuple[float, ...]] = None,
                            overwrite_plans_name: Optional[str] = None) -> Tuple[dict, str]:
    """
    Generate training plans for a single dataset.

    Creates nnU-Net configurations (2D, 3D full-res, 3D cascade) by analyzing the
    dataset fingerprint and applying rule-based heuristics for preprocessing,
    architecture, and training parameters.

    Args:
        dataset_id: Dataset ID (from DatasetXXX_Name)
        experiment_planner_class: Planner class to use for plan generation
        gpu_memory_target_in_gb: Target GPU memory in GB (affects patch size and batch size)
        preprocess_class_name: Name of preprocessor class to use
        overwrite_target_spacing: Override automatic spacing determination (3d_fullres and 3d_cascade only)
        overwrite_plans_name: Custom name for the generated plans file

    Returns:
        Tuple of (plans dictionary, plans identifier string)

    Note:
        overwrite_target_spacing only affects 3d_fullres and 3d_cascade_fullres configurations.
    """
    kwargs = {}
    if overwrite_plans_name is not None:
        kwargs['plans_name'] = overwrite_plans_name
    if gpu_memory_target_in_gb is not None:
        kwargs['gpu_memory_target_in_gb'] = gpu_memory_target_in_gb

    planner = experiment_planner_class(dataset_id,
                                       preprocessor_name=preprocess_class_name,
                                       overwrite_target_spacing=[float(i) for i in overwrite_target_spacing] if
                                       overwrite_target_spacing is not None else overwrite_target_spacing,
                                       suppress_transpose=False,  # might expose this later,
                                       **kwargs
                                       )
    ret = planner.plan_experiment()
    return ret, planner.plans_identifier


def plan_experiments(dataset_ids: List[int], experiment_planner_class_name: str = 'ExperimentPlanner',
                     gpu_memory_target_in_gb: float = None, preprocess_class_name: str = 'DefaultPreprocessor',
                     overwrite_target_spacing: Optional[Tuple[float, ...]] = None,
                     overwrite_plans_name: Optional[str] = None):
    """
    Generate training plans for multiple datasets.

    Batch processing version of plan_experiment_dataset. Dynamically loads the
    specified planner class and applies it to each dataset.

    Args:
        dataset_ids: List of dataset IDs to process
        experiment_planner_class_name: Name of planner class (must be findable in experiment_planning module)
        gpu_memory_target_in_gb: Target GPU memory in GB
        preprocess_class_name: Name of preprocessor class
        overwrite_target_spacing: Override automatic spacing (3d_fullres and 3d_cascade only)
        overwrite_plans_name: Custom plans file name

    Returns:
        Plans identifier string from the last processed dataset

    Note:
        Using 'ExperimentPlanner' will show a warning recommending ResEncUNet presets.
        overwrite_target_spacing only affects 3d_fullres and 3d_cascade_fullres configurations.
    """
    if experiment_planner_class_name == 'ExperimentPlanner':
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default planner. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    experiment_planner = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                     experiment_planner_class_name,
                                                     current_module="nnunetv2.experiment_planning")
    plans_identifier = None
    for d in dataset_ids:
        _, plans_identifier = plan_experiment_dataset(d, experiment_planner, gpu_memory_target_in_gb,
                                                      preprocess_class_name,
                                                      overwrite_target_spacing, overwrite_plans_name)
    return plans_identifier