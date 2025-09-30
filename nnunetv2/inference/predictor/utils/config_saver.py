"""Shared configuration saving utilities for nnUNet predictor."""

import inspect
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json, join
from nnunetv2.utilities.json_export import recursive_fix_for_json_export


def save_prediction_args_and_dataset(predictor, output_folder: str, calling_function, local_vars: dict):
    """
    Save prediction configuration, arguments, and dataset information for reproducibility.

    Args:
        predictor: The nnUNetPredictor instance
        output_folder: Output folder path
        calling_function: The function whose arguments to save
        local_vars: Local variables from the calling function
    """
    if output_folder is None:
        return

    # Store input arguments
    my_init_kwargs = {}
    for k in inspect.signature(calling_function).parameters.keys():
        if k != 'predictor' and k != 'self' and k in local_vars:
            my_init_kwargs[k] = local_vars[k]

    my_init_kwargs = deepcopy(my_init_kwargs)
    recursive_fix_for_json_export(my_init_kwargs)
    maybe_mkdir_p(output_folder)
    save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

    # Save dataset and plans for potential postprocessing
    save_json(predictor.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
    save_json(predictor.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)