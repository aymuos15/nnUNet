"""Sequential (non-parallel) prediction for nnUNet predictor."""

import os
from typing import Union, List

import torch
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from ..postprocessing.export_prediction import (
    export_prediction_from_logits,
    convert_predicted_logits_to_segmentation_with_correct_shape
)
from .sliding_window_utils import compute_gaussian
from ..utils.config_saver import save_prediction_args_and_dataset
from nnunetv2.utilities.helpers import empty_cache


def predict_from_files_sequential(predictor,
                                 list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                 output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                                 save_probabilities: bool = False,
                                 overwrite: bool = True,
                                 folder_with_segs_from_prev_stage: str = None):
    """
    Sequential prediction without multiprocessing. Slower but sometimes necessary.

    Args:
        predictor: The nnUNetPredictor instance
        list_of_lists_or_source_folder: Input files or folder path
        output_folder_or_list_of_truncated_output_files: Output location
        save_probabilities: Whether to save probability maps
        overwrite: Whether to overwrite existing predictions
        folder_with_segs_from_prev_stage: Previous stage segmentations folder

    Returns:
        List of predictions
    """
    # Determine output folder
    if isinstance(output_folder_or_list_of_truncated_output_files, str):
        output_folder = output_folder_or_list_of_truncated_output_files
    elif isinstance(output_folder_or_list_of_truncated_output_files, list):
        output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        if len(output_folder) == 0:  # just a file was given without a folder
            output_folder = os.path.curdir
    else:
        output_folder = None

    # Save configuration for reproducibility
    save_prediction_args_and_dataset(predictor, output_folder, predict_from_files_sequential, locals())

    # Check cascade requirements
    if predictor.configuration_manager.previous_stage_name is not None:
        assert folder_with_segs_from_prev_stage is not None, \
            f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
            f'stage ({predictor.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
            f' they are located via folder_with_segs_from_prev_stage'

    # Manage input and output file lists
    from ..io.file_manager import manage_input_and_output_lists
    list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
        manage_input_and_output_lists(predictor,
                                     list_of_lists_or_source_folder,
                                     output_folder_or_list_of_truncated_output_files,
                                     folder_with_segs_from_prev_stage,
                                     overwrite,
                                     0, 1,  # no distributed processing
                                     save_probabilities)

    if len(list_of_lists_or_source_folder) == 0:
        return

    # Setup preprocessing
    label_manager = predictor.plans_manager.get_label_manager(predictor.dataset_json)
    preprocessor = DefaultPreprocessor(verbose=predictor.verbose)

    # Handle None lists
    if output_filename_truncated is None:
        output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
    if seg_from_prev_stage_files is None:
        seg_from_prev_stage_files = [None] * len(list_of_lists_or_source_folder)

    # Process files sequentially
    ret = []
    for li, of, sps in zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files):
        # Preprocess
        data, seg, data_properties = preprocessor.run_case(
            li,
            sps,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json
        )

        print(f'perform_everything_on_device: {predictor.perform_everything_on_device}')

        # Predict
        from .logits import predict_logits_from_preprocessed_data
        prediction = predict_logits_from_preprocessed_data(
            predictor,
            torch.from_numpy(data)
        ).cpu()

        # Export or return
        if of is not None:
            export_prediction_from_logits(
                prediction,
                data_properties,
                predictor.configuration_manager,
                predictor.plans_manager,
                predictor.dataset_json,
                of,
                save_probabilities
            )
        else:
            ret.append(
                convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction,
                    predictor.plans_manager,
                    predictor.configuration_manager,
                    label_manager,
                    data_properties,
                    save_probabilities
                )
            )

    # Clear caches
    compute_gaussian.cache_clear()
    empty_cache(predictor.device)

    return ret


