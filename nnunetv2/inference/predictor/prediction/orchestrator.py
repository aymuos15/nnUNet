"""Orchestration of prediction workflows for nnUNet predictor."""

import os
from typing import Union, List

from nnunetv2.configuration import default_num_processes
from ..utils.config_saver import save_prediction_args_and_dataset


def predict_from_files(predictor,
                       list_of_lists_or_source_folder: Union[str, List[List[str]]],
                       output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                       save_probabilities: bool = False,
                       overwrite: bool = True,
                       num_processes_preprocessing: int = default_num_processes,
                       num_processes_segmentation_export: int = default_num_processes,
                       folder_with_segs_from_prev_stage: str = None,
                       num_parts: int = 1,
                       part_id: int = 0):
    """
    Main prediction function for batch predictions from files.

    This is nnU-Net's default function for making predictions. It works best for batch predictions
    (predicting many images at once).

    Args:
        predictor: The nnUNetPredictor instance
        list_of_lists_or_source_folder: Input files or folder path
        output_folder_or_list_of_truncated_output_files: Output location
        save_probabilities: Whether to save probability maps
        overwrite: Whether to overwrite existing predictions
        num_processes_preprocessing: Number of preprocessing processes
        num_processes_segmentation_export: Number of export processes
        folder_with_segs_from_prev_stage: Previous stage segmentations folder
        num_parts: Total number of parts for distributed processing
        part_id: Part ID for distributed processing

    Returns:
        List of predictions
    """
    assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                  "So if there are 3 parts then valid part IDs are 0, 1, 2")

    # Determine output folder
    if isinstance(output_folder_or_list_of_truncated_output_files, str):
        output_folder = output_folder_or_list_of_truncated_output_files
    elif isinstance(output_folder_or_list_of_truncated_output_files, list):
        output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
    else:
        output_folder = None

    # Save configuration and arguments for reproducibility
    save_prediction_args_and_dataset(predictor, output_folder, predict_from_files, locals())

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
                                     part_id,
                                     num_parts,
                                     save_probabilities)

    if len(list_of_lists_or_source_folder) == 0:
        return

    # Get data iterator
    data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        list_of_lists_or_source_folder,
        seg_from_prev_stage_files,
        output_filename_truncated,
        num_processes_preprocessing
    )

    # Perform predictions
    from .batch import predict_from_data_iterator
    return predict_from_data_iterator(predictor, data_iterator, save_probabilities, num_processes_segmentation_export)


def predict_from_list_of_npy_arrays(predictor,
                                   image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                   segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None, np.ndarray, List[np.ndarray]],
                                   properties_or_list_of_properties: Union[dict, List[dict]],
                                   truncated_ofname: Union[str, List[str], None],
                                   num_processes: int = default_num_processes,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
    """
    Predict from numpy arrays.

    Args:
        predictor: The nnUNetPredictor instance
        image_or_list_of_images: Input image(s) as numpy array(s)
        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Previous stage segmentations
        properties_or_list_of_properties: Image properties
        truncated_ofname: Truncated output filename(s)
        num_processes: Number of preprocessing processes
        save_probabilities: Whether to save probability maps
        num_processes_segmentation_export: Number of export processes

    Returns:
        List of predictions
    """
    import numpy as np

    # Get data iterator for numpy arrays
    data_iterator = predictor.get_data_iterator_from_raw_npy_data(
        image_or_list_of_images,
        segs_from_prev_stage_or_list_of_segs_from_prev_stage,
        properties_or_list_of_properties,
        truncated_ofname,
        num_processes
    )

    # Perform predictions
    from .batch import predict_from_data_iterator
    return predict_from_data_iterator(predictor, data_iterator, save_probabilities, num_processes_segmentation_export)


