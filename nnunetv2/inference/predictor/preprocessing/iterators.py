"""Data iterators for preprocessing in nnUNet predictor."""

from typing import List, Union
import numpy as np
from .data_iterators import (
    preprocessing_iterator_fromfiles,
    preprocessing_iterator_fromnpy
)


def get_data_iterator_from_lists_of_filenames(predictor,
                                             input_list_of_lists: List[List[str]],
                                             seg_from_prev_stage_files: Union[List[str], None],
                                             output_filenames_truncated: Union[List[str], None],
                                             num_processes: int):
    """
    Get data iterator for preprocessing files.

    Args:
        predictor: The nnUNetPredictor instance
        input_list_of_lists: List of input file lists
        seg_from_prev_stage_files: Previous stage segmentation files
        output_filenames_truncated: Truncated output filenames
        num_processes: Number of preprocessing processes

    Returns:
        Data iterator for preprocessing
    """
    return preprocessing_iterator_fromfiles(
        input_list_of_lists,
        seg_from_prev_stage_files,
        output_filenames_truncated,
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        num_processes,
        predictor.device.type == 'cuda',
        predictor.verbose_preprocessing
    )


def get_data_iterator_from_raw_npy_data(predictor,
                                       image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                       segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None, np.ndarray, List[np.ndarray]],
                                       properties_or_list_of_properties: Union[dict, List[dict]],
                                       truncated_ofname: Union[str, List[str], None],
                                       num_processes: int = 3):
    """
    Get data iterator for numpy arrays.

    Args:
        predictor: The nnUNetPredictor instance
        image_or_list_of_images: Input image(s) as numpy array(s)
        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Previous stage segmentations
        properties_or_list_of_properties: Image properties
        truncated_ofname: Truncated output filename(s)
        num_processes: Number of preprocessing processes

    Returns:
        Data iterator for preprocessing numpy arrays
    """
    # Ensure all inputs are lists
    list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
        image_or_list_of_images

    if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
        segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
            segs_from_prev_stage_or_list_of_segs_from_prev_stage]

    if isinstance(truncated_ofname, str):
        truncated_ofname = [truncated_ofname]

    if isinstance(properties_or_list_of_properties, dict):
        properties_or_list_of_properties = [properties_or_list_of_properties]

    num_processes = min(num_processes, len(list_of_images))

    return preprocessing_iterator_fromnpy(
        list_of_images,
        segs_from_prev_stage_or_list_of_segs_from_prev_stage,
        properties_or_list_of_properties,
        truncated_ofname,
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        num_processes,
        predictor.device.type == 'cuda',
        predictor.verbose_preprocessing
    )