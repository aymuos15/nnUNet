"""File and folder management for nnUNet predictor."""

import os
from typing import Union, List, Tuple, Optional
from batchgenerators.utilities.file_and_folder_operations import join, isfile
from nnunetv2.data.dataset_io.utils import create_lists_from_splitted_dataset_folder


def manage_input_and_output_lists(predictor,
                                 list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                 output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                 folder_with_segs_from_prev_stage: str = None,
                                 overwrite: bool = True,
                                 part_id: int = 0,
                                 num_parts: int = 1,
                                 save_probabilities: bool = False) -> Tuple[List, List, List]:
    """
    Manage input and output file lists for batch prediction.

    Args:
        predictor: The nnUNetPredictor instance
        list_of_lists_or_source_folder: Input files or folder path
        output_folder_or_list_of_truncated_output_files: Output location
        folder_with_segs_from_prev_stage: Previous stage segmentations folder
        overwrite: Whether to overwrite existing predictions
        part_id: Part ID for distributed processing
        num_parts: Total number of parts for distributed processing
        save_probabilities: Whether to save probability maps

    Returns:
        Tuple of (input_files, output_files, seg_from_prev_stage_files)
    """
    # Convert source folder to list of lists if needed
    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(
            list_of_lists_or_source_folder,
            predictor.dataset_json['file_ending']
        )

    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')

    # Handle distributed processing
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]

    # Extract case IDs
    caseids = [os.path.basename(i[0])[:-(len(predictor.dataset_json['file_ending']) + 5)]
               for i in list_of_lists_or_source_folder]

    print(f'I am processing {part_id} out of {num_parts} '
          f'(max process ID is {num_parts - 1}, we start counting with 0!)')
    print(f'There are {len(caseids)} cases that I would like to predict')

    # Prepare output filenames
    if isinstance(output_folder_or_list_of_truncated_output_files, str):
        output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
    elif isinstance(output_folder_or_list_of_truncated_output_files, list):
        output_filename_truncated = output_folder_or_list_of_truncated_output_files[part_id::num_parts]
    else:
        output_filename_truncated = None

    # Prepare segmentation files from previous stage
    seg_from_prev_stage_files = [
        join(folder_with_segs_from_prev_stage, i + predictor.dataset_json['file_ending'])
        if folder_with_segs_from_prev_stage is not None else None
        for i in caseids
    ]

    # Remove already predicted files if not overwriting
    if not overwrite and output_filename_truncated is not None:
        output_filename_truncated, list_of_lists_or_source_folder, seg_from_prev_stage_files = \
            _filter_existing_predictions(
                output_filename_truncated,
                list_of_lists_or_source_folder,
                seg_from_prev_stage_files,
                predictor.dataset_json['file_ending'],
                save_probabilities,
                overwrite
            )

    return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files


def _filter_existing_predictions(output_filename_truncated: List[str],
                                list_of_lists_or_source_folder: List[List[str]],
                                seg_from_prev_stage_files: List[Optional[str]],
                                file_ending: str,
                                save_probabilities: bool,
                                overwrite: bool) -> Tuple[List, List, List]:
    """
    Filter out already predicted files when overwrite is False.

    Args:
        output_filename_truncated: List of output filenames
        list_of_lists_or_source_folder: List of input file lists
        seg_from_prev_stage_files: List of previous stage segmentation files
        file_ending: File extension for predictions
        save_probabilities: Whether probability maps are being saved
        overwrite: Overwrite setting (for logging)

    Returns:
        Filtered tuple of (output_files, input_files, prev_stage_files)
    """
    # Check which files already exist
    tmp = [isfile(i + file_ending) for i in output_filename_truncated]

    if save_probabilities:
        tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
        tmp = [i and j for i, j in zip(tmp, tmp2)]

    not_existing_indices = [i for i, j in enumerate(tmp) if not j]

    # Filter all lists
    output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
    list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
    seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]

    print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
          f'That\'s {len(not_existing_indices)} cases.')

    return output_filename_truncated, list_of_lists_or_source_folder, seg_from_prev_stage_files