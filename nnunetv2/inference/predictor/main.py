"""Main nnUNet predictor class with delegation pattern."""

import torch
from torch import nn
from typing import Union, List, Tuple, Optional
import numpy as np

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class nnUNetPredictor(object):
    """
    Main predictor class for nnU-Net inference.

    This class follows a delegation pattern where functionality is organized into
    specialized modules for better maintainability and testability.
    """

    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        """
        Initialize the nnUNet predictor.

        Args:
            tile_step_size: Step size for sliding window (0.5 = 50% overlap)
            use_gaussian: Use Gaussian weighting for sliding window
            use_mirroring: Use test-time mirroring augmentation
            perform_everything_on_device: Keep everything on GPU if possible
            device: Device to use for prediction
            verbose: Enable verbose output
            verbose_preprocessing: Enable verbose preprocessing output
            allow_tqdm: Allow tqdm progress bars
        """
        from .initialization.config import setup_predictor_state
        setup_predictor_state(self, tile_step_size, use_gaussian, use_mirroring,
                            perform_everything_on_device, device, verbose,
                            verbose_preprocessing, allow_tqdm)

    def initialize_from_trained_model_folder(self,
                                            model_training_output_dir: str,
                                            use_folds: Union[Tuple[Union[int, str]], None],
                                            checkpoint_name: str = 'checkpoint_final.pth'):
        """
        Initialize from a trained model folder.

        Args:
            model_training_output_dir: Path to the trained model folder
            use_folds: Which folds to use for prediction
            checkpoint_name: Name of the checkpoint file to load
        """
        from .initialization.model_loader import initialize_from_trained_model_folder
        initialize_from_trained_model_folder(self, model_training_output_dir, use_folds, checkpoint_name)

    def manual_initialization(self,
                            network: nn.Module,
                            plans_manager: PlansManager,
                            configuration_manager: ConfigurationManager,
                            parameters: Optional[List[dict]],
                            dataset_json: dict,
                            trainer_name: str,
                            inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        Manual initialization used by nnUNetTrainer for final validation.

        Args:
            network: Pre-built network
            plans_manager: PlansManager instance
            configuration_manager: ConfigurationManager instance
            parameters: Network parameters
            dataset_json: Dataset configuration
            trainer_name: Name of the trainer class
            inference_allowed_mirroring_axes: Allowed mirroring axes for inference
        """
        from .initialization.model_loader import manual_initialization
        manual_initialization(self, network, plans_manager, configuration_manager,
                            parameters, dataset_json, trainer_name,
                            inference_allowed_mirroring_axes)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        """
        Auto-detect available folds in the model training output directory.

        Args:
            model_training_output_dir: Path to the model training output directory
            checkpoint_name: Name of the checkpoint file

        Returns:
            List of available fold numbers
        """
        from .initialization.model_loader import auto_detect_available_folds
        return auto_detect_available_folds(model_training_output_dir, checkpoint_name)

    def _manage_input_and_output_lists(self,
                                      list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                      output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                      folder_with_segs_from_prev_stage: str = None,
                                      overwrite: bool = True,
                                      part_id: int = 0,
                                      num_parts: int = 1,
                                      save_probabilities: bool = False):
        """
        Manage input and output file lists for batch prediction.

        Returns:
            Tuple of (input_files, output_files, seg_from_prev_stage_files)
        """
        from .io.file_manager import manage_input_and_output_lists
        return manage_input_and_output_lists(self, list_of_lists_or_source_folder,
                                            output_folder_or_list_of_truncated_output_files,
                                            folder_with_segs_from_prev_stage, overwrite,
                                            part_id, num_parts, save_probabilities)

    def predict_from_files(self,
                          list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                          save_probabilities: bool = False,
                          overwrite: bool = True,
                          num_processes_preprocessing: int = 8,
                          num_processes_segmentation_export: int = 8,
                          folder_with_segs_from_prev_stage: str = None,
                          num_parts: int = 1,
                          part_id: int = 0):
        """
        Main prediction function for batch predictions from files.

        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).

        Returns:
            List of predictions
        """
        from .prediction.orchestrator import predict_from_files
        return predict_from_files(self, list_of_lists_or_source_folder,
                                 output_folder_or_list_of_truncated_output_files,
                                 save_probabilities, overwrite,
                                 num_processes_preprocessing,
                                 num_processes_segmentation_export,
                                 folder_with_segs_from_prev_stage,
                                 num_parts, part_id)

    def predict_from_files_sequential(self,
                                     list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                     output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                                     save_probabilities: bool = False,
                                     overwrite: bool = True,
                                     folder_with_segs_from_prev_stage: str = None):
        """
        Sequential prediction without multiprocessing. Slower but sometimes necessary.

        Returns:
            List of predictions
        """
        from .prediction.sequential import predict_from_files_sequential
        return predict_from_files_sequential(self, list_of_lists_or_source_folder,
                                            output_folder_or_list_of_truncated_output_files,
                                            save_probabilities, overwrite,
                                            folder_with_segs_from_prev_stage)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                           input_list_of_lists: List[List[str]],
                                                           seg_from_prev_stage_files: Union[List[str], None],
                                                           output_filenames_truncated: Union[List[str], None],
                                                           num_processes: int):
        """Get data iterator for preprocessing files."""
        from .preprocessing.data_iterators import preprocessing_iterator_fromfiles
        return preprocessing_iterator_fromfiles(
            input_list_of_lists,
            seg_from_prev_stage_files,
            output_filenames_truncated,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

    def get_data_iterator_from_raw_npy_data(self,
                                           image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                           segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None, np.ndarray, List[np.ndarray]],
                                           properties_or_list_of_properties: Union[dict, List[dict]],
                                           truncated_ofname: Union[str, List[str], None],
                                           num_processes: int = 3):
        """Get data iterator for numpy arrays."""
        from .preprocessing.data_iterators import preprocessing_iterator_fromnpy

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
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

    def predict_from_list_of_npy_arrays(self,
                                       image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                       segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None, np.ndarray, List[np.ndarray]],
                                       properties_or_list_of_properties: Union[dict, List[dict]],
                                       truncated_ofname: Union[str, List[str], None],
                                       num_processes: int = 8,
                                       save_probabilities: bool = False,
                                       num_processes_segmentation_export: int = 8):
        """
        Predict from numpy arrays.

        Returns:
            List of predictions
        """
        from .prediction.orchestrator import predict_from_list_of_npy_arrays
        return predict_from_list_of_npy_arrays(self, image_or_list_of_images,
                                              segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                              properties_or_list_of_properties,
                                              truncated_ofname, num_processes,
                                              save_probabilities,
                                              num_processes_segmentation_export)

    def predict_from_data_iterator(self,
                                  data_iterator,
                                  save_probabilities: bool = False,
                                  num_processes_segmentation_export: int = 8):
        """
        Predict from a data iterator (batch prediction).

        Each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file

        Returns:
            List of predictions (if not saving to files)
        """
        from .prediction.batch import predict_from_data_iterator
        return predict_from_data_iterator(self, data_iterator, save_probabilities,
                                         num_processes_segmentation_export)

    def predict_single_npy_array(self,
                                input_image: np.ndarray,
                                image_properties: dict,
                                segmentation_previous_stage: np.ndarray = None,
                                output_file_truncated: str = None,
                                save_or_return_probabilities: bool = False):
        """
        Predict single numpy array.

        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.

        Returns:
            Segmentation array or tuple of (segmentation, probabilities)
        """
        from .prediction.single_array import predict_single_npy_array
        return predict_single_npy_array(self, input_image, image_properties,
                                       segmentation_previous_stage,
                                       output_file_truncated,
                                       save_or_return_probabilities)

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predict logits from preprocessed data using ensemble of models.

        Returns:
            Predicted logits tensor
        """
        from .prediction.logits import predict_logits_from_preprocessed_data
        return predict_logits_from_preprocessed_data(self, data)

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:
        """
        Predict using sliding window and return logits.

        Args:
            input_image: Input image tensor (c x y z format)

        Returns:
            Predicted logits tensor
        """
        from .prediction.sliding_window import predict_sliding_window_return_logits
        return predict_sliding_window_return_logits(self, input_image)


