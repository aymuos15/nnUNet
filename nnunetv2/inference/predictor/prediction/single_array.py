"""Single array prediction for nnUNet predictor."""

import numpy as np
import torch
from copy import deepcopy
from typing import Optional

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor


def predict_single_npy_array(predictor,
                            input_image: np.ndarray,
                            image_properties: dict,
                            segmentation_previous_stage: Optional[np.ndarray] = None,
                            output_file_truncated: Optional[str] = None,
                            save_or_return_probabilities: bool = False):
    """
    Predict single numpy array.

    Args:
        predictor: The nnUNetPredictor instance
        input_image: Input image as numpy array
        image_properties: Properties of the input image
        segmentation_previous_stage: Segmentation from previous stage (for cascades)
        output_file_truncated: Truncated output filename
        save_or_return_probabilities: Whether to save/return probability maps

    Returns:
        Tuple of (segmentation, probabilities) if returning probabilities, else just segmentation
    """
    # Preprocess the input
    preprocessor = DefaultPreprocessor(verbose=predictor.verbose_preprocessing)
    data, seg = preprocessor.run_case_npy(
        input_image,
        segmentation_previous_stage,
        image_properties,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json
    )

    # Move to correct device
    data = torch.from_numpy(data).to(dtype=torch.float32)

    # Predict logits
    from .logits import predict_logits_from_preprocessed_data
    predicted_logits = predict_logits_from_preprocessed_data(predictor, data)

    # Convert to segmentation
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.label_manager,
        image_properties,
        save_or_return_probabilities
    )

    # Export if output file specified
    if output_file_truncated is not None:
        from nnunetv2.inference.export_prediction import export_prediction_from_logits
        export_prediction_from_logits(
            predicted_logits,
            image_properties,
            predictor.configuration_manager,
            predictor.plans_manager,
            predictor.dataset_json,
            output_file_truncated,
            save_or_return_probabilities
        )

    # Handle probability output
    if save_or_return_probabilities and output_file_truncated is None:
        probabilities = predicted_logits
    else:
        probabilities = None

    if probabilities is not None:
        return segmentation, probabilities
    else:
        return segmentation