"""Logit prediction from preprocessed data for nnUNet predictor."""

import torch
from torch._dynamo import OptimizedModule
from nnunetv2.experiment_planning.config.defaults import DEFAULT_NUM_PROCESSES


@torch.inference_mode()
def predict_logits_from_preprocessed_data(predictor, data: torch.Tensor) -> torch.Tensor:
    """
    Predict logits from preprocessed data using ensemble of models.

    IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
    TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

    RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
    SEE convert_predicted_logits_to_segmentation_with_correct_shape

    Args:
        predictor: The nnUNetPredictor instance
        data: Preprocessed input data tensor

    Returns:
        Predicted logits tensor
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(DEFAULT_NUM_PROCESSES if DEFAULT_NUM_PROCESSES < n_threads else n_threads)
    prediction = None

    for params in predictor.list_of_parameters:
        # Load model parameters
        # Use strict=False to handle deep supervision heads mismatch between training and inference
        if not isinstance(predictor.network, OptimizedModule):
            predictor.network.load_state_dict(params, strict=False)
        else:
            predictor.network._orig_mod.load_state_dict(params, strict=False)

        # Perform prediction
        # Note: We move to CPU after each prediction to avoid OOM in ensemble predictions
        from .sliding_window import predict_sliding_window_return_logits
        if prediction is None:
            prediction = predict_sliding_window_return_logits(predictor, data).to('cpu')
        else:
            prediction += predict_sliding_window_return_logits(predictor, data).to('cpu')

    # Average predictions if using ensemble
    if len(predictor.list_of_parameters) > 1:
        prediction /= len(predictor.list_of_parameters)

    if predictor.verbose:
        print('Prediction done')

    torch.set_num_threads(n_threads)
    return prediction