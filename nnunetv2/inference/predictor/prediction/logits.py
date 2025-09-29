"""Logit prediction from preprocessed data for nnUNet predictor."""

import torch
from torch._dynamo import OptimizedModule
from nnunetv2.configuration import default_num_processes


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
    torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
    prediction = None

    for params in predictor.list_of_parameters:
        # Load model parameters
        if not isinstance(predictor.network, OptimizedModule):
            predictor.network.load_state_dict(params)
        else:
            predictor.network._orig_mod.load_state_dict(params)

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