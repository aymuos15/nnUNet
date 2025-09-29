"""Batch prediction for nnUNet predictor."""

import os
import multiprocessing
from time import sleep
from typing import Optional

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    convert_predicted_logits_to_segmentation_with_correct_shape
)
from nnunetv2.inference.sliding_window_prediction import compute_gaussian


def predict_from_data_iterator(predictor,
                              data_iterator,
                              save_probabilities: bool = False,
                              num_processes_segmentation_export: int = default_num_processes):
    """
    Predict from a data iterator (batch prediction).

    Each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
    If 'ofile' is None, the result will be returned instead of written to a file

    Args:
        predictor: The nnUNetPredictor instance
        data_iterator: Iterator providing preprocessed data
        save_probabilities: Whether to save probability maps
        num_processes_segmentation_export: Number of processes for export

    Returns:
        List of predictions (if not saving to files)
    """
    with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
        worker_list = [i for i in export_pool._pool]
        r = []

        for preprocessed in data_iterator:
            data = preprocessed['data']

            # Handle data that was saved to disk
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed['ofile']
            if ofile is not None:
                print(f'\nPredicting {os.path.basename(ofile)}:')
            else:
                print(f'\nPredicting image of shape {data.shape}:')

            print(f'perform_everything_on_device: {predictor.perform_everything_on_device}')

            properties = preprocessed['data_properties']

            # Prevent GPU from getting too far ahead of disk I/O
            proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
            while not proceed:
                sleep(0.1)
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

            # Predict and convert to numpy for multiprocessing serialization
            from .logits import predict_logits_from_preprocessed_data
            prediction = predict_logits_from_preprocessed_data(predictor, data).cpu().detach().numpy()

            # Send to background worker for postprocessing
            if ofile is not None:
                print('sending off prediction to background worker for resampling and export')
                r.append(
                    export_pool.starmap_async(
                        export_prediction_from_logits,
                        ((prediction, properties, predictor.configuration_manager, predictor.plans_manager,
                          predictor.dataset_json, ofile, save_probabilities),)
                    )
                )
            else:
                print('sending off prediction to background worker for resampling')
                r.append(
                    export_pool.starmap_async(
                        convert_predicted_logits_to_segmentation_with_correct_shape, (
                            (prediction, predictor.plans_manager,
                             predictor.configuration_manager, predictor.label_manager,
                             properties,
                             save_probabilities),)
                    )
                )

            if ofile is not None:
                print(f'done with {os.path.basename(ofile)}')
            else:
                print(f'\nDone with image of shape {data.shape}:')

        # Collect results
        ret = [i.get()[0] for i in r]

    # Cleanup
    if isinstance(data_iterator, MultiThreadedAugmenter):
        data_iterator._finish()

    # Clear caches
    compute_gaussian.cache_clear()
    empty_cache(predictor.device)

    return ret