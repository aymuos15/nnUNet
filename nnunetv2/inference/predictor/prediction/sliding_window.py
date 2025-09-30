"""Sliding window prediction for nnUNet predictor."""

import itertools
from typing import Tuple, Optional
from queue import Queue
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

from nnunetv2.experiment_planning.config.defaults import DEFAULT_NUM_PROCESSES
from .sliding_window_utils import compute_steps_for_sliding_window, compute_gaussian
from nnunetv2.utilities.helpers import empty_cache


def get_sliding_window_slicers(predictor, image_size: Tuple[int, ...]):
    """
    Generate slicers for sliding window prediction.

    Args:
        predictor: The nnUNetPredictor instance
        image_size: Size of the input image

    Returns:
        List of slice tuples for sliding window
    """
    slicers = []
    if len(predictor.configuration_manager.patch_size) < len(image_size):
        assert len(predictor.configuration_manager.patch_size) == len(
            image_size) - 1, 'if tile_size has less entries than image_size, ' \
                             'len(tile_size) ' \
                             'must be one shorter than len(image_size) ' \
                             '(only dimension ' \
                             'discrepancy of 1 allowed).'
        steps = compute_steps_for_sliding_window(image_size[1:], predictor.configuration_manager.patch_size,
                                                 predictor.tile_step_size)
        if predictor.verbose:
            print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                  f' {image_size}, tile_size {predictor.configuration_manager.patch_size}, '
                  f'tile_step_size {predictor.tile_step_size}\nsteps:\n{steps}')
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicers.append(
                        tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                 zip((sx, sy), predictor.configuration_manager.patch_size)]]))
    else:
        steps = compute_steps_for_sliding_window(image_size, predictor.configuration_manager.patch_size,
                                                 predictor.tile_step_size)
        if predictor.verbose:
            print(f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, '
                  f'tile_size {predictor.configuration_manager.patch_size}, '
                  f'tile_step_size {predictor.tile_step_size}\nsteps:\n{steps}')
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                              zip((sx, sy, sz), predictor.configuration_manager.patch_size)]]))
    return slicers


@torch.inference_mode()
def predict_sliding_window_return_logits(predictor, input_image: torch.Tensor) -> torch.Tensor:
    """
    Predict using sliding window and return logits.

    Args:
        predictor: The nnUNetPredictor instance
        input_image: Input image tensor

    Returns:
        Predicted logits tensor
    """
    assert input_image.ndim == 4, 'input_image must be c x x y z'

    window_size = predictor.configuration_manager.patch_size

    # Pad if necessary
    if predictor.verbose:
        print(f'Input shape: {input_image.shape}')
    need_to_pad = np.any(np.array(input_image.shape[1:]) - np.array(window_size) < 0)

    if need_to_pad:
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        input_image = pad_nd_image(input_image.numpy(), window_size, 'constant', kwargs={'constant_values': 0},
                                   return_slicer=False)
        input_image = torch.from_numpy(input_image)

    slicers = get_sliding_window_slicers(predictor, input_image.shape[1:])

    if predictor.perform_everything_on_device and input_image.numel() < 2e9:
        if predictor.verbose:
            print('Running everything on GPU')
        predicted_logits = _internal_predict_sliding_window_return_logits(
            predictor, input_image, slicers, predictor.perform_everything_on_device
        )
    else:
        if predictor.verbose:
            print('Running on CPU+GPU')
        predicted_logits = _internal_predict_sliding_window_return_logits(
            predictor, input_image, slicers, False
        )

    empty_cache(predictor.device)
    return predicted_logits


@torch.inference_mode()
def _internal_predict_sliding_window_return_logits(predictor,
                                                  data: torch.Tensor,
                                                  slicers,
                                                  do_on_device: bool = True):
    """
    Internal method for sliding window prediction.

    Args:
        predictor: The nnUNetPredictor instance
        data: Input data tensor
        slicers: List of slice tuples
        do_on_device: Whether to perform operations on device

    Returns:
        Predicted logits tensor
    """
    predicted_logits = n_predictions = prediction = gaussian = workon = None
    results_device = predictor.device if do_on_device else torch.device('cpu')

    def producer(d, slh, q):
        for s in slh:
            q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(predictor.device), s))
        q.put('end')

    try:
        empty_cache(predictor.device)

        # Move data to device
        if predictor.verbose:
            print(f'move image to device {results_device}')
        data = data.to(results_device)
        queue = Queue(maxsize=2)
        t = Thread(target=producer, args=(data, slicers, queue))
        t.start()

        # Preallocate arrays
        if predictor.verbose:
            print(f'preallocating results arrays on device {results_device}')
        predicted_logits = torch.zeros((predictor.label_manager.num_segmentation_heads, *data.shape[1:]),
                                       dtype=torch.half,
                                       device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

        if predictor.use_gaussian:
            gaussian = compute_gaussian(tuple(predictor.configuration_manager.patch_size), sigma_scale=1. / 8,
                                        value_scaling_factor=10,
                                        device=results_device)
        else:
            gaussian = 1

        if not predictor.allow_tqdm and predictor.verbose:
            print(f'running prediction: {len(slicers)} steps')

        with tqdm(desc=None, total=len(slicers), disable=not predictor.allow_tqdm) as pbar:
            while True:
                item = queue.get()
                if item == 'end':
                    queue.task_done()
                    break
                workon, sl = item

                # Use mirroring module for prediction
                from .mirroring import maybe_mirror_and_predict
                prediction = maybe_mirror_and_predict(predictor, workon)[0].to(results_device)

                if predictor.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian
                queue.task_done()
                pbar.update()
        queue.join()

        predicted_logits /= n_predictions
        if not predictor.allow_tqdm and predictor.verbose:
            print('done')

        if workon is not None:
            workon = workon.detach()
        if prediction is not None:
            prediction = prediction.detach()
        data = data.detach()
        t.join()

    except Exception as e:
        if workon is not None:
            workon = workon.detach()
        if prediction is not None:
            prediction = prediction.detach()
        data = data.detach()
        t.join()
        raise e

    return predicted_logits