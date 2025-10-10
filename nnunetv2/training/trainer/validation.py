"""
Full validation inference and metrics computation.

This module handles:
- Full validation with sliding window inference
- Metrics computation and export
"""

import multiprocessing
import warnings
from time import sleep

import numpy as np
import torch
from torch import distributed as dist
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.experiment_planning.config.defaults import DEFAULT_NUM_PROCESSES
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.predictor.postprocessing.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predictor.main import nnUNetPredictor
from nnunetv2.inference.predictor.prediction.sliding_window_utils import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.data.dataset import infer_dataset_class
from nnunetv2.data.dataset_io.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.experiment_planning.planning.label_handling import convert_labelmap_to_one_hot


def perform_actual_validation(trainer_instance, save_probabilities: bool = False):
    """
    Perform full validation with inference on validation set.

    This method handles:
    - Setting up predictor with current network state
    - Running inference on validation cases
    - Computing validation metrics
    - Exporting predictions for next stages if needed

    Args:
        trainer_instance: The nnUNetTrainer instance
        save_probabilities: Whether to save prediction probabilities
    """
    from nnunetv2.architecture import set_deep_supervision_enabled, _do_i_compile

    set_deep_supervision_enabled(trainer_instance, False)
    trainer_instance.network.eval()

    if (trainer_instance.is_ddp and trainer_instance.batch_size == 1 and
        trainer_instance.enable_deep_supervision and _do_i_compile(trainer_instance)):
        trainer_instance.print_to_log_file(
            "WARNING! batch size is 1 during training and torch.compile is enabled. If you "
            "encounter crashes in validation then this is because torch.compile forgets "
            "to trigger a recompilation of the model with deep supervision disabled. "
            "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
            "validation with --val (exactly the same as before) and then it will work. "
            "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
            "forward pass (where compile is triggered) already has deep supervision disabled. "
            "This is exactly what we need in perform_actual_validation")

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=trainer_instance.device,
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(trainer_instance.network, trainer_instance.plans_manager,
                                   trainer_instance.configuration_manager, None,
                                   trainer_instance.dataset_json, trainer_instance.__class__.__name__,
                                   trainer_instance.inference_allowed_mirroring_axes)

    with multiprocessing.get_context("spawn").Pool(DEFAULT_NUM_PROCESSES) as segmentation_export_pool:
        worker_list = [i for i in segmentation_export_pool._pool]
        validation_output_folder = join(trainer_instance.output_folder, 'validation')
        maybe_mkdir_p(validation_output_folder)

        # we cannot use trainer_instance.get_tr_and_val_datasets() here because we might be DDP
        # and then we have to distribute the validation keys across the workers.
        _, val_keys = trainer_instance.do_split()
        if trainer_instance.is_ddp:
            last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
            val_keys = val_keys[trainer_instance.local_rank:: dist.get_world_size()]
            # we cannot just have barriers all over the place because the number of keys each GPU receives can be different

        dataset_val = trainer_instance.dataset_class(
            trainer_instance.preprocessed_dataset_folder, val_keys,
            folder_with_segs_from_previous_stage=trainer_instance.folder_with_segs_from_previous_stage)

        next_stages = trainer_instance.configuration_manager.next_stage_names

        if next_stages is not None:
            _ = [maybe_mkdir_p(join(trainer_instance.output_folder_base, 'predicted_next_stage', n))
                for n in next_stages]

        results = []

        for i, k in enumerate(dataset_val.identifiers):
            proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                       allowed_num_queued=2)
            while not proceed:
                sleep(0.1)
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)

            trainer_instance.print_to_log_file(f"predicting {k}")
            data, _, seg_prev, properties = dataset_val.load_case(k)

            # we do [:] to convert blosc2 to numpy
            data = data[:]

            if trainer_instance.is_cascaded:
                seg_prev = seg_prev[:]
                data = np.vstack((data, convert_labelmap_to_one_hot(
                    seg_prev, trainer_instance.label_manager.foreground_labels,
                    output_dtype=data.dtype)))

            with warnings.catch_warnings():
                # ignore 'The given NumPy array is not writable' warning
                warnings.simplefilter("ignore")
                data = torch.from_numpy(data)

            trainer_instance.print_to_log_file(f'{k}, shape {data.shape}, rank {trainer_instance.local_rank}')
            output_filename_truncated = join(validation_output_folder, k)

            prediction = predictor.predict_sliding_window_return_logits(data)
            prediction = prediction.cpu()

            # this needs to go into background processes
            results.append(
                segmentation_export_pool.starmap_async(
                    export_prediction_from_logits, (
                        (prediction, properties, trainer_instance.configuration_manager,
                         trainer_instance.plans_manager, trainer_instance.dataset_json,
                         output_filename_truncated, save_probabilities),
                    )
                )
            )

            # if needed, export the softmax prediction for the next stage
            if next_stages is not None:
                for n in next_stages:
                    next_stage_config_manager = trainer_instance.plans_manager.get_configuration(n)
                    expected_preprocessed_folder = join(nnUNet_preprocessed,
                                                       trainer_instance.plans_manager.dataset_name,
                                                       next_stage_config_manager.data_identifier)
                    # next stage may have a different dataset class, do not use trainer_instance.dataset_class
                    dataset_class = infer_dataset_class(expected_preprocessed_folder)

                    try:
                        # we do this so that we can use load_case and do not have to hard code
                        # how loading training cases is implemented
                        tmp = dataset_class(expected_preprocessed_folder, [k])
                        d, _, _, _ = tmp.load_case(k)
                    except FileNotFoundError:
                        trainer_instance.print_to_log_file(
                            f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                            f"Run the preprocessing for this configuration first!")
                        continue

                    target_shape = d.shape[1:]
                    output_folder = join(trainer_instance.output_folder_base, 'predicted_next_stage', n)
                    output_file_truncated = join(output_folder, k)

                    results.append(segmentation_export_pool.starmap_async(
                        resample_and_save, (
                            (prediction, target_shape, output_file_truncated, trainer_instance.plans_manager,
                             trainer_instance.configuration_manager, properties, trainer_instance.dataset_json,
                             DEFAULT_NUM_PROCESSES, dataset_class),
                        )
                    ))

            # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
            if trainer_instance.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                dist.barrier()

        _ = [r.get() for r in results]

    if trainer_instance.is_ddp:
        dist.barrier()

    if trainer_instance.local_rank == 0:
        metrics = compute_metrics_on_folder(
            join(trainer_instance.preprocessed_dataset_folder_base, 'gt_segmentations'),
            validation_output_folder,
            join(validation_output_folder, 'summary.json'),
            trainer_instance.plans_manager.image_reader_writer_class(),
            trainer_instance.dataset_json["file_ending"],
            trainer_instance.label_manager.foreground_regions if trainer_instance.label_manager.has_regions else
            trainer_instance.label_manager.foreground_labels,
            trainer_instance.label_manager.ignore_label, chill=True,
            num_processes=DEFAULT_NUM_PROCESSES * dist.get_world_size() if
            trainer_instance.is_ddp else DEFAULT_NUM_PROCESSES)
        trainer_instance.print_to_log_file("Validation complete", also_print_to_console=True)
        trainer_instance.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                           also_print_to_console=True)

    set_deep_supervision_enabled(trainer_instance, True)
    compute_gaussian.cache_clear()