"""
Full validation method extracted from nnUNetTrainer.py
Contains perform_actual_validation method for complete validation inference.
"""

import multiprocessing
import warnings
from time import sleep

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from torch import distributed as dist

from nnunetv2.configuration import default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot


def perform_actual_validation(self, save_probabilities: bool = False):
    """
    Perform actual validation with full inference pipeline.

    This method:
    1. Sets the network to evaluation mode and disables deep supervision
    2. Creates a predictor for sliding window inference
    3. Processes each validation case through the full inference pipeline
    4. Exports predictions and computes metrics

    Args:
        save_probabilities: Whether to save prediction probabilities
    """
    self.set_deep_supervision_enabled(False)
    self.network.eval()

    if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
        self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                               "encounter crashes in validation then this is because torch.compile forgets "
                               "to trigger a recompilation of the model with deep supervision disabled. "
                               "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                               "validation with --val (exactly the same as before) and then it will work. "
                               "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                               "forward pass (where compile is triggered) already has deep supervision disabled. "
                               "This is exactly what we need in perform_actual_validation")

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=self.device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                    self.dataset_json, self.__class__.__name__,
                                    self.inference_allowed_mirroring_axes)

    with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
        worker_list = [i for i in segmentation_export_pool._pool]
        validation_output_folder = join(self.output_folder, 'validation')
        maybe_mkdir_p(validation_output_folder)

        # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
        # the validation keys across the workers.
        _, val_keys = self.do_split()
        if self.is_ddp:
            last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

            val_keys = val_keys[self.local_rank:: dist.get_world_size()]
            # we cannot just have barriers all over the place because the number of keys each GPU receives can be
            # different

        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

        next_stages = self.configuration_manager.next_stage_names

        if next_stages is not None:
            _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

        results = []

        for i, k in enumerate(dataset_val.identifiers):
            proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                       allowed_num_queued=2)
            while not proceed:
                sleep(0.1)
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)

            self.print_to_log_file(f"predicting {k}")
            data, _, seg_prev, properties = dataset_val.load_case(k)

            # we do [:] to convert blosc2 to numpy
            data = data[:]

            if self.is_cascaded:
                seg_prev = seg_prev[:]
                data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                    output_dtype=data.dtype)))
            with warnings.catch_warnings():
                # ignore 'The given NumPy array is not writable' warning
                warnings.simplefilter("ignore")
                data = torch.from_numpy(data)

            self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
            output_filename_truncated = join(validation_output_folder, k)

            prediction = predictor.predict_sliding_window_return_logits(data)
            prediction = prediction.cpu()

            # this needs to go into background processes
            results.append(
                segmentation_export_pool.starmap_async(
                    export_prediction_from_logits, (
                        (prediction, properties, self.configuration_manager, self.plans_manager,
                         self.dataset_json, output_filename_truncated, save_probabilities),
                    )
                )
            )
            # for debug purposes
            # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
            #      self.dataset_json, output_filename_truncated, save_probabilities)

            # if needed, export the softmax prediction for the next stage
            if next_stages is not None:
                for n in next_stages:
                    next_stage_config_manager = self.plans_manager.get_configuration(n)
                    expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                        next_stage_config_manager.data_identifier)
                    # next stage may have a different dataset class, do not use self.dataset_class
                    dataset_class = infer_dataset_class(expected_preprocessed_folder)

                    try:
                        # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                        tmp = dataset_class(expected_preprocessed_folder, [k])
                        d, _, _, _ = tmp.load_case(k)
                    except FileNotFoundError:
                        self.print_to_log_file(
                            f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                            f"Run the preprocessing for this configuration first!")
                        continue

                    target_shape = d.shape[1:]
                    output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                    output_file_truncated = join(output_folder, k)

                    # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                    #                   self.dataset_json)
                    results.append(segmentation_export_pool.starmap_async(
                        resample_and_save, (
                            (prediction, target_shape, output_file_truncated, self.plans_manager,
                             self.configuration_manager,
                             properties,
                             self.dataset_json,
                             default_num_processes,
                             dataset_class),
                        )
                    ))
            # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
            if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                dist.barrier()

        _ = [r.get() for r in results]

    if self.is_ddp:
        dist.barrier()

    if self.local_rank == 0:
        metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                            validation_output_folder,
                                            join(validation_output_folder, 'summary.json'),
                                            self.plans_manager.image_reader_writer_class(),
                                            self.dataset_json["file_ending"],
                                            self.label_manager.foreground_regions if self.label_manager.has_regions else
                                            self.label_manager.foreground_labels,
                                            self.label_manager.ignore_label, chill=True,
                                            num_processes=default_num_processes * dist.get_world_size() if
                                            self.is_ddp else default_num_processes)
        self.print_to_log_file("Validation complete", also_print_to_console=True)
        self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                               also_print_to_console=True)

    self.set_deep_supervision_enabled(True)
    compute_gaussian.cache_clear()