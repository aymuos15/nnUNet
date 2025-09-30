import multiprocessing
import os
from time import sleep
from typing import List, Type, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from tqdm import tqdm

from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.data.dataset_io.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.data.dataset_io.utils import get_filenames_of_train_images_and_targets
from .analyzers import analyze_case, aggregate_case_results, create_fingerprint_dict
from .collectors import calculate_global_intensity_statistics


class DatasetFingerprintExtractor(object):
    def __init__(self, dataset_name_or_id: Union[str, int], num_processes: int = 8, verbose: bool = False):
        """
        Extracts the dataset fingerprint used for experiment planning. The dataset fingerprint will be saved as a
        json file in the input_folder.

        Philosophy here is to do only what we really need. Don't store stuff that we can easily read from somewhere
        else. Don't compute stuff we don't need (except for intensity_statistics_per_channel).

        Args:
            dataset_name_or_id: Dataset name or ID
            num_processes: Number of processes for parallel processing
            verbose: Enable verbose output
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.verbose = verbose

        self.dataset_name = dataset_name
        self.input_folder = join(nnUNet_raw, dataset_name)
        self.num_processes = num_processes
        self.dataset_json = load_json(join(self.input_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.input_folder, self.dataset_json)

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)
        self.num_foreground_voxels_for_intensitystats = 10e7

    def _get_reader_writer_class(self):
        """Get the appropriate reader/writer class for this dataset."""
        return determine_reader_writer_from_dataset_json(
            self.dataset_json,
            # yikes. Rip the following line
            self.dataset[list(self.dataset.keys())[0]]['images'][0]
        )

    def _calculate_samples_per_case(self) -> int:
        """Calculate how many foreground voxels to sample per training case."""
        return int(self.num_foreground_voxels_for_intensitystats // len(self.dataset))

    def _process_cases_parallel(self, reader_writer_class, num_foreground_samples_per_case: int) -> List:
        """Process all cases in parallel using multiprocessing."""
        r = []
        with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
            for k in self.dataset.keys():
                r.append(p.starmap_async(analyze_case,
                                         ((self.dataset[k]['images'], self.dataset[k]['label'], reader_writer_class,
                                           num_foreground_samples_per_case),)))
            remaining = list(range(len(self.dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(self.dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

        return [i.get()[0] for i in r]

    def _get_num_channels(self) -> int:
        """Get the number of channels from dataset JSON."""
        return len(self.dataset_json['channel_names'].keys()
                   if 'channel_names' in self.dataset_json.keys()
                   else self.dataset_json['modality'].keys())

    def run(self, overwrite_existing: bool = False) -> dict:
        """
        Run the fingerprint extraction process.

        Args:
            overwrite_existing: Whether to overwrite existing fingerprint file

        Returns:
            Dictionary containing the dataset fingerprint
        """
        # we do not save the properties file in self.input_folder because that folder might be read-only. We can only
        # reliably write in nnUNet_preprocessed and nnUNet_results, so nnUNet_preprocessed it is
        preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')

        if not isfile(properties_file) or overwrite_existing:
            reader_writer_class = self._get_reader_writer_class()
            num_foreground_samples_per_case = self._calculate_samples_per_case()

            # Process all cases
            results = self._process_cases_parallel(reader_writer_class, num_foreground_samples_per_case)

            # Aggregate results
            shapes_after_crop, spacings, foreground_intensities_per_channel, median_relative_size_after_cropping = \
                aggregate_case_results(results)

            # Calculate global intensity statistics
            num_channels = self._get_num_channels()
            intensity_statistics_per_channel = calculate_global_intensity_statistics(
                foreground_intensities_per_channel, num_channels)

            # Create fingerprint dictionary
            fingerprint = create_fingerprint_dict(
                spacings, shapes_after_crop, intensity_statistics_per_channel, median_relative_size_after_cropping)

            # Save fingerprint
            try:
                save_json(fingerprint, properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)

        return fingerprint


if __name__ == '__main__':
    dfe = DatasetFingerprintExtractor(2, 8)
    dfe.run()