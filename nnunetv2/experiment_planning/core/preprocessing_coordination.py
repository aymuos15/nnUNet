from typing import Union, Tuple, List
from distutils.file_util import copy_file

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.data.dataset_io.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.experiment_planning.plans.plans_manager import PlansManager
from nnunetv2.data.dataset_io.utils import get_filenames_of_train_images_and_targets


def preprocess_dataset(dataset_id: int,
                       plans_identifier: str = 'nnUNetPlans',
                       configurations: Union[Tuple[str], List[str]] = ('2d', '3d_fullres', '3d_lowres'),
                       num_processes: Union[int, Tuple[int, ...], List[int]] = (8, 4, 8),
                       verbose: bool = False) -> None:
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f'The list provided with num_processes must either have len 1 or as many elements as there are '
            f'configurations (see --help). Number of configurations: {len(configurations)}, length '
            f'of num_processes: '
            f'{len(num_processes)}')

    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f'Preprocessing dataset {dataset_name}')
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations):
        print(f'Configuration: {c}...')
        if c not in plans_manager.available_configurations:
            print(
                f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of "
                f"dataset {dataset_name}. Skipping.")
            continue
        configuration_manager = plans_manager.get_configuration(c)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        preprocessor.run(dataset_id, c, plans_identifier, num_processes=n)

    # copy the gt to a folder in the nnUNet_preprocessed so that we can do validation even if the raw data is no
    # longer there (useful for compute cluster where only the preprocessed data is available)
    maybe_mkdir_p(join(nnUNet_preprocessed, dataset_name, 'gt_segmentations'))
    dataset_json = load_json(join(nnUNet_raw, dataset_name, 'dataset.json'))
    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)
    # only copy files that are newer than the ones already present
    for k in dataset:
        copy_file(dataset[k]['label'],
                  join(nnUNet_preprocessed, dataset_name, 'gt_segmentations', k + dataset_json['file_ending']),
                  update=True)


def preprocess(dataset_ids: List[int],
               plans_identifier: str = 'nnUNetPlans',
               configurations: Union[Tuple[str], List[str]] = ('2d', '3d_fullres', '3d_lowres'),
               num_processes: Union[int, Tuple[int, ...], List[int]] = (8, 4, 8),
               verbose: bool = False):
    for d in dataset_ids:
        preprocess_dataset(d, plans_identifier, configurations, num_processes, verbose)