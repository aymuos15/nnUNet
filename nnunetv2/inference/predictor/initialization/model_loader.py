"""Model loading and checkpoint management for nnUNet predictor."""

import os
from typing import Union, Tuple, List, Optional
import torch
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, subdirs

import nnunetv2
from nnunetv2.utilities.core.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.planning.label_handling import determine_num_input_channels
from nnunetv2.utilities.planning.plans_handler import PlansManager


def initialize_from_trained_model_folder(predictor,
                                        model_training_output_dir: str,
                                        use_folds: Union[Tuple[Union[int, str]], None],
                                        checkpoint_name: str = 'checkpoint_final.pth'):
    """
    Initialize predictor from a trained model folder.

    Args:
        predictor: The nnUNetPredictor instance
        model_training_output_dir: Path to the trained model folder
        use_folds: Which folds to use for prediction
        checkpoint_name: Name of the checkpoint file to load
    """
    if use_folds is None:
        use_folds = auto_detect_available_folds(model_training_output_dir, checkpoint_name)

    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, 'plans.json'))
    plans_manager = PlansManager(plans)

    if isinstance(use_folds, str):
        use_folds = [use_folds]

    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                map_location=torch.device('cpu'), weights_only=False)
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

    configuration_manager = plans_manager.get_configuration(configuration_name)

    # Restore network
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if trainer_class is None:
        raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                          f'Please place it there (or in a submodule) and make sure it can be imported with '
                          f'"from nnunetv2.training.nnUNetTrainer import {trainer_name}"')

    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False
    )

    # Manual initialization
    predictor.plans_manager = plans_manager
    predictor.configuration_manager = configuration_manager
    predictor.list_of_parameters = parameters
    predictor.network = network
    predictor.dataset_json = dataset_json
    predictor.trainer_name = trainer_name
    predictor.allowed_mirroring_axes = inference_allowed_mirroring_axes
    predictor.label_manager = plans_manager.get_label_manager(dataset_json)

    # Move network to device
    predictor.network = predictor.network.to(predictor.device)

    # Apply torch.compile if enabled
    compile_network(predictor)


def manual_initialization(predictor,
                         network: nn.Module,
                         plans_manager: PlansManager,
                         configuration_manager,
                         parameters: Optional[List[dict]],
                         dataset_json: dict,
                         trainer_name: str,
                         inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
    """
    Manual initialization used by nnUNetTrainer for final validation.

    Args:
        predictor: The nnUNetPredictor instance
        network: Pre-built network
        plans_manager: PlansManager instance
        configuration_manager: ConfigurationManager instance
        parameters: Network parameters
        dataset_json: Dataset configuration
        trainer_name: Name of the trainer class
        inference_allowed_mirroring_axes: Allowed mirroring axes for inference
    """
    predictor.plans_manager = plans_manager
    predictor.configuration_manager = configuration_manager
    predictor.list_of_parameters = parameters
    predictor.network = network
    predictor.dataset_json = dataset_json
    predictor.trainer_name = trainer_name
    predictor.allowed_mirroring_axes = inference_allowed_mirroring_axes
    predictor.label_manager = plans_manager.get_label_manager(dataset_json)

    # Move network to device
    predictor.network = predictor.network.to(predictor.device)

    compile_network(predictor)


def compile_network(predictor):
    """
    Apply torch.compile to the network if enabled.

    Args:
        predictor: The nnUNetPredictor instance
    """
    allow_compile = True
    allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
                os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
    allow_compile = allow_compile and not isinstance(predictor.network, OptimizedModule)
    if isinstance(predictor.network, DistributedDataParallel):
        allow_compile = allow_compile and isinstance(predictor.network.module, OptimizedModule)
    if allow_compile:
        print('Using torch.compile')
        predictor.network = torch.compile(predictor.network)


def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
    """
    Auto-detect available folds in the model training output directory.

    Args:
        model_training_output_dir: Path to the model training output directory
        checkpoint_name: Name of the checkpoint file

    Returns:
        List of available fold numbers
    """
    print('use_folds is None, attempting to auto detect available folds')
    fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
    fold_folders = [i for i in fold_folders if i != 'fold_all']
    fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
    use_folds = [int(i.split('_')[-1]) for i in fold_folders]
    print(f'found the following folds: {use_folds}')
    return use_folds