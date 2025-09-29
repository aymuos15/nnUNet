import inspect
from datetime import datetime

import torch
from torch import GradScaler, distributed as dist
from torch.cuda import device_count
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

from ..initialization.config import setup_output_folders, setup_cascaded_folders, _set_batch_size_and_oversample
from ..initialization.network import build_network_architecture, _do_i_compile
from ..loss.builder import _build_loss


def setup_trainer_state(trainer_instance, plans: dict, configuration: str, fold: int, dataset_json: dict, device):
    """
    Initialize all trainer state and configuration.

    Args:
        trainer_instance: The nnUNetTrainer instance to initialize
        plans: Training plans dictionary
        configuration: Configuration name
        fold: Fold number
        dataset_json: Dataset configuration
        device: Training device
    """
    # DDP setup
    trainer_instance.is_ddp = dist.is_available() and dist.is_initialized()
    trainer_instance.local_rank = 0 if not trainer_instance.is_ddp else dist.get_rank()
    trainer_instance.device = device

    # Print device information
    if trainer_instance.is_ddp:
        print(f"I am local rank {trainer_instance.local_rank}. {device_count()} GPUs are available. "
              f"The world size is {dist.get_world_size()}. Setting device to {trainer_instance.device}")
        trainer_instance.device = torch.device(type='cuda', index=trainer_instance.local_rank)
    else:
        if trainer_instance.device.type == 'cuda':
            trainer_instance.device = torch.device(type='cuda', index=0)
        print(f"Using device: {trainer_instance.device}")

    # Store init arguments for checkpointing
    trainer_instance.my_init_kwargs = {}
    for k in inspect.signature(trainer_instance.__class__.__init__).parameters.keys():
        if k in locals():
            trainer_instance.my_init_kwargs[k] = locals()[k]

    # Core configuration
    trainer_instance.plans_manager = PlansManager(plans)
    trainer_instance.configuration_manager = trainer_instance.plans_manager.get_configuration(configuration)
    trainer_instance.configuration_name = configuration
    trainer_instance.dataset_json = dataset_json
    trainer_instance.fold = fold

    # Setup folders
    setup_output_folders(trainer_instance)
    trainer_instance.dataset_class = None  # -> initialize
    setup_cascaded_folders(trainer_instance)

    # Training hyperparameters
    trainer_instance.initial_lr = 1e-2
    trainer_instance.weight_decay = 3e-5
    trainer_instance.oversample_foreground_percent = 0.33
    trainer_instance.probabilistic_oversampling = False
    trainer_instance.num_iterations_per_epoch = 250
    trainer_instance.num_val_iterations_per_epoch = 50
    trainer_instance.num_epochs = 1
    trainer_instance.current_epoch = 0
    trainer_instance.enable_deep_supervision = True

    # Label management
    trainer_instance.label_manager = trainer_instance.plans_manager.get_label_manager(dataset_json)

    # Initialize training components (set in initialize())
    trainer_instance.num_input_channels = None
    trainer_instance.network = None
    trainer_instance.optimizer = trainer_instance.lr_scheduler = None
    trainer_instance.grad_scaler = GradScaler("cuda") if trainer_instance.device.type == 'cuda' else None
    trainer_instance.loss = None

    # Logging setup
    timestamp = datetime.now()
    maybe_mkdir_p(trainer_instance.output_folder)
    trainer_instance.log_file = join(trainer_instance.output_folder,
                                   "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                   (timestamp.year, timestamp.month, timestamp.day,
                                    timestamp.hour, timestamp.minute, timestamp.second))
    trainer_instance.logger = nnUNetLogger()

    # Training state
    trainer_instance.dataloader_train = trainer_instance.dataloader_val = None
    trainer_instance._best_ema = None
    trainer_instance.inference_allowed_mirroring_axes = None

    # Checkpointing
    trainer_instance.save_every = 50
    trainer_instance.disable_checkpointing = False
    trainer_instance.was_initialized = False

    # Print citation
    trainer_instance.print_to_log_file(
        "\n#######################################################################\n"
        "Please cite the following paper when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n"
        "#######################################################################\n",
        also_print_to_console=True, add_timestamp=False)


def initialize_trainer(trainer_instance):
    """
    Initialize trainer components (network, optimizer, loss, etc.).

    Args:
        trainer_instance: The nnUNetTrainer instance to initialize
    """
    if not trainer_instance.was_initialized:
        # DDP batch size and oversampling can differ between workers and needs adaptation
        _set_batch_size_and_oversample(trainer_instance)

        trainer_instance.num_input_channels = determine_num_input_channels(
            trainer_instance.plans_manager, trainer_instance.configuration_manager, trainer_instance.dataset_json)

        trainer_instance.network = build_network_architecture(
            trainer_instance.configuration_manager.network_arch_class_name,
            trainer_instance.configuration_manager.network_arch_init_kwargs,
            trainer_instance.configuration_manager.network_arch_init_kwargs_req_import,
            trainer_instance.num_input_channels,
            trainer_instance.label_manager.num_segmentation_heads,
            trainer_instance.enable_deep_supervision
        ).to(trainer_instance.device)

        # Compile network for free speedup
        if _do_i_compile(trainer_instance):
            trainer_instance.print_to_log_file('Using torch.compile...')
            trainer_instance.network = torch.compile(trainer_instance.network)

        trainer_instance.optimizer, trainer_instance.lr_scheduler = trainer_instance.configure_optimizers()

        # DDP wrapper
        if trainer_instance.is_ddp:
            trainer_instance.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer_instance.network)
            trainer_instance.network = torch.nn.parallel.DistributedDataParallel(
                trainer_instance.network, device_ids=[trainer_instance.local_rank])

        trainer_instance.loss = _build_loss(trainer_instance)
        trainer_instance.dataset_class = infer_dataset_class(trainer_instance.preprocessed_dataset_folder)

        trainer_instance.was_initialized = True
    else:
        raise RuntimeError("You have called initialize even though the trainer was already initialized. "
                          "That should not happen.")