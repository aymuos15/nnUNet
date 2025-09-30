import inspect
from datetime import datetime
import torch
from typing import Union, Optional
from torch import GradScaler, distributed as dist
from torch.cuda import device_count
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

# Import our modular components
from .checkpointing import save_checkpoint, load_checkpoint
from .lifecycle import (setup_output_folders, setup_cascaded_folders, _set_batch_size_and_oversample,
                        run_training)
from nnunetv2.architecture import set_deep_supervision_enabled, _do_i_compile
from .training import configure_optimizers, train_step
from .validation import perform_actual_validation, validation_step
from nnunetv2.training.data.dataset import do_split, get_tr_and_val_datasets
from nnunetv2.training.data.loader import get_dataloaders
from nnunetv2.training.configs import TrainerConfig
from nnunetv2.training.logging.metrics import nnUNetLogger
from nnunetv2.training.logging.console import nnUNetTrainerLogging
from nnunetv2.training.losses.builder import _build_loss, _get_deep_supervision_scales
from nnunetv2.experiment_planning.planning.plans_handler import PlansManager
from nnunetv2.experiment_planning.planning.label_handling import determine_num_input_channels
from nnunetv2.architecture import build_network_architecture
from nnunetv2.training.data.dataset import infer_dataset_class


class nnUNetTrainer(object):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda'),
                 trainer_config: Optional[TrainerConfig] = None):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)
        # apex predator of grug is complexity
        # complexity bad - so we broke this down into modules!

        self.trainer_config = trainer_config
        self._setup_trainer_state(plans, configuration, fold, dataset_json, device)

    def _setup_trainer_state(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device):
        """
        Initialize all trainer state and configuration.

        Args:
            plans: Training plans dictionary
            configuration: Configuration name
            fold: Fold number
            dataset_json: Dataset configuration
            device: Training device
        """
        # DDP setup
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()
        self.device = device

        # Print device information
        if self.is_ddp:
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. "
                  f"The world size is {dist.get_world_size()}. Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")

        # Store init arguments for checkpointing
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__class__.__init__).parameters.keys():
            if k != 'self' and k in locals():
                self.my_init_kwargs[k] = locals()[k]

        # Core configuration
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold

        # Setup folders
        setup_output_folders(self)
        self.dataset_class = None  # -> initialize
        setup_cascaded_folders(self)

        # Training hyperparameters - defaults
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1
        self.current_epoch = 0
        self.enable_deep_supervision = True

        # Apply trainer config overrides if provided
        if hasattr(self, 'trainer_config') and self.trainer_config is not None:
            config = self.trainer_config
            if config.num_epochs is not None:
                self.num_epochs = config.num_epochs
            if config.initial_lr is not None:
                self.initial_lr = config.initial_lr
            if config.weight_decay is not None:
                self.weight_decay = config.weight_decay
            if config.enable_deep_supervision is not None:
                self.enable_deep_supervision = config.enable_deep_supervision
            if config.oversample_foreground_percent is not None:
                self.oversample_foreground_percent = config.oversample_foreground_percent
            if config.batch_size is not None:
                self.batch_size = config.batch_size
            if config.save_every is not None:
                self.save_every = config.save_every

        # Label management
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)

        # Initialize training components (set in initialize())
        self.num_input_channels = None
        self.network = None
        self.optimizer = self.lr_scheduler = None
        self.grad_scaler = GradScaler("cuda") if self.device.type == 'cuda' else None
        self.loss = None

        # Logging setup
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder,
                                      "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                      (timestamp.year, timestamp.month, timestamp.day,
                                       timestamp.hour, timestamp.minute, timestamp.second))
        self.logger = nnUNetLogger()

        # Console/file logger (create once, reuse forever)
        self._console_logger = nnUNetTrainerLogging(
            log_file=self.log_file,
            output_folder=self.output_folder,
            local_rank=self.local_rank,
            plans_manager=self.plans_manager,
            configuration_manager=self.configuration_manager,
            configuration_name=self.configuration_name,
            device=self.device
        )

        # Training state
        self.dataloader_train = self.dataloader_val = None
        self._best_ema = None
        self.inference_allowed_mirroring_axes = None

        # Checkpointing
        self.save_every = 50
        self.disable_checkpointing = False
        self.was_initialized = False

        # Print citation
        self.print_to_log_file(
            "\n#######################################################################\n"
            "Please cite the following paper when using nnU-Net:\n"
            "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
            "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
            "Nature methods, 18(2), 203-211.\n"
            "#######################################################################\n",
            also_print_to_console=True, add_timestamp=False)

    def initialize(self):
        """
        Initialize trainer components (network, optimizer, loss, etc.).
        """
        if not self.was_initialized:
            # DDP batch size and oversampling can differ between workers and needs adaptation
            _set_batch_size_and_oversample(self)

            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json)

            self.network = build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision
            ).to(self.device)

            # Compile network for free speedup
            if _do_i_compile(self):
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = torch.nn.parallel.DistributedDataParallel(
                    self.network, device_ids=[self.local_rank])

            self.loss = _build_loss(self)
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            self.was_initialized = True
        else:
            raise RuntimeError("You have called initialize even though the trainer was already initialized. "
                              "That should not happen.")

    def _save_debug_information(self):
        """Save debug information to JSON file."""
        return self._console_logger._save_debug_information(self)

    def _get_deep_supervision_scales(self):
        return _get_deep_supervision_scales(self)

    def _build_loss(self):
        return _build_loss(self)

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        """Print message to log file and optionally to console."""
        return self._console_logger.print_to_log_file(
            *args, 
            also_print_to_console=also_print_to_console, 
            add_timestamp=add_timestamp
        )

    def print_plans(self):
        """Print training plans and configuration to log file."""
        return self._console_logger.print_plans()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        return configure_optimizers(self)

    def do_split(self):
        """Perform dataset split - delegated to data.datasets module."""
        return do_split(self)

    def get_tr_and_val_datasets(self):
        """Get training and validation datasets - delegated to data.datasets module."""
        return get_tr_and_val_datasets(self)

    def get_dataloaders(self):
        """Get training and validation data loaders - delegated to data.loaders module."""
        return get_dataloaders(self)

    def set_deep_supervision_enabled(self, enabled: bool):
        """Enable/disable deep supervision - delegated to initialization.network module."""
        set_deep_supervision_enabled(self, enabled)

    def save_checkpoint(self, filename: str) -> None:
        save_checkpoint(self, filename)

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        load_checkpoint(self, filename_or_checkpoint)

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Perform full validation - delegated to validation.runner module."""
        return perform_actual_validation(self, save_probabilities)

    def run_training(self):
        """Main training loop - delegated to lifecycle.training module."""
        return run_training(self)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        """Build network architecture - delegated to initialization.network module."""
        return build_network_architecture(architecture_class_name,
                                           arch_init_kwargs,
                                           arch_init_kwargs_req_import,
                                           num_input_channels,
                                           num_output_channels,
                                           enable_deep_supervision)