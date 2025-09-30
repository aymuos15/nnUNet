from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Any, Dict


@dataclass
class TrainerConfig:
    """
    Configuration for nnUNet trainer variants.

    This replaces the inheritance-based trainer variants with a config-based approach.
    All parameters are optional - None means "use the default from the base trainer".
    """

    # Metadata
    name: str = "base"
    description: str = ""

    # Training parameters
    num_epochs: Optional[int] = None
    initial_lr: Optional[float] = None
    weight_decay: Optional[float] = None

    # Data augmentation parameters
    mirror_axes: Optional[Tuple[int, ...]] = None
    inference_allowed_mirroring_axes: Optional[Tuple[int, ...]] = None
    do_dummy_2d_data_aug: Optional[bool] = None
    disable_data_augmentation: bool = False
    rotation_for_DA: Optional[Any] = None

    # Architecture parameters
    enable_deep_supervision: Optional[bool] = None

    # Strategy overrides (callables that replace methods)
    # These should be functions with the same signature as the methods they replace
    optimizer_builder: Optional[Callable] = None  # Replaces configure_optimizers
    loss_builder: Optional[Callable] = None  # Replaces _build_loss
    training_transforms_builder: Optional[Callable] = None  # Replaces get_training_transforms
    network_builder: Optional[Callable] = None  # Replaces build_network_architecture
    lr_scheduler_builder: Optional[Callable] = None  # For custom LR schedulers

    # Batch and sampling configuration
    batch_size: Optional[int] = None
    oversample_foreground_percent: Optional[float] = None

    # Checkpointing
    save_every: Optional[int] = None

    # Additional custom parameters (for advanced users)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_epochs is not None and self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.initial_lr is not None and self.initial_lr <= 0:
            raise ValueError(f"initial_lr must be positive, got {self.initial_lr}")

    def copy(self, **overrides):
        """Create a copy of this config with specified overrides."""
        from dataclasses import replace
        return replace(self, **overrides)


# Global registry for trainer configs
_TRAINER_CONFIG_REGISTRY: Dict[str, TrainerConfig] = {}


def register_config(config: TrainerConfig):
    """Register a trainer config so it can be loaded by name."""
    if config.name in _TRAINER_CONFIG_REGISTRY:
        raise ValueError(f"Config '{config.name}' is already registered")
    _TRAINER_CONFIG_REGISTRY[config.name] = config
    return config


def get_config(name: str) -> TrainerConfig:
    """Get a registered config by name."""
    if name not in _TRAINER_CONFIG_REGISTRY:
        raise ValueError(
            f"Config '{name}' not found. Available configs: {list(_TRAINER_CONFIG_REGISTRY.keys())}"
        )
    return _TRAINER_CONFIG_REGISTRY[name]


def list_configs() -> list:
    """List all registered config names."""
    return list(_TRAINER_CONFIG_REGISTRY.keys())


# Register the base config
BASE_CONFIG = TrainerConfig(
    name="base",
    description="Default nnUNet trainer configuration"
)
register_config(BASE_CONFIG)