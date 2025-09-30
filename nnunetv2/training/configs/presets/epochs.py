"""Training length (epoch) configuration presets."""

from nnunetv2.training.configs import TrainerConfig, register_config


# Various epoch configurations
for num_epochs in [1, 2, 3, 4, 5, 10, 20, 50, 100, 250, 500, 750, 1000, 1500, 2000, 4000, 8000]:
    config = TrainerConfig(
        name=f"{num_epochs}epochs",
        description=f"Train for {num_epochs} epochs",
        num_epochs=num_epochs
    )
    register_config(config)


# Combined configs: epochs + no mirroring
for num_epochs in [1, 2, 3, 4, 5, 10, 20, 50, 100, 250, 500, 750, 1000, 1500, 2000, 4000, 8000]:
    config = TrainerConfig(
        name=f"{num_epochs}epochs_no_mirroring",
        description=f"Train for {num_epochs} epochs without mirroring",
        num_epochs=num_epochs,
        mirror_axes=None,
        inference_allowed_mirroring_axes=None
    )
    register_config(config)
