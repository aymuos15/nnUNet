import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


def configure_optimizers(trainer_instance):
    # Check if custom optimizer builder is provided in config
    if (hasattr(trainer_instance, 'trainer_config') and
        trainer_instance.trainer_config is not None and
        trainer_instance.trainer_config.optimizer_builder is not None):
        return trainer_instance.trainer_config.optimizer_builder(trainer_instance)

    # Default optimizer configuration
    optimizer = torch.optim.SGD(trainer_instance.network.parameters(), trainer_instance.initial_lr, weight_decay=trainer_instance.weight_decay,
                                momentum=0.99, nesterov=True)
    lr_scheduler = PolyLRScheduler(optimizer, trainer_instance.initial_lr, trainer_instance.num_epochs)
    return optimizer, lr_scheduler