import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


def configure_optimizers(trainer_instance):
    optimizer = torch.optim.SGD(trainer_instance.network.parameters(), trainer_instance.initial_lr, weight_decay=trainer_instance.weight_decay,
                                momentum=0.99, nesterov=True)
    lr_scheduler = PolyLRScheduler(optimizer, trainer_instance.initial_lr, trainer_instance.num_epochs)
    return optimizer, lr_scheduler