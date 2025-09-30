import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


def configure_optimizers(trainer_instance):
    """Wrapper function for optimizer configuration."""
    return OptimizerConfig.configure_optimizers(trainer_instance)


class OptimizerConfig:
    """Optimizer configuration component extracted from nnUNetTrainer"""

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler