from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.losses import Constrained__DC_and_CE_loss
from nnunetv2.training.loss.dice_loss import MemoryEfficientSoftDiceLoss
import numpy as np

class nnUNetTrainerConstrained__DC_and_CE_Loss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        loss = Constrained__DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                        'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                        ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
