import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

EPS: float = 1e-10

def get_region_proportion(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    if valid_mask is not None:
        x = torch.einsum("bcwhd,bwhd->bcwhd", x, valid_mask)
        cardinality = torch.einsum("bwhd->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3] * x.shape[4]

    region_proportion = (torch.einsum("bcwhd->bc", x) + EPS) / (cardinality + EPS)

    return region_proportion

class RegionLoss(nn.Module):
    def __init__(self, 
                #  mode: str,
                 alpha: float = 1.,
                 factor: float = 1.,
                 step_size: int = 0,
                 max_alpha: float = 100.,
                 temp: float = 1.,
                 ignore_index: int = 255,
                 background_index: int = -1,
                 weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.mode = 'binary'
        self.alpha = alpha
        self.max_alpha = 1
        self.factor = factor
        self.step_size = step_size
        self.temp = 1
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.weight = weight

    def cross_entropy(self, inputs: torch.Tensor, labels: torch.Tensor):
        if len(labels.shape) == len(inputs.shape):
            assert labels.shape[1] == 1
            labels = labels[:, 0]

        if labels.dim() == 3:
            labels = labels.unsqueeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(inputs, labels.type(torch.float32))
        return loss

    def adjust_alpha(self, epoch: int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            curr_alpha = self.alpha
            self.alpha = min(self.alpha * self.factor, self.max_alpha)
            print("CompoundLoss : Adjust the tradeoff param alpha : {:.3g} -> {:.3g}".format(curr_alpha, self.alpha))

    def get_gt_proportion(self, mode: str,
                          labels: torch.Tensor,
                          target_shape,
                          ignore_index: int = 1000):

        valid_mask = (labels >= 0) & (labels != ignore_index)

        if labels.dim() == 3:
            labels = labels.unsqueeze(dim=1)
        bin_labels = labels
        bin_labels = bin_labels.unsqueeze(dim=1)

        gt_proportion = get_region_proportion(bin_labels, valid_mask)

        return gt_proportion, valid_mask

    def get_pred_proportion(self, mode: str,
                            logits: torch.Tensor,
                            temp: float = 1.0,
                            valid_mask=None):

        preds = F.logsigmoid(temp * logits).exp()
        pred_proportion = get_region_proportion(preds, valid_mask)
        return pred_proportion