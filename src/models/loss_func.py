import torch
import torch.nn as nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class CombinedLoss(nn.Module):
    """
    结合 Charbonnier 损失和 LPIPS 损失的类。
    初始化时可以设置两种损失的权重系数。
    """
    def __init__(self, charbonnier_weight=1.0, lpips_weight=1.0, epsilon=1e-6, *args, **kwargs):
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.lpips_weight = lpips_weight

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", reduction="mean")
        self.charbonnier = CharbonnierLoss(epsilon=epsilon)

    def forward(
            self,
            predictions_1, predictions_2, predictions_4,
            targets_1, targets_2, targets_4
        ):
        loss_charbonnier_1 = self.charbonnier(predictions_1, targets_1)
        loss_lpips_1 = self.lpips(predictions_1, targets_1)
        loss_1 = self.charbonnier_weight * loss_charbonnier_1 + self.lpips_weight * loss_lpips_1

        loss_charbonnier_2 = self.charbonnier(predictions_2, targets_2)
        loss_lpips_2 = self.lpips(predictions_2, targets_2)
        loss_2 = self.charbonnier_weight * loss_charbonnier_2 + self.lpips_weight * loss_lpips_2

        loss_charbonnier_4 = self.charbonnier(predictions_4, targets_4)
        loss_lpips_4 = self.lpips(predictions_4, targets_4)
        loss_4 = self.charbonnier_weight * loss_charbonnier_4 + self.lpips_weight * loss_lpips_4

        loss_total = loss_1 + loss_2 + loss_4

        loss_dict = {
            "loss_charbonnier_1": loss_charbonnier_1,
            "loss_lpips_1": loss_lpips_1,
            "loss_1": loss_1,
            "loss_charbonnier_2": loss_charbonnier_2,
            "loss_lpips_2": loss_lpips_2,
            "loss_2": loss_2,
            "loss_charbonnier_4": loss_charbonnier_4,
            "loss_lpips_4": loss_lpips_4,
            "loss_4": loss_4,
            "loss_total": loss_total
        }

        return loss_total, loss_dict



class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        diff = predictions - targets
        loss = torch.mean(torch.sqrt(diff * diff + self.epsilon))
        return loss
