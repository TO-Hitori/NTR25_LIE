import torch

import lightning as L

from torchmetrics.image import PeakSignalNoiseRatio as PSNR

from NTR25_LIE.src.utils.basic_utils import instantiate_from_config
from NTR25_LIE.src.models.loss_func import CombinedLoss



class LowLightEnhancementModel(L.LightningModule):
    def __init__(
            self,
            # 网络
            esdnet_config,
            # 优化器
            learning_rate,
            T_0,
            T_mult,
            # 损失函数
            charbonnier_weight=1.0,
            lpips_weight=0.04,
    ):
        super().__init__()
        self.save_hyperparameters()  # 保存超参数
        self.example_input_array = torch.Tensor(1, 3, 512, 512)  # debug

        # model
        self.model = instantiate_from_config(esdnet_config)
        self.psnr = PSNR(data_range=(-1.0, 1.0))


        # loss
        self.loss_function = CombinedLoss(charbonnier_weight=charbonnier_weight, lpips_weight=lpips_weight)

        # optim
        self.lr = learning_rate
        self.T_0 = T_0
        self.T_mult = T_mult

    def forward(self, x):
        image_1, image_2, image_4 = self.model(x)
        return image_1, image_2, image_4

    def training_step(self, batch, batch_idx):
        image_in = batch['image_in']
        image_gt = batch['image_gt']
        image_gt_2 = batch['image_gt_2']
        image_gt_4 = batch['image_gt_4']

        image_pre, image_pre_2, image_pre_4 = self.model(image_in)

        loss, loss_dict = self.loss_function(
            predictions_1=image_pre,
            predictions_2=image_pre_2,
            predictions_4=image_pre_4,
            targets_1=image_gt,
            targets_2=image_gt_2,
            targets_4=image_gt_4
        )
        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=True)
        self.log("learning_rate", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=True)

        return loss

    @torch.no_grad()

    from torch.utils.tensorboard import SummaryWriter
    sw = SummaryWriter("./logs")
    sw.add_images()

    def validation_step(self, batch, batch_idx):
        image_in = batch['image_in']
        image_gt = batch['image_gt']

        image_val, image_val_2, image_val_4 = self.model(image_in)
        psnr_val = self.psnr(image_val, image_gt)
        self.log("psnr_val", psnr_val, on_step=True, on_epoch=True, prog_bar=True)

        tensorboard = self.logger.experiment
        image_val = (image_val + 1) / 2
        tensorboard.add_images(f"val_image_{batch_idx}", image_val, global_step=self.global_step)
        


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.90, 0.95), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult,
                                                                         eta_min=0)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    print("⚡" * 20)
