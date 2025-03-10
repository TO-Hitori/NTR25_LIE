from typing import Any

import torch
from torch import nn

import lightning as L
from lightning import Trainer

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme # #39C5BB

from ESDNet_arch import ESDNet
# from ..datasets import NTIRE_Dataset


class LowLightEnhancementModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ESDNet()
        self.lr = None

    def training_step(self, batch, batch_idx):
        a = torch.randn(1, 2)

        pass

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        optim.zero_grad()
        return optim



from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from rich.console import Console
if __name__ == "__main__":
    print("⚡" * 20)
    console = Console()

    trainer = Trainer(reload_dataloaders_every_n_epochs=1)
    trainer.fit()
    console.print("Hello", style="#39C5BB")



