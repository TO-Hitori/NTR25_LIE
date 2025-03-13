from src.utils.basic_utils import instantiate_from_config
from omegaconf import OmegaConf
from argparse import ArgumentParser
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar, ModelSummary

import megfile as mf

from src.datasets.NTIRE_Dataset import NTIRE_LIE_Dataset
from torch.utils.data import DataLoader


def main(config):
    L.seed_everything(config.lightning.seed)
    # 回调函数
    early_stop_callback = EarlyStopping(monitor="psnr_val", min_delta=0.00, patience=10, verbose=False, mode="max")
    ckpt_callback = ModelCheckpoint(
        dirpath=mf.smart_path_join("Experiment", config.lightning.exp_name, "checkpoints"),
        filename="{epoch}_{psnr_val: .4f}",
        monitor="psnr_val",
        mode="max",
        save_top_k=10,
        save_last=True
    )
    tqdm_callback = TQDMProgressBar(refresh_rate=2, process_position=0)
    ms_callback = ModelSummary(max_depth=3)
    callback_list = [early_stop_callback, ckpt_callback, tqdm_callback, ms_callback]

    # 模型
    model = instantiate_from_config(config.model)

    ds_train = NTIRE_LIE_Dataset(
        data_root=config.data.data_path,
        subfolder="Train",
        patch_size=config.data.patch_size_train,
    )
    ds_val = NTIRE_LIE_Dataset(
        data_root=config.data.data_path,
        subfolder="Val",
        patch_size=config.data.patch_size_val,
    )
    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        dataset=ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True
    )

    trainer = L.Trainer(
        max_epochs=config.lightning.max_epochs,
        min_epochs=config.lightning.min_epochs,
        log_every_n_steps=1,
        callbacks=callback_list,
        check_val_every_n_epoch=config.lightning.check_val_every_n_epoch,
        default_root_dir=mf.smart_path_join("Experiment", config.lightning.exp_name),
        num_sanity_val_steps=2,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/basic_config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(config))

    main(config)
