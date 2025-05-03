import argparse
import os
import yaml
from pathlib import Path
import torch
import pytorch_lightning as pl
# import wandb
from spgrasp import lightningmodel
from vgn.io import *


class CudaClearCacheCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
    def on_validation_end(self, trainer, pl_module):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--num", type=int, default=16)
    args = parser.parse_args()
    # wandb.login(key="60bbe63029dced07ed9626b5280ccc82890f646b")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_dataset_dir = os.path.join(config["new_dataset_dir"], f"data_{args.scene}_train_random_new_{args.num}M")
    raw_dataset_dir = os.path.join(config["raw_dataset_dir"], f"data_{args.scene}_train_random_raw_{args.num}M")
    dataset_split_dir = os.path.join(config["dataset_split_dir"], f"{args.scene}")
    df = read_df(Path(new_dataset_dir))
    df = df["scene_id"].to_numpy()
    df = np.unique(df)
    np.random.seed(config["seed"])
    val_set = np.random.choice(df, size=int(len(df) * 0.2), replace=False)  # 验证集比例
    train_set = np.setdiff1d(df, val_set, assume_unique=True)
    with open(Path(dataset_split_dir) / "train.txt", "w") as ft:
        for element in train_set:
            ft.write(f"{element}\n")
    with open(Path(dataset_split_dir) / "val.txt", "w") as fv:
        for element in val_set:
            fv.write(f"{element}\n")
    config["new_dataset_dir"] = new_dataset_dir
    config["raw_dataset_dir"] = raw_dataset_dir
    config["dataset_split_dir"] = dataset_split_dir
    # if training needs to be resumed from checkpoint,
    # it is helpful to change the seed so that
    # the same data augmentations are not re-used
    pl.seed_everything(config["seed"])
    save_dir = os.path.join("results_spgrasp", config["wandb_project_name"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = pl.loggers.WandbLogger(project=config["wandb_project_name"], save_dir=save_dir, config=config, offline=False) # Note offline
    
    if os.getenv("LOCAL_RANK", '0') == '0':
        ckpt_dir = os.path.join(logger.experiment.dir, "ckpts")
    else:
        ckpt_dir = os.path.join(config["wandb_project_name"], "ckpts")
    checkpointer = pl.callbacks.ModelCheckpoint(
        save_last=True,
        dirpath=ckpt_dir,
        verbose=True,
        filename='{epoch:03d}-{val/voxel_loss_dense:.3f}',
        auto_insert_metric_name=False,
        # save_top_k=-1,  # 默认是1
        monitor="val/loss"
    )
    callbacks = [checkpointer, lightningmodel.FineTuning(config["initial_epochs"]), CudaClearCacheCallback()]
    
    if config["use_amp"]:
        amp_kwargs = {"precision": 16}
    else:
        amp_kwargs = {}
    
    model = lightningmodel.LightningModel(config)
    trainer = pl.Trainer(
        gpus=args.gpus,
        strategy='ddp',
        # strategy="ddp_find_unused_parameters_false",
        # sync_batchnorm=True,
        num_sanity_val_steps=0,
        logger=logger,
        benchmark=False,
        max_epochs=config["initial_epochs"] + config["finetune_epochs"],
        # check_val_every_n_epoch=10,  # 实际存储检查点的轮数是check_val_every_n_epoch和ModelCheckpoint中every_n_epochs参数的最小公倍数
        check_val_every_n_epoch=1,
        detect_anomaly=False,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=10,
        **amp_kwargs,
        # log_every_n_steps=5
    )
    trainer.fit(model, ckpt_path=config["ckpt"])
