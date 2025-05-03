import os
import numpy as np
import torch
import pytorch_lightning as pl
from spgrasp import collate, data, utils
from spgrasp import spgnet
from spgrasp import spgnet_main


class FineTuning(pl.callbacks.BaseFinetuning):
    def __init__(self, initial_epochs):
        super().__init__()
        self.initial_epochs = initial_epochs

    def freeze_before_training(self, pl_module):
        pass

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        if current_epoch >= self.initial_epochs:
            for group in pl_module.optimizers().param_groups:
                group["lr"] = pl_module.config["finetune_lr"]


class LightningModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.spgnet = spgnet.SPGNet(config["voxel_size"])
        # self.spgnet = spgnet_main.SPGNetmain(config["voxel_size"])
        self.config = config

    def configure_optimizers(self):
        return torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.config["initial_lr"],
        )

    def on_train_epoch_start(self):
        self.epoch_train_logs = []

    def step(self, batch, batch_idx):

        voxel_coords_16 = []
        for i, x in enumerate(batch["input_voxels_16"]): 
            voxel_coords_16.append(torch.cat([x, torch.ones([x.shape[0], 1], device=x.device) * i], dim=1).int())
        voxel_coords_16 = torch.cat(voxel_coords_16, dim=0) 

        voxel_outputs = self.spgnet(batch, voxel_coords_16)
        voxel_gt = {
            "coarse": batch["voxel_gt_coarse"], 
            "medium": batch["voxel_gt_medium"], 
            "dense": batch["voxel_gt_dense"],
            "dense_label": batch["voxel_gt_label"],
            "dense_rotations": batch["voxel_gt_rotations"],
            "dense_width": batch["voxel_gt_width"],
            "dense_offset": batch["voxel_gt_offset"],
        }

        loss, logs = self.spgnet.losses(voxel_outputs, voxel_gt)
        logs["loss"] = loss.item()
        return loss, logs, voxel_outputs

    def training_step(self, batch, batch_idx):
        n_warmup_steps = 2_000
        if self.global_step < n_warmup_steps:
            target_lr = self.config["initial_lr"]
            lr = 1e-10 + self.global_step / n_warmup_steps * target_lr
            for group in self.optimizers().param_groups:
                group["lr"] = lr

        loss, logs, _ = self.step(batch, batch_idx)
        self.epoch_train_logs.append(logs)
        return loss

    def on_validation_epoch_start(self):
        self.epoch_val_logs = []

    def validation_step(self, batch, batch_idx):
        loss, logs, voxel_outputs = self.step(batch, batch_idx)
        self.epoch_val_logs.append(logs)

    def training_epoch_end(self, outputs): 
        self.epoch_end(self.epoch_train_logs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(self.epoch_val_logs, "val")

    def epoch_end(self, logs, prefix):
        keys = set([key for log in logs for key in log])
        results = {key: [] for key in keys}
        for log in logs:
            for key, value in log.items():
                results[key].append(value)
        logs = {f"{prefix}/{key}": np.nanmean(results[key]) for key in keys}
        self.log_dict(logs, rank_zero_only=True)

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("val")

    def dataloader(self, split):
        if split == "val":
            batch_size = 16
        elif self.current_epoch < self.config["initial_epochs"]:
            batch_size = self.config["initial_batch_size"]
        else:
            batch_size = self.config["finetune_batch_size"]

        scene_names = utils.load_info_files(self.config["dataset_split_dir"], split)
        dset = data.Dataset(
            scene_names,
            self.config["new_dataset_dir"],
        )
        return torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=64,
            collate_fn=collate.sparse_collate_fn,
            drop_last=True,
        )
