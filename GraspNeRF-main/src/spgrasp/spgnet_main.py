import collections
import numpy_indexed as npi

import torch
import spconv.pytorch as spconv
from spconv.pytorch.functional import _indice_to_scalar
import torch.nn.functional as F
from spgrasp import cnn3d, utils
from spgrasp.net3d.sparsecnn import SCN3d as backbone3d
from spgrasp.net3d.sparse3d import combineSparseConvTensor, xyzb2bxyz, bxyz2xyzb


class SPGNetmain(torch.nn.Module):
    def __init__(self, voxel_size=0.0075):
        super().__init__()
        self.resolutions = collections.OrderedDict(
            [
                ["dense", voxel_size],
            ]
        )

        net_grid_output_depths = [64, 32, 16]
        net3d_hidden_depths = [128, 64, 64]
        net3d_output_depths = [48, 24, 16]  # 降低输出通道数量，为了和更高分辨率的特征拼接55开
        net3d_channels = [
            [128, 256, 512],
            [64, 128, 256],
            [64, 64, 128, 256, 512]
        ]

        self.net_grid = cnn3d.C3Dmain(net_grid_output_depths)
        self.upsampler = Upsampler()

        self.output_layers = torch.nn.ModuleDict()
        self.net3d = torch.nn.ModuleDict()
        self.layer_norms = torch.nn.ModuleDict()

        self.task_layers_offset = spconv.SubMConv3d(net3d_output_depths[-1], 3, 1, 1, padding=1, bias=True, indice_key="dense")
        self.task_layers_rotations_score = spconv.SubMConv3d(net3d_output_depths[-1], 1, 1, 1, padding=1, bias=True, indice_key="dense")
        self.task_layers_rotations = spconv.SubMConv3d(net3d_output_depths[-1], 4, 1, 1, padding=1, bias=True, indice_key="dense")
        self.task_layers_width = spconv.SubMConv3d(net3d_output_depths[-1], 1, 1, 1, padding=1, bias=True, indice_key="dense")

        self.layer_norms = torch.nn.LayerNorm(net_grid_output_depths[2])  # 层正则化
        input_depth = net_grid_output_depths[2]

        net = backbone3d(input_depth=input_depth, channels=net3d_channels[2], post_deform=True,
                         hidden_depth=net3d_hidden_depths[2], output_depth=net3d_output_depths[2])
        output_depth = net.output_depth

        self.net3d = net
        self.output_layers = spconv.SubMConv3d(output_depth, 1, 1, 1, padding=1, bias=True, indice_key="dense")

    def forward(self, batch, voxel_inds_04):

        feats_grid = self.net_grid(batch["grid_input"])  # (16,64,10,10,10),(16,32,20,20,20),(16,16,40,40,40)

        batch_size = batch["grid_input"].shape[0]  # 16

        device = voxel_inds_04.device
        voxel_outputs = {}

        voxel_inds = voxel_inds_04  # 形状（10*10*10*16,4）
        voxel_dim_16 = voxel_inds_04[-1][:3] + 1  # 位置索引（10,10,10），形状（3,）
        voxel_features = torch.empty(  # 形状（10*10*10*16,0）
            (len(voxel_inds), 0), dtype=feats_grid.dtype, device=device
        )
        voxel_logits = torch.empty(  # 形状（10*10*10*16,0）
            (len(voxel_inds), 0), dtype=feats_grid.dtype, device=device
        )

        feats_3d = torch.permute(feats_grid, (2, 3, 4, 0, 1))
        feats_3d = feats_3d[voxel_inds[:, 0].long(), voxel_inds[:, 1].long(), voxel_inds[:, 2].long(), voxel_inds[:, 3].long()]

        feats_3d = self.layer_norms(feats_3d)

        voxel_features = torch.cat((voxel_features, feats_3d, voxel_logits), dim=-1)  # 先拼接后稀疏卷积  # 形状 （10*10*10*16,64）

        voxel_dim = voxel_dim_16.int().tolist()  # 实际值（10,10,10）
        voxel_features = spconv.SparseConvTensor(voxel_features, xyzb2bxyz(voxel_inds), voxel_dim, batch_size)  # 形状（10*10*10*16,64）,形状（10*10*10*16,4）,（10,10,10）,16
        voxel_features = self.net3d(voxel_features, voxel_dim)

        voxel_logits = self.output_layers(voxel_features)  # 形状（10*10*10*16,1）,形状（10*10*10*16,4）,（10,10,10）,16
        voxel_outputs["dense"] = voxel_logits

        voxel_outputs["offset"] = self.task_layers_offset(voxel_features)
        voxel_outputs["rotations_score"] = self.task_layers_rotations_score(voxel_features)
        voxel_outputs["rotations"] = self.task_layers_rotations(voxel_features)
        voxel_outputs["width"] = self.task_layers_width(voxel_features)

        return voxel_outputs

    def losses(self, voxel_logits, voxel_gt):
        voxel_losses = {}
        grasp_config_losses = {}
        logits = voxel_logits["dense"]
        # logits = voxel_logits["rotations_score"]
        gt = voxel_gt["dense"]
        gt = combineSparseConvTensor(gt, device=logits.features.device)  # 形状（i16*2，1），（i16*2，4）,（3,）,(1,)
        cur_loss = torch.zeros(1, device=logits.features.device, dtype=torch.float32)
        if len(logits.indices) > 0:  # 形状（10*10*10*16,4）
            pred_scalar = _indice_to_scalar(logits.indices, [logits.batch_size] + logits.spatial_shape)
            gt_scalar = _indice_to_scalar(gt.indices, [logits.batch_size] + logits.spatial_shape)
            idx_query = npi.indices(gt_scalar.cpu().numpy(), pred_scalar.cpu().numpy(), missing=-1)
            good_query = idx_query != -1

            gt = gt.features.squeeze(1)[idx_query[good_query]]  # 重合体素的标签（标签和预测重合的体素）
            logits = logits.features.squeeze(1)[good_query]  # 重合体素的预测
            if len(logits) > 0:

                cur_loss = F.l1_loss(
                    utils.log_transform(1.05 * torch.tanh(logits)),
                    utils.log_transform(gt),
                )

                offset = voxel_logits["offset"]
                rotations_score = voxel_logits["rotations_score"]
                rotations = voxel_logits["rotations"]
                width = voxel_logits["width"]

                dense_label = voxel_gt["dense_label"]
                dense_rotations = voxel_gt["dense_rotations"]
                dense_width = voxel_gt["dense_width"]
                dense_offset = voxel_gt["dense_offset"]

                dense_label = combineSparseConvTensor(dense_label, device=rotations_score.features.device)
                dense_rotations = combineSparseConvTensor(dense_rotations, device=rotations.features.device)
                dense_width = combineSparseConvTensor(dense_width, device=width.features.device)
                dense_offset = combineSparseConvTensor(dense_offset, device=offset.features.device)

                offset_loss = torch.zeros(1, device=offset.features.device, dtype=torch.float32)
                rotations_score_loss = torch.zeros(1, device=rotations_score.features.device, dtype=torch.float32)
                rotations_loss = torch.zeros(1, device=rotations.features.device, dtype=torch.float32)
                width_loss = torch.zeros(1, device=width.features.device, dtype=torch.float32)

                gt_dense_scalar = _indice_to_scalar(dense_label.indices, [rotations_score.batch_size] + rotations_score.spatial_shape)
                idx_query = npi.indices(gt_dense_scalar.cpu().numpy(), pred_scalar.cpu().numpy(), missing=-1)
                good_query = idx_query != -1

                dense_label = dense_label.features.squeeze(1)[idx_query[good_query]]
                dense_rotations = dense_rotations.features.squeeze(1)[idx_query[good_query]]
                dense_width = dense_width.features.squeeze(1)[idx_query[good_query]]
                dense_offset = dense_offset.features.squeeze(1)[idx_query[good_query]]
                offset = offset.features.squeeze(1)[good_query]
                rotations_score = rotations_score.features.squeeze(1)[good_query]
                rotations = rotations.features.squeeze(1)[good_query]
                width = width.features.squeeze(1)[good_query]

                offset_loss = F.mse_loss(torch.sigmoid(offset), dense_offset, reduction="none").sum(dim=-1)

                rotations_score_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    rotations_score, dense_label * torch.max(torch.abs(torch.sum(F.normalize(rotations, dim=1) * dense_rotations[:, :4], dim=1)),
                                                             torch.abs(torch.sum(F.normalize(rotations, dim=1) * dense_rotations[:, 4:], dim=1))),
                    reduction="none")

                rotations_loss = _rot_loss_fn(F.normalize(rotations, dim=1), dense_rotations)

                width_loss = F.mse_loss(width, dense_width, reduction="none")

                total_loss = (rotations_score_loss + dense_label *
                              (rotations_loss + offset_loss + 0.01 * width_loss))

                voxel_losses["grasp_config"] = 0.1 * total_loss.mean()

                grasp_config_losses["rotations_score"] = rotations_score_loss.mean()
                grasp_config_losses["rotations"] = (dense_label * rotations_loss).mean()
                grasp_config_losses["width"] = (0.01 * dense_label * width_loss).mean()
                grasp_config_losses["offset"] = (dense_label * offset_loss).mean()
                voxel_losses["dense"] = cur_loss

        loss = sum(voxel_losses.values())
        logs = {
            **{
                f"voxel_loss_{resname}": voxel_losses[resname].item()
                for resname in voxel_losses
            },
            **{
                f"grasp_config_loss_{name}": grasp_config_losses[name].item()
                for name in grasp_config_losses
            },
        }
        return loss, logs


class Upsampler(torch.nn.Module):
    # nearest neighbor 2x upsampling for sparse 3D array

    def __init__(self):
        super().__init__()
        self.upsample_offsets = torch.nn.Parameter(
            torch.Tensor(
                [
                    [
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [1, 0, 1, 0],
                        [1, 1, 1, 0],
                    ]
                ]
            ).to(torch.int32),
            requires_grad=False,
        )
        self.upsample_mul = torch.nn.Parameter(
            torch.Tensor([[[2, 2, 2, 1]]]).to(torch.int32), requires_grad=False
        )

    def upsample_inds(self, voxel_inds):  # 形状（10,4）
        return (
            voxel_inds[:, None] * self.upsample_mul + self.upsample_offsets
        ).reshape(-1, 4)

    def upsample_feats(self, feats):
        return (
            feats[:, None]
            .repeat(1, 8, 1)
            .reshape(-1, feats.shape[-1])
            .to(torch.float32)
        )


def _rot_loss_fn(pred, gt):
    loss0 = _quat_loss_fn(pred, gt[:, :4])
    loss1 = _quat_loss_fn(pred, gt[:, 4:])
    return torch.min(loss0, loss1)


def _quat_loss_fn(pred, gt):
    return 1.0 - torch.abs(torch.sum(pred * gt, dim=1))

