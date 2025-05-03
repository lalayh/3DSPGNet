import torch
import torch.nn as nn
from torch.nn import functional as F
import spconv.pytorch as spconv
from spconv.pytorch.tables import AddTable

import numpy as np

from spgrasp.net3d.sparse3d import SparseBasicBlock, change_default_args, autocast_norm


class SCN3d(nn.Module):
    def __init__(self, post_deform=False, **kwargs):
        super(SCN3d, self).__init__()

        channels = kwargs['channels']  # [128, 256, 512]
        hidden_depth = kwargs['hidden_depth']  # 128
        output_depth = kwargs['output_depth']  # 48
        self.output_depth = output_depth
        self.channels = channels

        self.sync_bn = True
        self.global_avg = post_deform
        self.post_deform = post_deform
        
        if self.sync_bn == True:
            BatchNorm1d = autocast_norm(change_default_args(eps=1e-3, momentum=0.01)(nn.SyncBatchNorm))
            BatchNorm3d = autocast_norm(change_default_args(eps=1e-3, momentum=0.01)(nn.SyncBatchNorm))
        else:
            BatchNorm1d = (change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d))
            BatchNorm3d = (change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm3d))
        LayerNorm = autocast_norm(change_default_args(eps=1e-3)(nn.LayerNorm))
        SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
        SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
        SparseInverseConv3d = change_default_args(bias=False)(spconv.SparseInverseConv3d)

        self.stem = nn.Sequential(
            nn.Linear(kwargs["input_depth"], channels[0]), 
            BatchNorm1d(channels[0]),
            nn.ReLU(True)
        )

        self.sp_convs = nn.ModuleList()
        for i in range(len(channels)-1):
            indice_key = 'cp%d'%(i+1) 
            sp_conv = spconv.SparseSequential(
                # FIXME: whether to use SparseMaxPool3d?
                SpConv3d(channels[i], channels[i+1], 3, stride=2,
                        bias=False, padding=1, indice_key=indice_key),
                BatchNorm1d(channels[i+1]),
                nn.ReLU(True)
            )
            self.sp_convs.append(sp_conv)

        if self.global_avg:
            self.pool_scales = [1, 2, 3]
            self.global_convs = nn.ModuleList()
            for i in range(len(self.pool_scales)):
                self.global_convs.append(nn.Conv3d(channels[-1], channels[-1], kernel_size=1))
            self.global_norm = nn.Sequential(
                    BatchNorm1d(channels[-1]*len(self.pool_scales)),
                    nn.ReLU(True))

        if self.post_deform == False:
            self.upconvs = nn.ModuleList([
                spconv.SparseSequential(
                    SubMConv3d(hidden_depth, output_depth, 3, 1, padding=1, bias=False, indice_key="up0"),
                    BatchNorm1d(output_depth),  # 96
                    nn.ReLU(True),
                    SparseBasicBlock(output_depth, output_depth))
            ])
        else:
            self.upconvs = nn.ModuleList([
                spconv.SparseSequential(
                    SpConv3d(hidden_depth, output_depth, 3, 1, padding=1, bias=False, indice_key="up0"),
                    BatchNorm1d(output_depth),
                    nn.ReLU(True),
                    SparseBasicBlock(output_depth, output_depth))
            ])
        
        for i in range(1, len(channels)):
            self.upconvs.append(
                spconv.SparseSequential(
                    SparseInverseConv3d(hidden_depth, hidden_depth, 3, indice_key="cp%d"%i),
                    BatchNorm1d(hidden_depth),
                    nn.ReLU(True))
            )

        self.lateral_attns = nn.ModuleList()
        for i, channel in enumerate(channels):
            if self.global_avg and i == len(channels)-1:
                channel = channel * (1 + len(self.pool_scales))
            self.lateral_attns.append(
                spconv.SparseSequential(
                    SubMConv3d(channel, hidden_depth, 1, indice_key="lsubm%d"%i),
                    BatchNorm1d(hidden_depth),
                    nn.ReLU(True))
            )
        
        self.sp_add = AddTable()

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, voxel_dim):

        # ========== Encoder ==========
        voxel_features, voxel_coords_bxyz, batch_size = input.features, input.indices, input.batch_size 
        voxel_features = self.stem(voxel_features)

        spconv_tensor = spconv.SparseConvTensor(voxel_features, voxel_coords_bxyz, voxel_dim, batch_size)

        feats = [spconv_tensor]

        for i in range(len(self.channels)-1):
            spconv_tensor = self.sp_convs[i](spconv_tensor)
            feats.append(spconv_tensor)

        # ========== Decoder ==========

        if self.global_avg:
            inputs = feats[-1]
            inputs_dense = inputs.dense()
            input_size = np.array(inputs.spatial_shape)
            pools = []
            for i, pool_scale in enumerate(self.pool_scales):
                output_size = pool_scale
                stride = (input_size / output_size).astype(np.int8)
                kernel_size = input_size - (output_size - 1) * stride
                out = F.avg_pool3d(inputs_dense, kernel_size=tuple(kernel_size), stride=tuple(stride), ceil_mode=False)
                out = self.global_convs[i](out)
                out = F.interpolate(out, input_size.tolist(), mode='nearest')
                pools.append(out)
            pools = torch.cat(pools, dim=1)
            valid = ~ ((inputs_dense == 0).all(1).unsqueeze(1))

            valid = valid.transpose(0, 1).flatten(1).transpose(0, 1)
            features = pools.transpose(0, 1).flatten(1).transpose(0, 1)
            valid_features = features[torch.nonzero(valid[:, 0]).squeeze(1)]
            
            outputs = inputs.replace_feature(torch.cat([inputs.features, self.global_norm(valid_features)], dim=1))
            feats[-1] = outputs
        
        x = None
        # x = self.lateral_attns[len(feats)-1](feats[len(feats)-1])
        for i in range(len(feats)-1, 0, -1):
            x = self.sp_add([x, self.lateral_attns[i](feats[i])]) if x is not None else self.lateral_attns[i](feats[i])
            x = self.upconvs[i](x)

        x = self.sp_add([x, self.lateral_attns[0](feats[0])])
        out = self.upconvs[0](x)

        return out


# class Vgn(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv0 = conv_stride(16, 16, 5)
#         self.conv1 = conv_stride(16, 32, 3)
#         self.conv2 = conv_stride(32, 64, 3)
#         self.conv3 = conv(64, 64, 3)
#         self.conv4 = conv(64, 32, 3)
#         self.conv5 = conv(32, 16, 5)
#
#     def forward(self, x):
#         x = self.conv0(x)
#         x = F.relu(x)
#
#         x = self.conv1(x)
#         x = F.relu(x)
#
#         x = self.conv2(x)
#         x = F.relu(x)
#
#         x = self.conv3(x)
#         x = F.relu(x)
#
#         x = F.interpolate(x, 10)
#         x = self.conv4(x)
#         x = F.relu(x)
#
#         x = F.interpolate(x, 20)
#         x = self.conv5(x)
#         x = F.relu(x)
#
#         x = F.interpolate(x, 40)
#
#         return x
#
#
# def conv(in_channels, out_channels, kernel_size):
#     return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
#
#
# def conv_stride(in_channels, out_channels, kernel_size):
#     return nn.Conv3d(
#         in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2
#     )
