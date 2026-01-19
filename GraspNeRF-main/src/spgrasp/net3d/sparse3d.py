import inspect

import torch
from torch import nn
import spconv.pytorch as spconv


def xyzb2bxyz(indices):  # 形状（6912*2，4）
    new_indices = torch.zeros_like(indices)
    new_indices[:, 0] = indices[:, 3]
    new_indices[:, 1] = indices[:, 0]
    new_indices[:, 2] = indices[:, 1]
    new_indices[:, 3] = indices[:, 2]
    return new_indices


def bxyz2xyzb(indices):
    new_indices = torch.zeros_like(indices)
    new_indices[:, 0] = indices[:, 1]
    new_indices[:, 1] = indices[:, 2]
    new_indices[:, 2] = indices[:, 3]
    new_indices[:, 3] = indices[:, 0]
    return new_indices


def combineSparseConvTensor(xs, device):  # 形状list，[occ_16,occ_16]
    features_batch = []
    indices_batch = []
    spatial_shape = xs[0].spatial_shape
    batch_size = len(xs)
    for i, x in enumerate(xs):
        features_batch.append(x.features.to(device))
        inds = x.indices
        inds[:, 0] = i
        indices_batch.append(inds.to(device))
    features_batch = torch.cat(features_batch, dim=0)
    indices_batch = torch.cat(indices_batch, dim=0)

    spconvTensor = spconv.SparseConvTensor(  # 形状（i16*2，1），（i16*2，4）,（3,）,(1,)
        features_batch, indices_batch, 
        spatial_shape, batch_size)

    return spconvTensor


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper


def autocast_norm(layer_class):
    class AutocastNorm(layer_class):
        def forward(self, input):
            if input.dtype == torch.float16:
                output = super().forward(input.float()).half()
            else:
                output = super().forward(input)
            return output
    return AutocastNorm


class SparseResNet(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=32):
        super().__init__()
        self.spconv1 = spconv.SparseSequential(
                spconv.SparseConv3d(input_dim, hidden_dim, 3, 1, 1), # just like nn.Conv3d but don't support group
                nn.BatchNorm1d(hidden_dim), # non-spatial layers can be used directly in SparseSequential.
                nn.ReLU())
        self.block1 = SparseBasicBlock(hidden_dim, hidden_dim, indice_key='block1')
        self.block2 = SparseBasicBlock(hidden_dim, hidden_dim, indice_key='block2')
        self.spconv2 = spconv.SparseSequential(
                spconv.SparseConv3d(hidden_dim, output_dim, 3, 1, 1),
                nn.BatchNorm1d(output_dim),
                nn.ReLU())

    def forward(self, x):
        x = self.spconv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.spconv2(x)
        return [x]# .dense()


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 indice_key=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        # out.features += identity.features
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=False,
        indice_key=indice_key)