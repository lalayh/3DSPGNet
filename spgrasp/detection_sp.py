import time

import numpy as np
import trimesh
from scipy import ndimage
import torch
from spgrasp import lightningmodel
# from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation

from vgn.utils import visual
from vgn.utils.implicit import as_mesh
import torch.nn.functional as F


LOW_TH = 0.5


class SPG(object):
    def __init__(self, model_path, config, best=False, force_detection=False, qual_th=0.9, out_th=0.5,
                 visualize=False, resolution=40, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_model(model_path, config)
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        self.resolution = resolution

        x = torch.arange(self.resolution // 4, dtype=torch.int32)  # 10
        y = torch.arange(self.resolution // 4, dtype=torch.int32)  # 10
        z = torch.arange(self.resolution // 4, dtype=torch.int32)  # 10
        # x = torch.arange(40, dtype=torch.int32)  # 40
        # y = torch.arange(40, dtype=torch.int32)  # 40
        # z = torch.arange(40, dtype=torch.int32)  # 40
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        self.input_voxels_16 = torch.stack(  # 在最后新增一个维度进行拼接，形状（10*10*10=1000，3）
            (xx.flatten(), yy.flatten(), zz.flatten()), dim=-1
        )

    def __call__(self, state, scene_mesh=None, aff_kwargs={}):
        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / self.resolution
            size = 0.3
        else:
            tsdf_vol = state.tsdf.get_grid()
            voxel_size = state.tsdf.voxel_size
            size = state.tsdf.size

        tic = time.time()
        qual_vol, rot_vol, width_vol, offset_vol, tsdf_real_vol = predict(self.input_voxels_16, tsdf_vol, self.net.spgnet, self.device)
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        qual_vol = bound(qual_vol, voxel_size)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution,
                                                          **aff_kwargs)
        grasps, scores = select(qual_vol.copy(),
                                offset_vol, rot_vol,
                                width_vol, threshold=self.qual_th, force_detection=self.force_detection,
                                max_filter_size=8 if self.visualize else 4)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))

            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
            scores = scores[p]

        if self.visualize:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            composed_scene = trimesh.Scene(colored_scene_mesh)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            return grasps, scores, toc, composed_scene, tsdf_real_vol
        else:
            return grasps, scores, toc, tsdf_real_vol


def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    # avoid grasp out of bound [0.02  0.02  0.055]
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol


def predict(input_voxels_16, tsdf_vol, net, device):

    assert tsdf_vol.shape == (1, 40, 40, 40)

    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)
    voxel_inds_16 = torch.cat([input_voxels_16.to(device), torch.zeros([input_voxels_16.shape[0], 1], device=device)], dim=1).int()  # (1000,4)

    batch = {"grid_input": tsdf_vol}

    # with torch.cuda.amp.autocast():
    with torch.no_grad():
        voxel_outputs = net(batch, voxel_inds_16)

    rotations_score = torch.sigmoid(voxel_outputs["rotations_score"].features.squeeze(1)).cpu().numpy()

    offset = torch.sigmoid(voxel_outputs["offset"].features.squeeze(1)).cpu().numpy()  # (m,3)
    rotations = F.normalize(voxel_outputs["rotations"].features.squeeze(1), dim=1).cpu().numpy()  # (m,4)
    width = voxel_outputs["width"].features.squeeze(1).cpu().numpy()  # (m,)
    tsdf_real = 1.05 * torch.tanh(voxel_outputs["dense"].features).squeeze(-1).cpu().numpy()  # (m,)
    index = voxel_outputs["dense"].indices[:, 1:].cpu().numpy()  # (m,3)

    qual_vol = np.zeros(voxel_outputs["dense"].spatial_shape, dtype=np.float32)
    qual_vol[index[:, 0], index[:, 1], index[:, 2]] = rotations_score
    rot_vol = np.zeros(voxel_outputs["dense"].spatial_shape + [4], dtype=np.float32)
    rot_vol[index[:, 0], index[:, 1], index[:, 2]] = rotations
    width_vol = np.zeros(voxel_outputs["dense"].spatial_shape, dtype=np.float32)
    width_vol[index[:, 0], index[:, 1], index[:, 2]] = width
    # offset_vol = np.zeros(voxel_outputs["dense"].spatial_shape + [3], dtype=np.float32) + 0.5
    offset_vol = np.zeros(voxel_outputs["dense"].spatial_shape + [3], dtype=np.float32)
    offset_vol[index[:, 0], index[:, 1], index[:, 2]] = offset

    tsdf_real_vol = np.ones(voxel_outputs["dense"].spatial_shape, dtype=np.float32) * (-2)
    # tsdf_real_vol = np.ones(voxel_outputs["dense"].spatial_shape, dtype=np.float32) * np.nan
    tsdf_real_vol[index[:, 0], index[:, 1], index[:, 2]] = tsdf_real

    return qual_vol, rot_vol, width_vol, offset_vol, tsdf_real_vol


def process(
        tsdf_vol,
        qual_vol,
        rot_vol,
        width_vol,
        gaussian_filter_sigma=1.0,
        min_width=1.33,
        max_width=9.33,
        out_th=0.5
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )

    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, offset_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, offset_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]

    return sorted_grasps, sorted_scores


def select_index(qual_vol, offset_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    offset = offset_vol[i, j, k].astype(np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos + offset), width), score


def load_model(ckpt_file, config):
    model = lightningmodel.LightningModel.load_from_checkpoint(
        ckpt_file,
        config=config,
    )
    model = model.cuda()
    model = model.eval()
    model.requires_grad_(False)
    return model