import os

import numpy as np
import PIL.Image
import skimage.morphology
import torch
import spconv.pytorch as spconv
import torchvision


def load_data(data_dir, scene_name):
    input_data_path = os.path.join(data_dir, "scenes", f"{scene_name}.npz")
    label_data_path = os.path.join(data_dir, "scenes_label", f"{scene_name}.npz")
    with np.load(input_data_path) as input_data:
        tsdf_input = input_data["tsdf"].astype(np.float32)
        rgb_imgs_input = input_data["rgb_imgs"].astype(np.float32)
        depth_imgs_input = input_data["depth_imgs"]
        grid_input = input_data["grid"]
        extrinsics_input = input_data["extrinsics"].astype(np.float32)

    with np.load(label_data_path) as label_data:
        tsdf_label = label_data["tsdf"].astype(np.float32)
        label_label = label_data["label"].astype(np.float32)
        rotations_label = label_data["rotations"].astype(np.float32)
        width_label = label_data["width"].astype(np.float32)
        offset_label = label_data["offset"].astype(np.float32)
        index_label = label_data["index"]

    return grid_input, tsdf_label, label_label, rotations_label, width_label, offset_label, index_label


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_names,
        new_dataset_dir,
    ):
        self.scene_names = scene_names
        self.new_dataset_dir = new_dataset_dir

    def __len__(self):
        return len(self.scene_names)

    def getitem(self, ind, **kwargs):
        return self.__getitem__(ind, **kwargs)

    def __getitem__(self, ind):
        scene_name = self.scene_names[ind]

        (grid_input, tsdf_label, label_label, rotations_label, width_label,
         offset_label, index_label) = load_data(self.new_dataset_dir, scene_name)

        tsdf_04 = tsdf_label  # 形状（40，40，40）,为tsdf体，元素范围[-1，1]
        """可见区域体素包括表面占用体素和非表面占用体素"""
        occ_04 = np.abs(tsdf_04) < 0.999  # 表面占用体素,形状（40，40，40）,元素二值[Ture，False]
        seen_04 = tsdf_04 < 0.999  # 可见区域体素，形状（40，40，40）,元素二值[Ture，False]

        # seems like a bug -- dilation should happen before cropping
        occ_08 = skimage.morphology.dilation(occ_04, selem=np.ones((3, 3, 3)))  # 用滑动窗口膨胀占用体素，增加表面的占用体素
        not_occ_08 = seen_04 & ~occ_08
        occ_08 = occ_08[::2, ::2, ::2]  # 表面占用体素,形状（20，20，20），元素二值[Ture，False]
        not_occ_08 = not_occ_08[::2, ::2, ::2]
        seen_08 = occ_08 | not_occ_08  # 可见区域体素，形状（20，20，20）,元素二值[Ture，False]

        occ_16 = skimage.morphology.dilation(occ_08, selem=np.ones((3, 3, 3)))
        not_occ_16 = seen_08 & ~occ_16
        occ_16 = occ_16[::2, ::2, ::2]  # 表面占用体素,形状（10，10，10），元素二值[Ture，False]
        not_occ_16 = not_occ_16[::2, ::2, ::2]
        seen_16 = occ_16 | not_occ_16  # 可见区域体素，形状（10，10，10）,元素二值[Ture，False]

        spatial_shape_04 = occ_04.shape  # （40，40，40）
        spatial_shape_08 = occ_08.shape  # （20，20，20）
        spatial_shape_16 = occ_16.shape  # （10，10，10）

        inds_04_index_label = np.argwhere(index_label == 1)  # 形状（i,3）

        """对所有体素求损失"""
        inds_04 = np.argwhere(tsdf_04 > -2)
        inds_08 = np.argwhere(seen_08 > -2)
        inds_16 = np.argwhere(seen_16 > -2)
        """  """

        tsdf_04 = tsdf_04[inds_04[:, 0], inds_04[:, 1], inds_04[:, 2]]  # 形状（i04,）
        tsdf_04_label_label = label_label[inds_04_index_label[:, 0], inds_04_index_label[:, 1], inds_04_index_label[:, 2]]
        tsdf_04_rotations_label = rotations_label[inds_04_index_label[:, 0], inds_04_index_label[:, 1], inds_04_index_label[:, 2]].reshape(-1, 8)
        tsdf_04_width_label = width_label[inds_04_index_label[:, 0], inds_04_index_label[:, 1], inds_04_index_label[:, 2]]
        tsdf_04_offset_label = offset_label[inds_04_index_label[:, 0], inds_04_index_label[:, 1], inds_04_index_label[:, 2]]

        occ_08 = occ_08[inds_08[:, 0], inds_08[:, 1], inds_08[:, 2]].astype(np.float32)  # 形状（i08,）
        occ_16 = occ_16[inds_16[:, 0], inds_16[:, 1], inds_16[:, 2]].astype(np.float32)  # 形状（i16,）

        batch_size = 1
        tsdf_04 = spconv.SparseConvTensor(  # 形状（i04，1），（i04，4）,（3,）,(1,)
            torch.from_numpy(tsdf_04[:, None]), torch.cat([torch.zeros(inds_04.shape[0], 1), torch.from_numpy(inds_04)], dim=1).int(), 
            spatial_shape_04, batch_size)
        tsdf_04_label_label = spconv.SparseConvTensor(  # 形状（i，1），（i，4）,（3,）,(1,)
            torch.from_numpy(tsdf_04_label_label[:, None]),
            torch.cat([torch.zeros(inds_04_index_label.shape[0], 1), torch.from_numpy(inds_04_index_label)], dim=1).int(),
            spatial_shape_04, batch_size)
        tsdf_04_rotations_label = spconv.SparseConvTensor(  # 形状（i，8），（i，4）,（3,）,(1,)
            torch.from_numpy(tsdf_04_rotations_label),
            torch.cat([torch.zeros(inds_04_index_label.shape[0], 1), torch.from_numpy(inds_04_index_label)], dim=1).int(),
            spatial_shape_04, batch_size)
        tsdf_04_width_label = spconv.SparseConvTensor(  # 形状（i，1），（i，4）,（3,）,(1,)
            torch.from_numpy(tsdf_04_width_label[:, None]),
            torch.cat([torch.zeros(inds_04_index_label.shape[0], 1), torch.from_numpy(inds_04_index_label)], dim=1).int(),
            spatial_shape_04, batch_size)
        tsdf_04_offset_label = spconv.SparseConvTensor(  # 形状（i，3），（i，4）,（3,）,(1,)
            torch.from_numpy(tsdf_04_offset_label),
            torch.cat([torch.zeros(inds_04_index_label.shape[0], 1), torch.from_numpy(inds_04_index_label)], dim=1).int(),
            spatial_shape_04, batch_size)

        occ_08 = spconv.SparseConvTensor(  # 形状（i08，1），（i08，4）,（3,）,(1,)
            torch.from_numpy(occ_08[:, None]), torch.cat([torch.zeros(inds_08.shape[0], 1), torch.from_numpy(inds_08)], dim=1).int(), 
            spatial_shape_08, batch_size)
        occ_16 = spconv.SparseConvTensor(  # 形状（i16，1），（i16，4）,（3,）,(1,)
            torch.from_numpy(occ_16[:, None]), torch.cat([torch.zeros(inds_16.shape[0], 1), torch.from_numpy(inds_16)], dim=1).int(), 
            spatial_shape_16, batch_size)

        # generate dense initial grid
        x = torch.arange(seen_16.shape[0], dtype=torch.int32)  # 10
        y = torch.arange(seen_16.shape[1], dtype=torch.int32)  # 10
        z = torch.arange(seen_16.shape[2], dtype=torch.int32)  # 10
        # x = torch.arange(40, dtype=torch.int32)  # 40
        # y = torch.arange(40, dtype=torch.int32)  # 40
        # z = torch.arange(40, dtype=torch.int32)  # 40
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        input_voxels_16 = torch.stack(  # 在最后新增一个维度进行拼接，形状（10*10*10=1000，3）
            (xx.flatten(), yy.flatten(), zz.flatten()), dim=-1
        )

        scene = {
            "input_voxels_16": input_voxels_16,  # 形状（10*10*10=1000，3）
            "grid_input": grid_input,  # (1,40,40,40)
            "voxel_gt_dense": tsdf_04,  # spconv.SparseConvTensor(  # 形状（i04，1），（i04，4）,（3,）,(1,)
            "voxel_gt_label": tsdf_04_label_label,  # spconv.SparseConvTensor(  # 形状（i，1），（i，4）,（3,）,(1,)
            "voxel_gt_rotations": tsdf_04_rotations_label,  # spconv.SparseConvTensor(  # 形状（i，8），（i，4）,（3,）,(1,)
            "voxel_gt_width": tsdf_04_width_label,  # spconv.SparseConvTensor(  # 形状（i，1），（i，4）,（3,）,(1,)
            "voxel_gt_offset": tsdf_04_offset_label,  # spconv.SparseConvTensor(  # 形状（i，3），（i，4）,（3,）,(1,)
            "voxel_gt_medium": occ_08,  # spconv.SparseConvTensor(  # 形状（i08，1），（i08，4）,（3,）,(1,)
            "voxel_gt_coarse": occ_16,  # spconv.SparseConvTensor(  # 形状（i16，1），（i16，4）,（3,）,(1,)
        }
        return scene

