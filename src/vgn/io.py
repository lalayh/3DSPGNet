import json
import uuid

import numpy as np
import pandas as pd

from vgn.grasp import Grasp
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform


def write_setup(root, size, intrinsic, max_opening_width, finger_depth):
    data = {
        "size": size,
        "intrinsic": intrinsic.to_dict(),
        "max_opening_width": max_opening_width,
        "finger_depth": finger_depth,
    }
    write_json(data, root / "setup.json")


def read_setup(root):
    data = read_json(root / "setup.json")
    size = data["size"]
    intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
    max_opening_width = data["max_opening_width"]
    finger_depth = data["finger_depth"]
    return size, intrinsic, max_opening_width, finger_depth


def write_sensor_data(root, rgb_imgs_side, depth_imgs_side, extrinsics_side, rgb_imgs, depth_imgs, extrinsics):
    scene_id = uuid.uuid4().hex
    path_scenes = root / "scenes" / (scene_id + ".npz")
    path_scenes_label = root / "scenes_label" / (scene_id + ".npz")
    assert not path_scenes.exists()
    assert not path_scenes_label.exists()
    np.savez_compressed(path_scenes, rgb_imgs=rgb_imgs_side, depth_imgs=depth_imgs_side, extrinsics=extrinsics_side)
    np.savez_compressed(path_scenes_label, rgb_imgs=rgb_imgs, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id


def write_full_sensor_data(root, depth_imgs, extrinsics, scene_id=None):
    if scene_id is None:
        scene_id = uuid.uuid4().hex
    path = root / "full_scenes" / (scene_id + ".npz")
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id


def read_sensor_data(root, scene_id):
    data_scenes = np.load(root / "scenes" / (scene_id + ".npz"))
    data_scenes_label = np.load(root / "scenes_label" / (scene_id + ".npz"))
    return data_scenes["rgb_imgs"], data_scenes["depth_imgs"], data_scenes["extrinsics"], data_scenes_label["rgb_imgs"], data_scenes_label["depth_imgs"], data_scenes_label["extrinsics"]

def read_full_sensor_data(root, scene_id):
    data = np.load(root / "full_scenes" / (scene_id + ".npz"))
    return data["depth_imgs"], data["extrinsics"]


def write_grasp(root, scene_id, grasp, label):
    # TODO concurrent writes could be an issue
    csv_path = root / "grasps.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label)


def read_grasp(df, i):
    scene_id = df.loc[i, "scene_id"]
    orientation = Rotation.from_quat(df.loc[i, "qx":"qw"].to_numpy(np.double))
    position = df.loc[i, "x":"z"].to_numpy(np.double)
    width = df.loc[i, "width"]
    label = df.loc[i, "label"]
    grasp = Grasp(Transform(orientation, position), width)
    return scene_id, grasp, label


def read_df(root):
    return pd.read_csv(root / "grasps.csv")


def write_df(df, root):
    df.to_csv(root / "grasps.csv", index=False)


def write_voxel_grid(root, scene_id, voxel_grid, tsdf_real, rgb_imgs_side, depth_imgs_side, extrinsics_side):
    path = root / "scenes" / (scene_id + ".npz")
    np.savez_compressed(path, grid=voxel_grid, tsdf=tsdf_real, rgb_imgs=rgb_imgs_side, depth_imgs=depth_imgs_side, extrinsics=extrinsics_side)


def write_tsdf_label(root, scene_id, tsdf_real, tsdf_label, tsdf_rotations, tsdf_width, tsdf_offset, tsdf_index):
    path = root / "scenes_label" / (scene_id + ".npz")
    np.savez_compressed(path, tsdf=tsdf_real, label=tsdf_label, rotations=tsdf_rotations, width=tsdf_width, offset=tsdf_offset, index=tsdf_index)


def write_point_cloud(root, scene_id, point_cloud, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    np.savez_compressed(path, pc=point_cloud)


def write_meshpath_scale_pose(root, scene_id, meshpath_list, scale_list, pose_list, name="mesh_pose_list"):
    path = root / name / (scene_id + ".npz")
    np.savez_compressed(path, ml=meshpath_list, sl=scale_list, pl=pose_list)


def read_voxel_grid(root, scene_id):
    path = root / "scenes" / (scene_id + ".npz")
    return np.load(path)["grid"]


def read_point_cloud(root, scene_id, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    return np.load(path)["pc"]


def read_json(path):
    with path.open("r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def create_csv(path, columns):
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")
