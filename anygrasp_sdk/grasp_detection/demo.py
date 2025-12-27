import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from types import SimpleNamespace
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup


def create_anygrasp(
    checkpoint_path: str,
    max_gripper_width: float = 0.1,
    gripper_height: float = 0.03,
    top_down_grasp: bool = False,
    debug: bool = False,
):

    cfgs = SimpleNamespace(
        checkpoint_path=checkpoint_path,
        max_gripper_width=max_gripper_width,
        gripper_height=gripper_height,
        top_down_grasp=top_down_grasp,
        debug=debug,
    )

    cfgs.max_gripper_width = max(0.0, min(0.1, cfgs.max_gripper_width))

    anygrasp = AnyGrasp(cfgs)
    return anygrasp


def demo(data_dir):
    anygrasp = create_anygrasp(
        checkpoint_path="log/checkpoint_detection.tar",
        max_gripper_width=0.1,
        gripper_height=0.03,
        top_down_grasp=False,
        debug=True
    )
    anygrasp.load_net()

    # get data
    colors = np.array(Image.open(os.path.join(data_dir, 'rgb_image.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth_image.png')))
    # print(colors.max(),colors.min(),depths.max(),depths.min())
    # get camera intrinsics
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    fx, fy = 540.0, 540.0
    cx, cy = 320.0, 240.0
    scale = 1000.0
    # set workspace to filter output grasps
    xmin, xmax = -0.3, 0.3
    ymin, ymax = -0.3, 0.3
    zmin, zmax = 0.0, 1.5
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    # print(points.min(axis=0), points.max(axis=0))

    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    best_translation = gg_pick[0].translation.copy()
    # best_rotation = gg_pick[0].rotation_matrix
    # print("best_translation", isinstance(best_translation, np.ndarray))
    # print("best_rotation", isinstance(best_rotation, np.ndarray))
    # print(gg_pick.scores)
    print(gg_pick[0].translation)
    # print('grasp score:', gg_pick[0].score)

    # visualization
    # if cfgs.debug:
    if True:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud])
        o3d.visualization.draw_geometries([grippers[0], cloud])


if __name__ == '__main__':
    
    demo('./example_data/')
