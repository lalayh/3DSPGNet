import time
import numpy as np
import trimesh
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.utils import visual
from anygrasp_sdk.grasp_detection.demo import create_anygrasp


LOW_TH = 0.5


class ANYGrasp(object):
    def __init__(self, checkpoint_path, max_gripper_width=0.1, gripper_height=0.03, top_down_grasp=False, debug=True,
                 visualize=False, resolution=40, **kwargs):
        self.net = create_anygrasp(checkpoint_path=checkpoint_path, max_gripper_width=max_gripper_width,
                                   gripper_height=gripper_height, top_down_grasp=top_down_grasp, debug=debug)
        self.net.load_net()
        self.visualize = visualize
        self.resolution = resolution

    def __call__(self, intrinsic=None, extrinsic=None, rgb_imgs=None, depth_imgs=None, scene_mesh=None, aff_kwargs={}):
        assert intrinsic.shape == (3, 3)
        assert extrinsic.shape == (4, 4)
        assert rgb_imgs.shape == (1, 480, 640, 3)
        assert depth_imgs.shape == (1, 480, 640)

        tic = time.time()
        qual_vol = np.zeros([self.resolution, self.resolution, self.resolution], dtype=np.float32)
        rot_vol = np.zeros([self.resolution, self.resolution, self.resolution, 4], dtype=np.float32)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, 0.3, self.resolution,
                                                          **aff_kwargs)
        colors = rgb_imgs[0] / 255.0
        depths = depth_imgs[0]
        # print(colors.max(),colors.min(),depths.max(),depths.min())
        # get camera intrinsics
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        # set workspace to filter output grasps
        xmin, xmax = -0.3, 0.3
        ymin, ymax = -0.3, 0.3
        zmin, zmax = 0.0, 1.5
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        # get point cloud
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # set your workspace to crop point cloud
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)
        # print(points.min(axis=0), points.max(axis=0))

        gg, cloud = self.net.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False,
                                       collision_detection=True)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')

        gg = gg.nms().sort_by_score()
        gg_pick = gg[0:20]
        # print(gg_pick.scores)
        # print('grasp score:', gg_pick[0].score)
        R_delta = Rotation.from_euler('y', 90, degrees=True)
        T_task_cam = Transform.from_matrix(extrinsic).inverse()
        grasps = []
        scores = []
        for i in range(len(gg_pick)):
            best_translation = gg_pick[i].translation.copy()
            best_rotation = gg_pick[i].rotation_matrix.copy()
            best_score = gg_pick[i].score
            ori = Rotation.from_matrix(best_rotation)
            T_cam_grasp = Transform(ori * R_delta, best_translation)
            T_task_grasp = T_task_cam * T_cam_grasp
            if np.all((T_task_grasp.translation > 0.0) & (T_task_grasp.translation < 0.3)):
                grasps = [Grasp(T_task_grasp, 0.08)]
                scores = [best_score]
                break

        # best_translation = gg_pick[0].translation.copy()
        # best_rotation = gg_pick[0].rotation_matrix.copy()
        # ori = Rotation.from_matrix(best_rotation)
        # T_cam_grasp = Transform(ori, best_translation)

        # T_task_grasp = T_task_cam * T_cam_grasp
        # grasps = [Grasp(T_task_grasp, 0.08)]
        # scores = [gg_pick[0].score]

        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        if self.visualize:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            composed_scene = trimesh.Scene(colored_scene_mesh)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            return grasps, scores, toc, composed_scene
        else:
            return grasps, scores, toc
