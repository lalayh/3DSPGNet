import time
import cv2
from pathlib import Path
import numpy as np
import trimesh
from scipy import ndimage
import torch
import yaml
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.utils import visual
from vgn.utils.implicit import as_mesh
from nr.network.renderer import name2network


class GraspNeRFPlanner(object):

    def set_params(self, args):
        # self.args = args
        self.voxel_size = 0.3 / 40
        self.bbox3d = [[-0.15, -0.15, -0.0503], [0.15, 0.15, 0.2497]]
        # self.bbox3d = [[0.0, 0.0, -0.0503], [0.3, 0.3, 0.2497]]
        self.tsdf_thres_high = 0
        self.tsdf_thres_low = -0.85
        #
        # self.renderer_root_dir = self.args.renderer_root_dir
        tp, split, scene_type, scene_split, scene_id, background_size = args.database_name.split('/')
        background, size = background_size.split('_')
        # self.split = split
        self.tp = tp
        self.downSample = float(size)
        tp2wh = {
            'vgn_syn': (640, 360)
            # 'vgn_syn': (640, 480)
        }
        src_wh = tp2wh[tp]
        self.img_wh = (np.array(src_wh) * self.downSample).astype(int)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.K = np.array([[892.62, 0.0, 639.5],
                           [0.0, 892.62, 359.5],
                           [0.0, 0.0, 1.0]])
        # self.K[:2] = self.K[:2] * self.downSample
        # if self.tp == 'vgn_syn':
        #     self.K[:2] /= 2
        # self.depth_thres = {
        #     'vgn_syn': 0.8,
        # }
        #
        # if args.object_set == "graspnet":
        #     dir_name = "pile_graspnet_test"
        # else:
        #     if self.args.scene == "pile":
        #         dir_name = "pile_pile_test_200"
        #     elif self.args.scene == "packed":
        #         dir_name = "packed_packed_test_200"
        #     elif self.args.scene == "single":
        #         dir_name = "single_single_test_200"
        #
        # scene_root_dir = os.path.join(self.renderer_root_dir, "data/mesh_pose_list", dir_name)
        # self.mesh_pose_list = [i for i in sorted(os.listdir(scene_root_dir))]
        # self.depth_root_dir = ""
        # self.depth_list = []

    def __init__(self, args=None, cfg_fn=None, best=False, visualize=False):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.net = load_network(model_path, self.device, model_type=model_type)
        # self.net.eval()
        self.best = best
        self.visualize = visualize

        # default_render_cfg = {
        #     'min_wn': 3,  # working view number
        #     'ref_pad_interval': 16,  # input image size should be multiple of 16
        #     'use_src_imgs': False,  # use source images to construct cost volume or not
        #     'cost_volume_nn_num': 3,  # number of source views used in cost volume
        #     'use_depth': True,  # use colmap depth in rendering or not,
        # }
        # load render cfg
        if cfg_fn is None:
            self.set_params(args)
            cfg = load_cfg(args.cfg_fn)
        else:
            cfg = load_cfg(cfg_fn)

        # print(f"[I] GraspNeRFPlanner: using ckpt: {cfg['name']}")
        # render_cfg = cfg['train_dataset_cfg'] if 'train_dataset_cfg' in cfg else {}
        # render_cfg = {**default_render_cfg, **render_cfg}
        cfg['render_rgb'] = False  # only for training. Disable in grasping.
        # load model
        self.net = name2network[cfg['network']](cfg)
        ckpt_filename = 'model_best'
        ckpt = torch.load(Path('src/nr/ckpt') / f'{ckpt_filename}.pth')
        self.net.load_state_dict(ckpt['network_state_dict'])
        self.net.cuda()
        self.net.eval()
        self.step = ckpt["step"]
        # self.output_dir = debug_dir
        # if debug_dir is not None:
        #     if not Path(debug_dir).exists():
        #         Path(debug_dir).mkdir(parents=True)
        # self.loss = VGNLoss({})
        # self.num_input_views = render_cfg['num_input_views']
        # print(f"[I] GraspNeRFPlanner: load model at step {self.step} of best metric {ckpt['best_para']}")

    def get_image(self, img_id, rgb_imgs):
        img = rgb_imgs[img_id]
        img = cv2.resize(img, self.img_wh)
        return np.asarray(img, dtype=np.float32)

    def get_pose(self, img_id):
        poses_ori = np.load(Path("src/nr/ckpt") / 'camera_pose.npy')
        poses = [np.linalg.inv(p @ self.blender2opencv)[:3, :] for p in poses_ori]
        return poses[img_id].astype(np.float32).copy()

    def get_K(self, img_id):
        return self.K.astype(np.float32).copy()

    # def get_depth_range(self, img_id, round_idx, fixed=False):
    #     if fixed:
    #         return np.array([0.2, 0.8])
    #     depth = self.get_depth(img_id, round_idx)
    #     nf = [max(0, np.min(depth)), min(self.depth_thres[self.tp], np.max(depth))]
    #     return np.array(nf)

    def __call__(self, n=None, intrinsic=None, extrinsic=None, rgb_imgs=None, scene_mesh=None, aff_kwargs={}):
        assert intrinsic.shape == (3, 3)
        assert extrinsic.shape == (n, 7)
        assert rgb_imgs.shape == (n, 480, 640, 3)

        # images = (rgb_imgs.astype(np.float32) / 255).transpose([0, 3, 1, 2])
        images = [self.get_image(i, rgb_imgs) for i in range(n)]
        images = color_map_forward(np.stack(images, 0)).transpose([0, 3, 1, 2])

        # extrinsics = np.stack([Transform.from_list(extrinsic[i]).as_matrix() for i in range(n)], 0)
        extrinsics = np.stack([self.get_pose(i) for i in [2, 6, 10, 14, 18, 22]], 0)

        # intrinsics = np.stack([intrinsic for i in range(n)], 0)
        intrinsics = np.stack([self.get_K(i) for i in range(n)], 0)

        depth_range = np.asarray([np.array([0.2, 0.8]) for i in range(n)], dtype=np.float32)
        tic = time.time()
        tsdf_vol, qual_vol_ori, rot_vol_ori, width_vol_ori = self.core(images, extrinsics, intrinsics, depth_range, self.bbox3d)
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol_ori, rot_vol_ori, width_vol_ori,
                                               tsdf_thres_high=self.tsdf_thres_high, tsdf_thres_low=self.tsdf_thres_low)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(
                qual_vol, rot_vol.transpose(1, 2, 3, 0),
                scene_mesh, 0.3, 40, **aff_kwargs)

        grasps, scores, indexs = select(qual_vol.copy(), rot_vol, width_vol)
        toc = time.time() - tic
        grasps, scores, indexs = np.asarray(grasps), np.asarray(scores), np.asarray(indexs)

        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            grasps = [from_voxel_coordinates(g, self.voxel_size) for g in grasps[p]]
            scores = scores[p]
            indexs = indexs[p]

        if self.visualize:
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            composed_scene = trimesh.Scene(colored_scene_mesh)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            return grasps, scores, toc, composed_scene
        else:
            return grasps, scores, toc

        # tic = time.time()
        # qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        #
        # qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        # qual_vol = bound(qual_vol, voxel_size)

        # if self.visualize:
        #     colored_scene_mesh = visual.affordance_visual(
        #         qual_vol, rot_vol.transpose(1, 2, 3, 0),
        #         scene_mesh, 0.3, 40, **aff_kwargs)

        # grasps, scores = select(qual_vol.copy(), rot_vol, width_vol, threshold=self.qual_th,
        #                         force_detection=self.force_detection, max_filter_size=8 if self.visualize else 4)
        # toc = time.time() - tic

        # grasps, scores = np.asarray(grasps), np.asarray(scores)

        # if len(grasps) > 0:
        #     if self.best:
        #         p = np.arange(len(grasps))
        #     else:
        #         p = np.random.permutation(len(grasps))
        #
        #     grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
        #     scores = scores[p]

        # if self.visualize:
        #     grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
        #     composed_scene = trimesh.Scene(colored_scene_mesh)
        #     for i, g_mesh in enumerate(grasp_mesh_list):
        #         composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
        #     return grasps, scores, toc, composed_scene
        # else:
        #     return grasps, scores, toc

    def core(self,
             images: np.ndarray,
             extrinsics: np.ndarray,
             intrinsics: np.ndarray,
             depth_range=[0.2, 0.8],
             bbox3d=[[-0.15, -0.15, -0.05], [0.15, 0.15, 0.25]], gt_info=None, que_id=0):
        """
        @args
            images: np array of shape (3, 3, h, w), image in RGB format
            extrinsics: np array of shape (3, 4, 4), the transformation matrix from world to camera
            intrinsics: np array of shape (3, 3, 3)
        @rets
            volume, label, rot, width: np array of shape (1, 1, res, res, res)
        """
        _, _, h, w = images.shape
        assert h % 32 == 0 and w % 32 == 0
        extrinsics = extrinsics[:, :3, :]
        que_imgs_info = build_render_imgs_info(extrinsics[que_id], intrinsics[que_id], (h, w), depth_range[que_id])
        src_imgs_info = {'imgs': images, 'poses': extrinsics.astype(np.float32), 'Ks': intrinsics.astype(np.float32),
                         'depth_range': depth_range.astype(np.float32),
                         'bbox3d': np.array(bbox3d)}

        ref_imgs_info = src_imgs_info.copy()
        num_views = images.shape[0]
        ref_imgs_info['nn_ids'] = np.arange(num_views).repeat(num_views, 0)
        data = {'step': self.step, 'eval': True, 'full_vol': True,
                'que_imgs_info': to_cuda(imgs_info_to_torch(que_imgs_info)),
                'src_imgs_info': to_cuda(imgs_info_to_torch(src_imgs_info)),
                'ref_imgs_info': to_cuda(imgs_info_to_torch(ref_imgs_info))}
        # if not gt_info:
        #     data['full_vol'] = True
        # else:
        #     data['grasp_info'] = to_cuda(grasp_info_to_torch(gt_info))

        with torch.no_grad():
            render_info = self.net(data)

        # if gt_info:
        #     return self.loss(render_info, data, self.step, False)

        label, rot, width = render_info['vgn_pred']

        return render_info['volume'].cpu().numpy(), label.cpu().numpy(), rot.cpu().numpy(), width.cpu().numpy()


def build_render_imgs_info(que_pose,que_K,que_shape,que_depth_range):
    h, w = que_shape
    h, w = int(h), int(w)
    que_coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1)
    que_coords = que_coords.reshape([1, -1, 2]).astype(np.float32)
    return {'poses': que_pose.astype(np.float32)[None,:,:],  # 1,3,4
            'Ks': que_K.astype(np.float32)[None,:,:],  # 1,3,3
            'coords': que_coords,
            'depth_range': np.asarray(que_depth_range, np.float32)[None, :],
            'shape': (h,w)}


def to_cuda(data):
    if type(data) == list:
        results = []
        for i, item in enumerate(data):
            results.append(to_cuda(item))
        return results
    elif type(data) == dict:
        results = {}
        for k, v in data.items():
            results[k] = to_cuda(v)
        return results
    elif type(data).__name__ == "Tensor" or type(data).__name__=="Parameter":
        return data.cuda()
    else:
        return data


def imgs_info_to_torch(imgs_info):
    for k, v in imgs_info.items():
        if isinstance(v,np.ndarray):
            imgs_info[k] = torch.from_numpy(v).float()
    return imgs_info


def process(
        tsdf_vol,
        qual_vol,
        rot_vol,
        width_vol,
        gaussian_filter_sigma=1.0,
        min_width=1.33,
        max_width=9.33,
        tsdf_thres_high=0.5,
        tsdf_thres_low=1e-3
):
    tsdf_vol = tsdf_vol.squeeze()
    qual_vol = qual_vol.squeeze()
    rot_vol = rot_vol.squeeze()
    width_vol = width_vol.squeeze()
    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > tsdf_thres_high
    inside_voxels = np.logical_and(tsdf_thres_low < tsdf_vol, tsdf_vol < tsdf_thres_high)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
    qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)

    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores, indexs = [], [], []
    for index in np.argwhere(mask):
        indexs.append(index)
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    return grasps, scores, indexs


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    rot = rot_vol[:, i, j, k]
    ori = Rotation.from_quat(rot)
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def color_map_forward(rgb):
    return rgb.astype(np.float32) / 255
