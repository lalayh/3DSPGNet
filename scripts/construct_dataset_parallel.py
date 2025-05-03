import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm
import multiprocessing as mp

from vgn.io import *
from vgn.perception import *
from vgn.utils.misc import apply_noise


RESOLUTION = 40

def process_one_scene(args, f, df):
    if f.suffix != ".npz":
        return f.stem
    rgb_imgs_side, depth_imgs_side, extrinsics_side, _, depth_imgs, extrinsics = read_sensor_data(args.raw, f.stem)
    # add noise
    depth_imgs_side = np.array([apply_noise(x, args.add_noise) for x in depth_imgs_side])
    if args.single_view:
        tsdf = create_tsdf(size, RESOLUTION, depth_imgs_side[[0]], intrinsic, extrinsics_side[[0]])
    else:
        tsdf = create_tsdf(size, RESOLUTION, depth_imgs_side, intrinsic, extrinsics_side)
    grid = tsdf.get_grid()
    tsdf_real = tsdf.get_tsdf()
    write_voxel_grid(args.dataset, f.stem, grid, tsdf_real, rgb_imgs_side, depth_imgs_side, extrinsics_side)  # scenes中写入体素数据

    tsdf_voxel = create_tsdf(size, 40, depth_imgs, intrinsic, extrinsics)
    tsdf_voxel_real = tsdf_voxel.get_tsdf()
    tsdf_label = np.zeros([40, 40, 40]).astype(np.longlong)
    tsdf_rotations = np.zeros([40, 40, 40, 2, 4]).astype(np.single)
    tsdf_width = np.zeros([40, 40, 40]).astype(np.single)
    tsdf_offset = np.zeros([40, 40, 40, 3]).astype(np.single)  # 位置偏移量，大小为占一个体素边长的百分比
    tsdf_index = np.zeros([40, 40, 40]).astype(np.longlong)
    df_one_scene = df[df["scene_id"] == f.stem]
    for k in range(len(df_one_scene.index)):
        k_index = df_one_scene.index[k]
        ori = Rotation.from_quat(df_one_scene.loc[k_index, "qx":"qw"].to_numpy(np.single))
        pos = df_one_scene.loc[k_index, "i":"k"].to_numpy(np.single)
        width = df_one_scene.loc[k_index, "width"].astype(np.single)
        label = df_one_scene.loc[k_index, "label"].astype(np.longlong)
        index = (pos // 1).astype(np.longlong)  # 向下取整
        rotations_k = np.empty((2, 4), dtype=np.single)
        r = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations_k[0] = ori.as_quat()
        rotations_k[1] = (ori * r).as_quat()
        tsdf_label[index[0], index[1], index[2]] = label
        tsdf_rotations[index[0], index[1], index[2], :, :] = rotations_k
        tsdf_width[index[0], index[1], index[2]] = width
        tsdf_offset[index[0], index[1], index[2], :] = pos - index
        tsdf_index[index[0], index[1], index[2]] = 1
    write_tsdf_label(args.dataset, f.stem, tsdf_voxel_real, tsdf_label, tsdf_rotations, tsdf_width, tsdf_offset, tsdf_index)

    pc = tsdf.get_cloud()
    # crop surface and borders from point cloud
    lower = np.array([0.02 , 0.02 , 0.055])
    upper = np.array([0.28, 0.28, 0.3])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
    pc = pc.crop(bounding_box)
    pc = np.asarray(pc.points)
    write_point_cloud(args.dataset, f.stem, pc)  # point_clouds中写入点云数据
    return str(f.stem)

def log_result(result):
    g_num_completed_jobs.append(result)
    elapsed_time = time.time() - g_starting_time

    if len(g_num_completed_jobs) % 1000 == 0:
        msg = "%05d/%05d %s finished! " % (len(g_num_completed_jobs), g_num_total_jobs, result)
        msg = msg + 'Elapsed time: ' + \
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        print(msg)

def main(args):
    if args.single_view:
        print('Loading first view only!')
    # create directory of new dataset
    (args.dataset / "scenes").mkdir(parents=True)
    (args.dataset / "scenes_label").mkdir(parents=True)
    (args.dataset / "point_clouds").mkdir(parents=True)

    global g_num_completed_jobs
    global g_num_total_jobs
    global g_starting_time
    global size
    global intrinsic

    # load setup information
    size, intrinsic, _, finger_depth = read_setup(args.raw)
    assert np.isclose(size, 6.0 * finger_depth)
    voxel_size = size / RESOLUTION

    # create df
    df = read_df(args.raw)
    df["x"] /= voxel_size
    df["y"] /= voxel_size
    df["z"] /= voxel_size
    df["width"] /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k"})
    write_df(df, args.dataset)  # 修改grasps.csv文件，并写入new_dataset
    df = read_df(args.dataset)

    g_num_completed_jobs = []
    file_list = list((args.raw / "scenes").iterdir())
    g_num_total_jobs = len(file_list)
    g_starting_time = time.time()

    # create tsdfs and pcs

    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc)

        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        for f in file_list:
            pool.apply_async(func=process_one_scene, args=(args, f,), callback=log_result)
        pool.close()
        pool.join()
    else:
        for f in tqdm(file_list, total=len(file_list)):
            process_one_scene(args, f, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--single-view", action='store_true')
    parser.add_argument("--add-noise", type=str, default='')
    args = parser.parse_args()
    main(args)
