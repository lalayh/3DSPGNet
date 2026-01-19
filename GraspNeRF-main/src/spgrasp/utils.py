import os

import numpy as np
import open3d as o3d
import skimage.measure


def log_transform(x, shift=1):
    # https://github.com/magicleap/Atlas
    """rescales TSDF values to weight voxels near the surface more than close
    to the truncation distance"""
    return x.sign() * (1 + x.abs() / shift).log()


def to_mesh(vol, voxel_size=1, origin=np.zeros(3), level=0, mask=None):
    verts, faces, _, _ = skimage.measure.marching_cubes(vol, level=level, mask=mask)
    verts *= voxel_size
    verts += origin

    bad_face_inds = np.any(np.isnan(verts[faces]), axis=(1, 2))
    faces = faces[~bad_face_inds]

    bad_vert_inds = np.any(np.isnan(verts), axis=-1)
    reindex = np.cumsum(~bad_vert_inds) - 1
    faces = reindex[faces]
    verts = verts[~bad_vert_inds]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
    )
    mesh.compute_vertex_normals()
    return mesh


def load_info_files(dataset_split_dir, split):
    with open(os.path.join(dataset_split_dir, f"{split}.txt"), "r") as f:
        scene_names = f.read().split()

    return scene_names

