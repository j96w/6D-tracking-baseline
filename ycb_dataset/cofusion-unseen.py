#!/usr/bin/env python

import numpy as np
import cv2
import time
import scipy.io as scio
from open3d import *
from utils import *
from PIL import Image
import scipy.misc
import os
import matplotlib.pyplot as plt
import copy
from open3d import Image as IMG
from fusion import *

gap = 1
voxel_scale = 1.0
voxel_sz = 0.005

cam_intr = np.loadtxt("camera-ycb.txt",delimiter=' ')
data_root = '/Users/jeremywang/Desktop/SSSIIIXXX/merge/syn_data/0'

files = os.listdir('{0}/rgb'.format(data_root))
data_path_rgb = []
for i in range(1, len(files)):
    data_path_rgb.append('{0}.png'.format(i))
print(data_path_rgb)

files = os.listdir('{0}/depth'.format(data_root))
data_path_depth = []
for i in range(1, len(files)):
    data_path_depth.append('{0}.png'.format(i))
print(data_path_depth)

files = os.listdir('{0}/pose'.format(data_root))
data_path_pose = []
for i in range(1, len(files)):
    data_path_pose.append('{0}_pose.npy'.format(i))
print(data_path_pose)

print(len(data_path_rgb), len(data_path_depth), len(data_path_pose))

pinhole_camera_intrinsic = read_pinhole_camera_intrinsic("camera_ycb.json")


bbox = np.loadtxt("bbox.txt",delimiter=' ')
bbox = np.array(bbox)
min_x, min_y, min_z = bbox[7]
max_x, max_y, max_z = bbox[0]
print(bbox)
print(min_x, max_x, min_y, max_y, min_z, max_z)


current_pose = np.load("{0}/pose/{1}".format(data_root, data_path_pose[395]), encoding='latin1')

current_bbox = PointCloud()
current_bbox.points = Vector3dVector(bbox)
# current_bbox.transform(pose_bbox)

current_r = current_pose[0][()]['R']
current_t = current_pose[0][()]['T']/1000.0
print(current_r, current_t)

i = 395
vol_bnds = load_init_voxel(min_x, max_x, min_y, max_y, min_z, max_z)
tsdf_vol = fusion.TSDFVolume(vol_bnds,voxel_size=voxel_sz)
load_init_pc(tsdf_vol, data_root, data_path_rgb[i], data_path_depth[i], bbox, 1.1, cam_intr, current_r, current_t)

verts,faces,norms,colors = tsdf_vol.get_mesh()
meshwrite('mesh/{0}.ply'.format('%04d' % 0), verts,faces,norms,colors)
first_p = PointCloud()
first_p.points = Vector3dVector(verts)
first_p.normals = Vector3dVector(norms)
first_p.colors = Vector3dVector(colors/255.)
# first_p = voxel_down_sample(first_p, voxel_size = 0.01)

# draw_geometries([first_p, current_bbox])
output_pose(i, current_r, current_t)

for i in range(396, 1095):
    print(i)
    next_p = load_next_pc(data_root, data_path_rgb[i], data_path_depth[i], bbox, 1.1, cam_intr, current_r, current_t)

    # draw_geometries([first_p, next_p, current_bbox])
    current_r, current_t = ICP_wo_color(first_p, next_p, current_r, current_t)

    if i % 10 == 0:
        apply_fusion(tsdf_vol, data_root, data_path_rgb[i], data_path_depth[i], bbox, 1.1, cam_intr, current_r, current_t)
        verts,faces,norms,colors = tsdf_vol.get_mesh()
        first_p = PointCloud()
        first_p.points = Vector3dVector(verts)
        first_p.normals = Vector3dVector(norms)
        first_p.colors = Vector3dVector(colors/255.)

        # draw_geometries([first_p])
        meshwrite('mesh/{0}.ply'.format('%04d' % i), verts,faces,norms,colors)

    print(current_r, current_t)
    output_pose(i, current_r, current_t)

