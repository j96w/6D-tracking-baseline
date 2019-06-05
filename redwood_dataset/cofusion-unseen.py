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
import math
from transformations import euler_matrix
from open3d import Image as IMG

gap = 2
voxel_scale = 1.0
voxel_sz = 0.01

cam_intr = np.loadtxt("camera-redwood.txt",delimiter=' ')
data_root = '/Users/jeremywang/Desktop/00704'

files = os.listdir('{0}/rgb'.format(data_root))
data_path_rgb = []
data_path_rgb_id = []
for item in files:
    if item[0] != '.':
        data_path_rgb.append(item)
        data_path_rgb_id.append(int(item[:7]))
data_path_rgb = np.array(data_path_rgb)
data_path_rgb_id = np.array(data_path_rgb_id)
temp = data_path_rgb_id.argsort()
data_path_rgb = data_path_rgb[temp]
print(data_path_rgb)

files = os.listdir('{0}/depth'.format(data_root))
data_path_depth = []
data_path_depth_id = []
for item in files:
    if item[0] != '.':
        data_path_depth.append(item)
        data_path_depth_id.append(int(item[:7]))
data_path_depth = np.array(data_path_depth)
data_path_depth_id = np.array(data_path_depth_id)
temp = data_path_depth_id.argsort()
data_path_depth = data_path_depth[temp]
print(data_path_depth)

print(len(data_path_rgb), len(data_path_depth))

pinhole_camera_intrinsic = read_pinhole_camera_intrinsic("camera_redwood.json")

min_x = -0.17
max_x = 0.17
min_y = -0.17
max_y = 0.17
min_z = -0.1
max_z = 0.1
bbox =[[min_x, min_y, min_z],
       [min_x, min_y, max_z],
       [min_x, max_y, min_z],
       [min_x, max_y, max_z],
       [max_x, min_y, min_z],
       [max_x, min_y, max_z],
       [max_x, max_y, min_z],
       [max_x, max_y, max_z]]
bbox = np.array(bbox)

trans_x = -0.08
trans_y = 0.09
trans_z = 0.65
current_pose = euler_matrix(-math.pi / 10.0, 0, math.pi / 10.0)
current_pose[0][3] = trans_x
current_pose[1][3] = trans_y
current_pose[2][3] = trans_z

# color_image = np.array(Image.open("{0}/rgb/{1}".format(data_root, data_path_rgb[0])).convert('RGB'))
# print(color_image)

# color_raw = IMG(color_image)
# depth_raw = read_image("{0}/depth/{1}".format(data_root, data_path_depth[0]))

# rgbd_image = create_rgbd_image_from_color_and_depth(color_raw, depth_raw)

# plt.subplot(1, 2, 1)
# plt.title('Redwood grayscale image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title('Redwood depth image')
# plt.imshow(rgbd_image.depth)
# plt.show()

current_bbox = PointCloud()
current_bbox.points = Vector3dVector(bbox)
# current_bbox.transform(current_pose)

current_pose = np.array(current_pose)
current_r = current_pose[:3,:3]
current_t = current_pose[:3,-1]

i = 0
vol_bnds = load_init_voxel(min_x, max_x, min_y, max_y, min_z, max_z)
tsdf_vol = fusion.TSDFVolume(vol_bnds,voxel_size=voxel_sz)
load_init_pc(tsdf_vol, data_root, data_path_rgb[i], data_path_depth[i], bbox, 1.1, cam_intr, current_r, current_t)

verts,faces,norms,colors = tsdf_vol.get_mesh()
first_p = PointCloud()
first_p.points = Vector3dVector(verts)
first_p.normals = Vector3dVector(norms)
first_p.colors = Vector3dVector(colors/255.)

draw_geometries([first_p, current_bbox])
output_pose(i, current_r, current_t)

for i in range(1, 1800):
    print(i)
    next_p = load_next_pc(data_root, data_path_rgb[i], data_path_depth[i], bbox, 1.1, cam_intr, current_r, current_t)

    # draw_geometries([first_p, next_p, current_bbox])
    current_r, current_t = ICP_color(first_p, next_p, current_r, current_t)

    if i % 20 == 0:
        apply_fusion(tsdf_vol, data_root, data_path_rgb[i], data_path_depth[i], bbox, 1.1, cam_intr, current_r, current_t)
        verts,faces,norms,colors = tsdf_vol.get_mesh()
        first_p = PointCloud()
        first_p.points = Vector3dVector(verts)
        first_p.normals = Vector3dVector(norms)
        first_p.colors = Vector3dVector(colors/255.)
        draw_geometries([first_p, next_p, current_bbox])

    print(current_r, current_t)
    output_pose(i, current_r, current_t)

