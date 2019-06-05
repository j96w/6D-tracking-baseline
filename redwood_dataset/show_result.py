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

bbox = build_frame(min_x, max_x, min_y, max_y, min_z, max_z)

current_bbox = PointCloud()
current_bbox.points = Vector3dVector(bbox)

for i in range(0, 1094):
    current_r, current_t = load_pose(i)
    project(data_root, data_path_rgb[i], bbox, current_r, current_t, cam_intr, i)