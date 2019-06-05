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

voxel_scale = 1.0
voxel_sz = 0.01

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

print(len(data_path_rgb), len(data_path_depth))

pinhole_camera_intrinsic = read_pinhole_camera_intrinsic("camera_ycb.json")

bbox = np.loadtxt("bbox.txt",delimiter=' ')
bbox = np.array(bbox)
min_x, min_y, min_z = bbox[7]
max_x, max_y, max_z = bbox[0]

bbox = build_frame(min_x, max_x, min_y, max_y, min_z, max_z)

current_bbox = PointCloud()
current_bbox.points = Vector3dVector(bbox)

for i in range(395, 1000):
    current_r, current_t = load_pose(i)
    project(data_root, data_path_rgb[i], bbox, current_r, current_t, cam_intr, i)