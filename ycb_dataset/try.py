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

gap = 1
voxel_scale = 1.0
voxel_sz = 0.01

cam_intr = np.loadtxt("camera-ycb.txt",delimiter=' ')
data_root = '/Users/jeremywang/Desktop/SSSIIIXXX/sixd/output/render/01'

files = os.listdir('{0}/rgb'.format(data_root))
data_path_rgb = []
for i in range(1, len(files)):
        data_path_rgb.append('{0}.png'.format('%04d' % i))
print(data_path_rgb)

files = os.listdir('{0}/depth'.format(data_root))
data_path_depth = []
for i in range(1, len(files)):
        data_path_depth.append('{0}.png'.format('%04d' % i))
print(data_path_depth)

files = os.listdir('{0}/pose'.format(data_root))
data_path_pose = []
for i in range(1, len(files)):
        data_path_pose.append('{0}.npy'.format('%04d' % i))
print(data_path_pose)

print(len(data_path_rgb), len(data_path_depth), len(data_path_pose))

pinhole_camera_intrinsic = read_pinhole_camera_intrinsic("camera_ycb.json")


bbox = np.loadtxt("bbox.txt",delimiter=' ')
bbox = np.array(bbox)
min_x, min_y, min_z = bbox[7]
max_x, max_y, max_z = bbox[0]
print(bbox)
print(min_x, max_x, min_y, max_y, min_z, max_z)


current_pose = np.load("{0}/pose/{1}".format(data_root, data_path_pose[0]), encoding='latin1')

current_bbox = PointCloud()
current_bbox.points = Vector3dVector(bbox)
# current_bbox.transform(pose_bbox)

current_r = current_pose[()]['R']
current_t = current_pose[()]['T']/1000.0
print(current_r, current_t)

depth = np.array(Image.open("{0}/depth/{1}".format(data_root, data_path_depth[0])))
cam_scale = 1000.0


cam_fx = cam_intr[0][0]
cam_fy = cam_intr[1][1]
cam_cx = cam_intr[0][2]
cam_cy = cam_intr[1][2]

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

depth_masked = depth.flatten()[:, np.newaxis].astype(np.float32)
xmap_masked = xmap.flatten()[:, np.newaxis].astype(np.float32)
ymap_masked = ymap.flatten()[:, np.newaxis].astype(np.float32)

pt2 = depth_masked / cam_scale
pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
cloud = np.concatenate((pt0, pt1, pt2), axis=1)

bbox = np.dot(bbox, current_r.T) + current_t

fw = open('model_points.xyz', 'w')
for it in cloud:
   fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
fw.close()

fw = open('bbox.xyz', 'w')
for it in bbox:
   fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
fw.close()

