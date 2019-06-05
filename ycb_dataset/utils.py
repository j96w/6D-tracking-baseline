import numpy as np
import fusion
from open3d import *
from open3d import Image as IMG
from PIL import Image
import scipy.io as scio
import cv2
import matplotlib.pyplot as plt
import torch
import scipy.misc

border_list = [-1, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640
color = np.array([[255, 69, 0], [124, 252, 0], [0, 238, 238], [238, 238, 0], [155, 48, 255], [0, 0, 238], [255, 131, 250], [189, 183, 107], [165, 42, 42]])


def get_2dbbox(cloud, cam_intr):
    cam_fx = cam_intr[0][0]
    cam_fy = cam_intr[1][1]
    cam_cx = cam_intr[0][2]
    cam_cy = cam_intr[1][2]

    img = np.array([[[0, 0, 0] for i in range(640)] for j in range(480)]).astype(np.int32)

    rmin = 10000
    rmax = -10000
    cmin = 10000
    cmax = -10000

    for tg in cloud:
        p1 = int(tg[0] * cam_fx / tg[2] + cam_cx)
        p0 = int(tg[1] * cam_fy / tg[2] + cam_cy)
        # print(p0, p1)
        if p0 < rmin:
            rmin = p0
        if p0 > rmax:
            rmax = p0
        if p1 < cmin:
            cmin = p1
        if p1 > cmax:
            cmax = p1
    rmax += 1
    cmax += 1
    if rmin < 0:
        rmin = 0
    if cmin < 0:
        cmin = 0
    if rmax >= 480:
        rmax = 479
    if cmax >= 640:
        cmax = 639

    r_b = rmax - rmin
    #print(rmax - rmin, cmax - cmin)
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
        
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
        
    return rmin, rmax, cmin, cmax


def calculate_inside(cloud, limit, choose):
    ans = (cloud[:, 0] > limit[0]) * \
          (cloud[:, 0] < limit[1]) * \
          (cloud[:, 1] > limit[2]) * \
          (cloud[:, 1] < limit[3]) * \
          (cloud[:, 2] > limit[4]) * \
          (cloud[:, 2] < limit[5])

    # print(cloud)
    if len(ans.nonzero()[0]) == 0:
        cld = cloud
        cho = choose
    else:
        cld = cloud[ans.nonzero()]
        cho = choose[ans.nonzero()]

    # print(len(cld))
    return cld, cho


def search_fit(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]



def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax



def get_mask(depth, xmap, ymap, cam_intr, gt_r, gt_t, bbox):
    # print("MASK", bbox)
    cam_fx = cam_intr[0][0]
    cam_fy = cam_intr[1][1]
    cam_cx = cam_intr[0][2]
    cam_cy = cam_intr[1][2]

    mask = np.array([[0 for i in range(640)] for j in range(480)])

    depth_masked = depth.flatten()[:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[:, np.newaxis].astype(np.float32)
    pt2 = depth_masked
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)
    choose = np.concatenate((xmap_masked, ymap_masked), axis=1).astype(np.int32)

    cloud = np.dot(cloud - gt_t, gt_r)
    limit = search_fit(bbox)
    # print(limit)
    # print("pose", gt_r, gt_t)

    cloud, choose = calculate_inside(cloud, limit, choose)

    for tg in choose:
        mask[tg[0]][tg[1]] = 1

    return mask



def pre_load(data_root):
    bbox = {}
    mesh = {}

    obj_list = {}
    obj_list['real'] = {}
    obj_list['syn'] = {}

    obj_id = 1
    input_file = open("{0}/models/classes.txt".format(data_root), 'r')
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]

        input_file_2 = open("{0}/models/{1}/points.xyz".format(data_root, input_line), 'r')
        mesh[obj_id] = []
        while 1:
            input_line_2 = input_file_2.readline()
            if not input_line_2:
                break
            if input_line_2[-1:] == '\n':
                input_line_2 = input_line_2[:-1]
            input_line_2 = input_line_2.split(' ')
            mesh[obj_id].append([float(input_line_2[0]), float(input_line_2[1]), float(input_line_2[2])])
        input_file_2.close()
        mesh[obj_id] = np.array(mesh[obj_id])

        input_file_2 = open("{0}/models/{1}/bbox.xyz".format(data_root, input_line), 'r')
        bbox[obj_id] = []
        while 1:
            input_line_2 = input_file_2.readline()
            if not input_line_2:
                break
            if input_line_2[-1:] == '\n':
                input_line_2 = input_line_2[:-1]
            input_line_2 = input_line_2.split(' ')
            bbox[obj_id].append([float(input_line_2[0]), float(input_line_2[1]), float(input_line_2[2])])
        input_file_2.close()
        bbox[obj_id] = np.array(bbox[obj_id])

        obj_id += 1

    for i in range(1, 22):
        input_file = open("{0}/obj_list/{1}/{2}.txt".format(data_root, 'test', i), 'r')
        obj_list['real'][i] = {}
        obj_list['syn'][i] = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]

            if input_line[:8] == 'data_syn':
               obj_list['syn'][i].append(input_line)
            else:
                video_id = int(input_line[5:9])
                if not video_id in obj_list['real'][i]:
                   obj_list['real'][i][video_id] = []
                obj_list['real'][i][video_id].append(input_line)

        input_file.close()

    return bbox, mesh, obj_list



def load_init_voxel(min_x, max_x, min_y, max_y, min_z, max_z):
    vol_bnds = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])
    # # Read depth image and camera pose
    # depth_im = np.array(Image.open('{0}/data/{1}/{2}-depth.png'.format(data_root,'%04d' %  choose_video, '%06d' % img_id)))/10000. # depth is saved in 16-bit PNG in millimeters
    # # label = np.array(Image.open('{0}/{1}/{2}-label.png'.format(data_root, choose_video, '%06d' % img_id)))

    # meta = scio.loadmat('{0}/data/{1}/{2}-meta.mat'.format(data_root, '%04d' % choose_video, '%06d' % img_id))
    # obj_list = meta['cls_indexes'].flatten().astype(np.int32)
    # idx = np.where(obj_list == choose_obj)[0][0]

    # gt_r = meta['poses'][:, :, idx][:, 0:3]
    # gt_t = np.array(meta['poses'][:, :, idx][:, -1])

    # xmap = np.array([[j for i in range(640)] for j in range(480)])
    # ymap = np.array([[i for i in range(640)] for j in range(480)])

    # rmin, rmax, cmin, cmax = get_2dbbox(np.dot(bbox[choose_obj]*bbox_scale, gt_r.T) + gt_t, cam_intr)
    # mask = get_mask(depth_im[rmin:rmax, cmin:cmax], xmap[rmin:rmax, cmin:cmax], ymap[rmin:rmax, cmin:cmax], cam_intr, gt_r, gt_t, bbox[choose_obj]*bbox_scale)

    # depth_im = depth_im * mask # set invalid depth to 0 (specific to 7-scenes dataset)

    # cam_pose = np.zeros((4, 4))
    # cam_pose[:3, :3] = gt_r
    # cam_pose[:3, 3:4] = gt_t.reshape(3, 1)
    # cam_pose[3, 3] = 1.0
    # cam_pose = np.linalg.inv(cam_pose)

    # # Compute camera view frustum and extend convex hull
    # view_frust_pts = fusion.get_view_frustum(depth_im,cam_intr,cam_pose)
    # vol_bnds[:,0] = np.minimum(vol_bnds[:,0],np.amin(bbox,axis=1)) * voxel_scale
    # vol_bnds[:,1] = np.maximum(vol_bnds[:,1],np.amax(bbox,axis=1)) * voxel_scale

    return vol_bnds



def load_init_pc(tsdf_vol, data_root, rgb_name, dep_name, bbox, bbox_scale, cam_intr, current_r, current_t):
    # Read RGB-D image and camera pose
    color_image = np.array(Image.open("{0}/rgb/{1}".format(data_root, rgb_name)))
    depth_im = np.array(Image.open("{0}/depth/{1}".format(data_root, dep_name))).astype(np.float32)/1000.0 # depth is saved in 16-bit PNG in millimeters
    # label = np.array(Image.open('{0}/{1}/{2}-label.png'.format(data_root, choose_video, '%06d' % img_id)))

    # print(np.array(Image.open("{0}/depth/{1}".format(data_root, dep_name))).max(), depth_im.max())

    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

    # print(bbox[choose_obj])
    # print(np.dot(bbox*bbox_scale, current_r.T) + current_t)
    rmin, rmax, cmin, cmax = get_2dbbox(np.dot(bbox*bbox_scale, current_r.T) + current_t, cam_intr)
    # print(rmin, rmax, cmin, cmax)

    # plt.title('Depth image')
    # plt.imshow(depth_im)
    # plt.show()

    mask = get_mask(depth_im[rmin:rmax, cmin:cmax], xmap[rmin:rmax, cmin:cmax], ymap[rmin:rmax, cmin:cmax], cam_intr, current_r, current_t, bbox*bbox_scale)


    depth_im = depth_im * mask

    # plt.title('Depth image')
    # plt.imshow(depth_im)
    # plt.show()

    # color_raw = IMG(color_image)
    # depth_raw = IMG(depth_im.astype(np.float32))

    # rgbd_image = create_rgbd_image_from_color_and_depth(color_raw, depth_raw, depth_scale=1.0, convert_rgb_to_intensity=False)

    # print(rgbd_image)
    # plt.subplot(1, 2, 1)
    # plt.title('Color image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()

    # pinhole_camera_intrinsic = read_pinhole_camera_intrinsic("camera_redwood.json")

    # pcd = create_point_cloud_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)


    pose = np.zeros((4, 4))
    pose[:3, :3] = current_r
    pose[:3, 3:4] = current_t.reshape(3, 1)
    pose[3, 3] = 1.0
    pose = np.linalg.inv(pose)
    # pose = pose.tolist()
    # print("AAAAA", img_id)
    # print(pose)
    tsdf_vol.integrate(color_image,depth_im,cam_intr,pose,obs_weight=1.)


def load_next_pc(data_root, rgb_name, dep_name, bbox, bbox_scale, cam_intr, current_r, current_t):
    # Read RGB-D image and camera pose
    color_image = np.array(Image.open("{0}/rgb/{1}".format(data_root, rgb_name)))
    depth_im = np.array(Image.open("{0}/depth/{1}".format(data_root, dep_name)))/1000.0 # depth is saved in 16-bit PNG in millimeters

    # label = np.array(Image.open('{0}/{1}/{2}-label.png'.format(data_root, choose_video, '%06d' % img_id)))

    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

    # print(bbox[choose_obj])
    rmin, rmax, cmin, cmax = get_2dbbox(np.dot(bbox*bbox_scale, current_r.T) + current_t, cam_intr)
    mask = get_mask(depth_im[rmin:rmax, cmin:cmax], xmap[rmin:rmax, cmin:cmax], ymap[rmin:rmax, cmin:cmax], cam_intr, current_r, current_t, bbox*bbox_scale)


    depth_im = depth_im * mask

    # plt.title('Depth image')
    # plt.imshow(depth_im)
    # plt.show()

    color_raw = IMG(color_image)
    depth_raw = IMG(depth_im.astype(np.float32))

    rgbd_image = create_rgbd_image_from_color_and_depth(color_raw, depth_raw, depth_scale=1.0, convert_rgb_to_intensity=False)

    # print(rgbd_image)
    # plt.subplot(1, 2, 1)
    # plt.title('Color image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.title('Depth image')
    # plt.imshow(rgbd_image.depth)
    # plt.show()

    pinhole_camera_intrinsic = read_pinhole_camera_intrinsic("camera_ycb.json")

    pcd = create_point_cloud_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)


    pose = np.zeros((4, 4))
    pose[:3, :3] = current_r
    pose[:3, 3:4] = current_t.reshape(3, 1)
    pose[3, 3] = 1.0
    pose = np.linalg.inv(pose)
    pose = pose.tolist()
    # print("AAAAA", img_id)
    # print(pose)
    pcd.transform(pose)

    return pcd



def apply_fusion(tsdf_vol, data_root, rgb_name, dep_name, bbox, bbox_scale, cam_intr, current_r, current_t):

    color_image = np.array(Image.open("{0}/rgb/{1}".format(data_root, rgb_name)))
    depth_im = np.array(Image.open("{0}/depth/{1}".format(data_root, dep_name)))/1000.0 # depth is saved in 16-bit PNG in millimeters
    # label = np.array(Image.open('{0}/{1}/{2}-label.png'.format(data_root, choose_video, '%06d' % img_id)))

    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

    # print(bbox[choose_obj])
    rmin, rmax, cmin, cmax = get_2dbbox(np.dot(bbox*bbox_scale, current_r.T) + current_t, cam_intr)
    mask = get_mask(depth_im[rmin:rmax, cmin:cmax], xmap[rmin:rmax, cmin:cmax], ymap[rmin:rmax, cmin:cmax], cam_intr, current_r, current_t, bbox*bbox_scale)


    depth_im = depth_im * mask


    cam_pose = np.zeros((4, 4))
    cam_pose[:3, :3] = current_r
    cam_pose[:3, 3:4] = current_t.reshape(3, 1)
    cam_pose[3, 3] = 1.0
    cam_pose = np.linalg.inv(cam_pose)

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image,depth_im,cam_intr,cam_pose,obs_weight=1.)



def ICP_color(now_pc, next_pc, current_r, current_t):
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)

    # print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        # print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = voxel_down_sample(now_pc, radius)
        target_down = voxel_down_sample(next_pc, radius)

        # print("3-2. Estimate normal.")
        estimate_normals(source_down, KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))
        estimate_normals(target_down, KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))

        # print("3-3. Applying colored point cloud registration")
        result_icp = registration_colored_icp(source_down, target_down,
                radius, current_transformation,
                ICPConvergenceCriteria(relative_fitness = 1e-6,
                relative_rmse = 1e-6, max_iteration = iter))
        current_transformation = result_icp.transformation
        # print(result_icp)

    # print(result_icp.transformation)
    delta_r = result_icp.transformation[:3, :3]
    delta_t = result_icp.transformation[:3, -1]
    current_t = current_t + np.dot(current_r, delta_t.reshape(3, 1)).reshape(-1)
    current_r = np.dot(current_r, delta_r)

    return current_r, current_t

def ICP_wo_color(now_pc, next_pc, current_r, current_t):
    current_transformation = np.identity(4)
    radius = 0.01

    source_down = voxel_down_sample(now_pc, radius)
    target_down = voxel_down_sample(next_pc, radius)

    # print("3-2. Estimate normal.")
    estimate_normals(source_down, KDTreeSearchParamHybrid(radius = radius * 2, max_nn = 30))
    estimate_normals(target_down, KDTreeSearchParamHybrid(radius = radius * 2, max_nn = 30))

    result_icp = registration_icp(source_down, target_down, 0.01, current_transformation, TransformationEstimationPointToPlane())
    # print(result_icp)
    # draw_registration_result_original_color(source, target, result_icp.transformation)

    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017

    # print(result_icp.transformation)
    delta_r = result_icp.transformation[:3, :3]
    delta_t = result_icp.transformation[:3, -1]
    current_t = current_t + np.dot(current_r, delta_t.reshape(3, 1)).reshape(-1)
    current_r = np.dot(current_r, delta_r)

    return current_r, current_t

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    draw_geometries([source_temp, target])


def add(mesh, current_r, current_t, gt_r, gt_t):
    #print(current_r, current_t, gt_r, gt_t)

    pred = np.dot(mesh, current_r.T) + current_t
    gt = np.dot(mesh, gt_r.T) + gt_t

    dis = np.linalg.norm((pred - gt), axis=1).mean()
    
    return dis

def get_AUC(dis_box):
    dis_box = np.array(dis_box)
    space_my = 0.0
    space_all = 0.0
    for ki in range(1, 1000):
        space_all += 1.0
        kj = float(ki) / 10000.0

        tmp_my = 0.0
        tmp_my = float(len(np.where(dis_box <= kj)[0])) / float(len(dis_box))
        space_my += tmp_my

    return space_my/space_all

def output_pose(i, r, t):
    fw = open('temp/{0}.txt'.format(i), 'w')
    fw.write('{0} {1} {2}\n'.format(r[0, 0], r[0, 1], r[0, 2]))
    fw.write('{0} {1} {2}\n'.format(r[1, 0], r[1, 1], r[1, 2]))
    fw.write('{0} {1} {2}\n'.format(r[2, 0], r[2, 1], r[2, 2]))
    fw.write('{0} {1} {2}\n'.format(t[0], t[1], t[2]))
    fw.close()

def load_pose(i):
    current_r = []
    fw = open('temp/{0}.txt'.format(i), 'r')

    for i in range(3):
        input_line = fw.readline()
        input_line = input_line[:-1].split(' ')
        current_r.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])

    input_line = fw.readline()
    input_line = input_line[:-1].split(' ')
    current_t = [float(input_line[0]), float(input_line[1]), float(input_line[2])]

    current_r = np.array(current_r)
    current_t = np.array(current_t)

    print(current_r, current_t)

    return current_r, current_t

def build_frame(min_x, max_x, min_y, max_y, min_z, max_z):
    bbox = []
    for i in np.arange(min_x, max_x, 0.001):
        bbox.append([i, min_y, min_z])

    for i in np.arange(min_x, max_x, 0.001):
        bbox.append([i, min_y, max_z])

    for i in np.arange(min_x, max_x, 0.001):
        bbox.append([i, max_y, min_z])

    for i in np.arange(min_x, max_x, 0.001):
        bbox.append([i, max_y, max_z])



    for i in np.arange(min_y, max_y, 0.001):
        bbox.append([min_x, i, min_z])

    for i in np.arange(min_y, max_y, 0.001):
        bbox.append([min_x, i, max_z])

    for i in np.arange(min_y, max_y, 0.001):
        bbox.append([max_x, i, min_z])

    for i in np.arange(min_y, max_y, 0.001):
        bbox.append([max_x, i, max_z])


    for i in np.arange(min_z, max_z, 0.001):
        bbox.append([min_x, min_y, i])

    for i in np.arange(min_z, max_z, 0.001):
        bbox.append([min_x, max_y, i])

    for i in np.arange(min_z, max_z, 0.001):
        bbox.append([max_x, min_y, i])

    for i in np.arange(min_z, max_z, 0.001):
        bbox.append([max_x, max_y, i])

    bbox = np.array(bbox)

    return bbox

def project(data_root, rgb_name, bbox, current_r, current_t, cam_intr, i):
    img = np.array(Image.open("{0}/rgb/{1}".format(data_root, rgb_name)))
    bbox = np.dot(bbox, current_r.T) + current_t

    cam_fx = cam_intr[0][0]
    cam_fy = cam_intr[1][1]
    cam_cx = cam_intr[0][2]
    cam_cy = cam_intr[1][2]

    for tg in bbox:
        y = int(tg[0] * cam_fx / tg[2] + cam_cx)
        x = int(tg[1] * cam_fy / tg[2] + cam_cy)

        if x - 2 < 0 or x + 2 > 479 or y - 2 < 0 or y + 2 > 639:
            continue
        # print(x, y)
        img[x+2][y+2] = color[0]
        img[x+2][y+1] = color[0]
        img[x+2][y] = color[0]
        img[x+2][y-1] = color[0]
        img[x+2][y-2] = color[0]

        img[x+1][y+2] = color[0]
        img[x+1][y+1] = color[0]
        img[x+1][y] = color[0]
        img[x+1][y-1] = color[0]
        img[x+1][y-2] = color[0]

        img[x][y+2] = color[0]
        img[x][y+1] = color[0]
        img[x][y] = color[0]
        img[x][y-1] = color[0]
        img[x][y-2] = color[0]

        img[x-1][y+2] = color[0]
        img[x-1][y+1] = color[0]
        img[x-1][y] = color[0]
        img[x-1][y-1] = color[0]
        img[x-1][y-2] = color[0]

        img[x-2][y+2] = color[0]
        img[x-2][y+1] = color[0]
        img[x-2][y] = color[0]
        img[x-2][y-1] = color[0]
        img[x-2][y-2] = color[0]


    scipy.misc.imsave('results/{0}.png'.format(int(i)), img)






