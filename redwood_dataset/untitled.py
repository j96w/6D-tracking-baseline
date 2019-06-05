def ICP_wo_color(now_pc, next_pc, current_r, current_t):
    current_transformation = np.identity(4)
    radius = 0.01

    source_down = voxel_down_sample(now_pc, radius)
    target_down = voxel_down_sample(next_pc, radius)

    estimate_normals(source_down, KDTreeSearchParamHybrid(radius = radius * 2, max_nn = 30))
    estimate_normals(target_down, KDTreeSearchParamHybrid(radius = radius * 2, max_nn = 30))

    result_icp = registration_icp(source_down, target_down, 0.01, current_transformation, TransformationEstimationPointToPlane())

    delta_r = result_icp.transformation[:3, :3]
    delta_t = result_icp.transformation[:3, -1]
    current_t = current_t + np.dot(current_r, delta_t.reshape(3, 1)).reshape(-1)
    current_r = np.dot(current_r, delta_r)

    return current_r, current_t



# To get the now_pc and next_pc, there are two ways:
#(1) you can load from RGBD image by

from open3d import Image as IMG
from open3d import create_point_cloud_from_rgbd_image, create_rgbd_image_from_color_and_depth, read_pinhole_camera_intrinsic

color_image = np.array(Image.open("{0}/rgb/{1}".format(data_root, rgb_name)))
depth_im = np.array(Image.open("{0}/depth/{1}".format(data_root, dep_name)))/1000.0
color_raw = IMG(color_image)
depth_raw = IMG(depth_im.astype(np.float32))
rgbd_image = create_rgbd_image_from_color_and_depth(color_raw, depth_raw, depth_scale=1.0, convert_rgb_to_intensity=False)
pinhole_camera_intrinsic = read_pinhole_camera_intrinsic("camera_redwood.json")
new_pc = create_point_cloud_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)

# you can get the next_pc in the same way


#(2) direct change from .xyz
from open3d import PointCloud

new_pc = PointCloud()
new_pc.points = Vector3dVector(verts)
new_pc.normals = Vector3dVector(norms)
new_pc.colors = Vector3dVector(colors/255.)

# you can get the next_pc in the same way