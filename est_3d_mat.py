import numpy as np
import argparse
from skimage import io
import open3d as o3d
import os
import copy
from pcd_regist_utils import *
from config import Config

"""
Open3d Documentation: Global Registration
Global registration + ICP refine
scene => ply file(point cloud)
voxel size = 0.01
1. get keypoints => scene_kp, scene_fpfh = preprocess_point_cloud(scene, voxel_size)
2. ransac global registration => result_ransac = execute_global_registration(scene1_kp, scene2_kp, scene1_fpfh, scene2_fpfh, voxel_size)
3. refine Iterative closest point(ICP) => refine_registration(scene1, scene2, result_ransac.transformation)

"""

def test_odometry():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Data')
    parser.add_argument('--source_fnum', type=int, default=10)
    parser.add_argument('--target_fnum', type=int, default=20)
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--center_move', type=bool, default=False)
    args = parser.parse_args()
    
    dataset_dir = '../rgbd_dataset/RGBD/{}/'.format(args.dataset)
    rgb_dir = dataset_dir + 'rgb/'
    depth_dir = dataset_dir + 'depth/'
    calib_dir = dataset_dir + 'calib/'
    
    source_rgb_file = rgb_dir + '{}.png'.format(args.source_fnum)
    source_depth_file = depth_dir + '{}.png'.format(args.source_fnum)

    target_rgb_file = rgb_dir + '{}.png'.format(args.target_fnum)
    target_depth_file = depth_dir + '{}.png'.format(args.target_fnum)
    
    config = Config()
    
    # read image
    source_rgb = o3d.io.read_image(source_rgb_file)
    source_depth = o3d.io.read_image(source_depth_file)

    target_rgb = o3d.io.read_image(target_rgb_file)
    target_depth = o3d.io.read_image(target_depth_file)
    
    # set pinhole camera
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(config.width, config.height, config.fu, config.fv, config.cu, config.cv)

    # source
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_rgb, source_depth)
    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)

    # target
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            target_rgb, target_depth)
    
    # compute odometry from source and target rgbd image
    option = o3d.odometry.OdometryOption()
    odo_init = np.identity(4)
    option.min_depth = config.min_high
    option.max_depth = config.max_high

    [success_color_term, trans_color_term, info] = o3d.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
            odo_init, o3d.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    [success_hybrid_term, trans_hybrid_term, info] = o3d.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
            odo_init, o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    if success_color_term:
        print('using RGBD odometry')
        print(trans_color_term)  # matrix 4 * 4(R, t)
        source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, 
                                                                            pinhole_camera_intrinsic)
        source_pcd_color_term.transform(trans_color_term)
        target_pcd = source_pcd_color_term
        
    if success_hybrid_term:
        print('using RGBD odometry')
        print(trans_hybrid_term)
        source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        target_pcd = source_pcd_hybrid_term
    
    if args.center_move:
        target_pcd = get_bbox_and_aabox(target_pcd)
        

    o3d.io.write_point_cloud('source.ply', source_pcd)
    o3d.io.write_point_cloud('target.ply', target_pcd)

    

def test():
    # after detection
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Data')
    parser.add_argument('--source_fnum', type=int, default=10)
    parser.add_argument('--target_fnum', type=int, default=20)
    parser.add_argument('--voxel_size', type=float, default=0.05)
    args = parser.parse_args()

    dataset_dir = '../rgbd_dataset/SHAPE/{}/'.format(args.dataset_name)
    rgb_dir = dataset_dir + 'detect_rgb/'
    depth_dir = dataset_dir + 'detect_depth/'
    better_depth_dir = dataset_dir + 'better_depth/'
    calib_dir = dataset_dir + 'calib/'
    
    os.makedirs(better_depth_dir, exist_ok=True)

    source_rgb_file = rgb_dir + '{}.png'.format(args.source_fnum)
    source_depth_file = depth_dir + '{}.png'.format(args.source_fnum)

    target_rgb_file = rgb_dir + '{}.png'.format(args.target_fnum)
    target_depth_file = depth_dir + '{}.png'.format(args.target_fnum)

    source_depth_image = io.imread(source_depth_file)
    source_depth_mask = source_depth_image < 400
    source_depth_image = np.multiply(source_depth_image, source_depth_mask)
    
    target_depth_image = io.imread(target_depth_file)
    target_depth_mask = target_depth_image < 400
    target_depth_image = np.multiply(target_depth_image, target_depth_mask)
    
    source_depth_file = better_depth_dir + '{}.png'.format(args.source_fnum)
    target_depth_file = better_depth_dir + '{}.png'.format(args.target_fnum)
    
    io.imsave(source_depth_file, source_depth_image)
    io.imsave(target_depth_file, target_depth_image)
    
    config = Config()


    source_rgb = o3d.io.read_image(source_rgb_file)
    source_depth = o3d.io.read_image(source_depth_file)

    target_rgb = o3d.io.read_image(target_rgb_file)
    target_depth = o3d.io.read_image(target_depth_file)
    # pinhole_camera_intrinsic = np.array([[fu, 0, cu],
    #                                    [0, fv, cv],
    #                                    [0, 0,  1]])

    # pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(config.width, config.height, config.fu, config.fv, config.cu, config.cv)
    
    max_nn = 50
    voxel_size = args.voxel_size
    
    # source
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_rgb, source_depth)
    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)

    # target
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            target_rgb, target_depth)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, pinhole_camera_intrinsic)
    
    # estimate normals
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=max_nn))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=max_nn))
    
    # preprocess data
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    
    # execute global registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        
    # Iterative Closest Point(refine)
    result_icp = refine_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, voxel_size, result_ransac)
        
    ans_pcd = copy.deepcopy(source_pcd)
    ans_pcd.transform(result_icp.transformation)
    print()
    print('--results---')
    print(result_icp.correspondence_set)
    print(result_icp.fitness)
    print(result_icp.inlier_rmse)
    print(result_icp.transformation)  # transfomation matrix
    print()
    

    o3d.io.write_point_cloud('source.ply', ans_pcd)
    o3d.io.write_point_cloud('target.ply', target_pcd)
    

# main function
# Base matching and successive macthing
# Base matching
def base_matching():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Data')
    parser.add_argument('--source_fnum', type=int, default=10)
    parser.add_argument('--target_fnums', type=str, default='20/30/40/50/60/70/80')
    parser.add_argument('--voxel_size', type=float, default=0.1)
    args = parser.parse_args()
    
    source_files, target_files, target_fnums, pinhole_camera_intrinsic= Initialize_base(args)
    source_pcd = create_point_cloud(source_files, pinhole_camera_intrinsic)[0]
    target_pcds = create_point_cloud(target_files, pinhole_camera_intrinsic)
    
    """
    o3d.io.write_point_cloud('source.ply', source_pcd)
    for idx, target_pcd in enumerate(target_pcds):
        o3d.io.write_point_cloud('target{}.ply'.format(idx), target_pcd)
    """   
    
    cloudify_integ_point_cloud(source_pcd, target_pcds, args.voxel_size, target_fnums)
    
# successive matching
def successive_matching():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Data')
    parser.add_argument('--start_fnum', type=int, default=1)
    parser.add_argument('--diff_fnum', type=int, default=10)
    parser.add_argument('--div_num', type=int, default=5)
    parser.add_argument('--voxel_size', type=float, default=0.1)
    
    args = parser.parse_args()
    
    dataset_dir = '../rgbd_dataset/RGBD/{}/'.format(args.dataset)
    rgb_dir = dataset_dir + 'rgb/'
    depth_dir = dataset_dir + 'depth/'
    
    nums = [args.start_fnum + args.diff_fnum * i for i in range(args.div_num)]
    rgb_files = [rgb_dir + '{}.png'.format(n) for n in nums]
    depth_files = [depth_dir + '{}.png'.format(n) for n in nums]
    
    # Hyper parameters for point cloud registration
    voxel_size = args.voxel_size
    max_nn = 50
    
    config = Config()
    
    # set pinhole camera intrinsic
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(config.width, config.height, config.fu, config.fv, config.cu, config.cv)
    
    for i in range(len(rgb_files) - 1):
        # get source data
        source_rgb = o3d.io.read_image(rgb_files[i])
        source_depth = o3d.io.read_image(depth_files[i])
        
        # create source point cloud from rgbd image
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_rgb, source_depth)
        source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)
        
        # get target data
        target_rgb = o3d.io.read_image(rgb_files[i + 1])
        target_depth = o3d.io.read_image(depth_files[i + 1])
        
        # create target point cloud from rgbd image
        target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_rgb, target_depth)
        target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, pinhole_camera_intrinsic)
        
        # estimate normals(preprocess)
        source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=max_nn))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=max_nn))
        
        # down sampling
        source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
        
        # execute global registration
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        
        # Iterative Closest Point(refine)
        result_icp = refine_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, voxel_size, result_ransac)
        
        # desctibe result's info
        print()
        print('--results---')
        print(result_icp.correspondence_set)
        print(result_icp.fitness)
        print(result_icp.inlier_rmse)
        print(result_icp.transformation)  # transfomation matrix
        print()
        
        ans_pcd = copy.deepcopy(source_pcd)
        ans_pcd.transform(result_icp.transformation)
        
        o3d.io.write_point_cloud('./SmnetData/success{}.ply'.format(i), ans_pcd)
        
    
    
    
## helper function for main function ##    
def Initialize_base(args):
    dataset_dir = '../rgbd_dataset/RGBD/{}/'.format(args.dataset)
    rgb_dir = dataset_dir + 'rgb/'
    depth_dir = dataset_dir + 'depth/'
    
    source_fnum = args.source_fnum
    target_fnums = [int(a) for a in args.target_fnums.split('/')]

    source_rgb_file = [rgb_dir + '{}.png'.format(source_fnum)]
    source_depth_file = [depth_dir + '{}.png'.format(source_fnum)]

    target_rgb_files = [rgb_dir + '{}.png'.format(tn) for tn in target_fnums]
    target_depth_files = [depth_dir + '{}.png'.format(tn) for tn in target_fnums]
    
    source_files = (source_rgb_file, source_depth_file)
    target_files = (target_rgb_files, target_depth_files)
    
    # Realsense configuration
    config = Config()
    
    # intrinsic parameters
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(config.width, config.height, config.fu, config.fv, config.cu, config.cv)
    
    return source_files, target_files, target_fnums, pinhole_camera_intrinsic


def create_point_cloud(files, pinhole_camera_intrinsic):
    pcds = []
    rgb_files, depth_files = files
    
    for rgb_file, depth_file in zip(rgb_files, depth_files):
        rgb = o3d.io.read_image(rgb_file)
        depth = o3d.io.read_image(depth_file)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
        pcds.append(pcd)
        
    return pcds
    


def cloudify_integ_point_cloud(source_pcd, target_pcds, voxel_size, fnums):
    max_nn = 50
    # estimate source normals
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=max_nn))
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    for fnum, target_pcd in zip(fnums, target_pcds):
        # estimate target normals
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=max_nn))
        target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
        
        # execute global registration
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        
        # Iterative Closest Point(refine)
        result_icp = refine_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, voxel_size, result_ransac)
        
        ans_pcd = copy.deepcopy(source_pcd)
        ans_pcd.transform(result_icp.transformation)
        print()
        print('--results---')
        print(result_icp.correspondence_set)
        print(result_icp.fitness)
        print(result_icp.inlier_rmse)
        print(result_icp.transformation)  # transfomation matrix
        print()
        
        o3d.io.write_point_cloud('./SmnetData/base{}.ply'.format(fnum), ans_pcd)
        
        
if __name__ == '__main__':
    test()
    # test_odometry()
    # base_matching()
    # successive_matching()