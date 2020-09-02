import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pymagsac
from time import time

from copy import deepcopy


def create_rotation_matrix(axis_angles):
    # angle
    anglex = axis_angles[0]
    angley = axis_angles[1]
    anglez = axis_angles[2]

    # x axis rotation
    anglex = anglex * np.pi / 180.0
    Rx = np.array([[1, 0, 0],
                [0, np.cos(anglex), -np.sin(anglex)],
                [0, np.sin(anglex), np.cos(anglex)]])

    # y axis rotation
    angley = angley * np.pi / 180.0
    Ry = np.array([[np.cos(angley), 0, np.sin(angley)],
                [0, 1, 0],
                [-np.sin(angley), 0, np.cos(angley)]])

    # z axis rotation
    anglez = anglez * np.pi / 180
    Rz = np.array([[np.cos(anglez), -np.sin(anglez), 0],
                [np.sin(anglez), np.cos(anglez), 0],
                [0,              0,             1]])

    R = Rz.dot(Ry).dot(Rx)
    
    # print('Rz:', Rz)
    
    return R

def change_rotation_mat_for_open3d(R):
    R_s = np.zeros((4, 4))
    R_s[:3, :3] = R
    R_s[3, 3] = 1.0
    
    return R_s

def decolorize(img):
    return  cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
def draw_matches(kps1, kps2, tentatives, img1, img2, H, mask):
    if H is None:
        print ("No homography found")
        return
    matchesMask = mask.ravel().tolist()
    h,w,ch = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)
    # Ground truth transformation
    # dst_GT = cv2.perspectiveTransform(pts, H_gt)
    img2_tr = cv2.polylines(decolorize(img2),[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
    # img2_tr = cv2.polylines(deepcopy(img2_tr),[np.int32(dst_GT)],True,(0,255,0),3, cv2.LINE_AA)
    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor = (255,255,0), # draw matches in yellow color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img_out = cv2.drawMatches(decolorize(img1),kps1,img2_tr,kps2,tentatives,None,**draw_params)
    plt.figure()
    plt.imshow(img_out)
    plt.show()
    return


def verify_cv2(kps1, kps2, tentatives, th=1.0):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, th)
    print(H)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers')
    return H, mask

def verify_magsac(kps1, kps2, tentatives, th=1.0):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 2)
    H, mask = pymagsac.findHomography(src_pts, dst_pts, th)
    F, mask = pymagsac.findFundamentalMatrix(src_pts, dst_pts, th)
    print(H)
    print(F)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers')
    return H, mask

def verify_magsac_plus_plus(kps1, kps2, tentatives, th=1.0):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in tentatives]).reshape(-1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in tentatives]).reshape(-1, 2)
    H, mask = pymagsac.findHomography(src_pts, dst_pts, th)
    F, mask = pymagsac.findFundamentalMatrix(src_pts, dst_pts, th)
    print(H)
    print(F)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers')
    return H, mask

root_dir = '../rgbd_dataset/RGBD/'
dataset_name = 'Nami1'

dataset_dir = root_dir + dataset_name + '/rgb/'

start_img_path = dataset_dir + '11.png'
start_depth_path = root_dir + dataset_name + '/depth/11.png'
finish_img_path = dataset_dir + '51.png'
finish_depth_path = root_dir + dataset_name + '/depth/51.png'


start_img = cv2.cvtColor(cv2.imread(start_img_path), cv2.COLOR_BGR2RGB)
finish_img = cv2.cvtColor(cv2.imread(finish_img_path), cv2.COLOR_BGR2RGB)

start_depth = cv2.imread(start_depth_path, 0)
finish_depth = cv2.imread(finish_depth_path, 0)


fig, ax = plt.subplots(2, 1)
ax[0].imshow(start_img)
ax[1].imshow(finish_img)
plt.show()

# detect feature points
det = cv2.ORB_create(500)
kps1, descs1 = det.detectAndCompute(start_img, None)
kps2, descs2 = det.detectAndCompute(finish_img, None)

# matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

tentatives = bf.match(descs1, descs2)

th = 4.0
max_th = 25.0
t=time()
cv2_H, cv2_mask = verify_cv2(kps1,kps2,tentatives, th)
print (time()-t, ' sec cv2')
t=time()
mag_H, mag_mask = verify_magsac(kps1,kps2,tentatives, max_th)
print (time()-t, ' sec magsac')
t=time()
magpp_H, magpp_mask = verify_magsac_plus_plus(kps1,kps2,tentatives, max_th)
print (time()-t, ' sec magsac++')

draw_matches(kps1, kps2, tentatives, start_img, finish_img, cv2_H, cv2_mask )
draw_matches(kps1, kps2, tentatives, start_img, finish_img, mag_H, mag_mask)
draw_matches(kps1, kps2, tentatives, start_img, finish_img, magpp_H, magpp_mask)


width = 640
height = 480
fx = 381.694
fy = 381.694
cx = 323.56
cy = 237.11
        
# camera intrinsic parameters        
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0,  1]])

M_est = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]])

targetP = np.dot(K, M_est)

rotation_2d = magpp_H

# start = 11
# finish = 661
# diff_angle = 360 / (finish - start)
# angle = diff_angle * (101 - 11)

angle = 2

axis_angles = [0, angle, 0]
# axis_angles = [0, 0, angle]
R = create_rotation_matrix(axis_angles)
rotation_3d = np.zeros((3, 4))
rotation_3d[:, :3] = R


ansP = np.linalg.inv(rotation_2d).dot(K).dot(rotation_3d)

print('target:', target)
print('ans:', ans)
print()
print('rotation:', rotation_3d)


