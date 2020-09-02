import cv2
import glob
import numpy as np


# This code is unusable!!
fnum = 9
# dataset_dir = '../../rgbd_dataset/LIDAR/Nami1/'
# left_dir = dataset_dir + 'image_2/'
# right_dir = dataset_dir + 'image_3/'
# imgL = cv2.imread(left_dir + '{}.png'.format(fnum))
# imgR = cv2.imread(right_dir + '{}.png'.format(fnum))


dataset_dir = '../../rgbd_dataset/SHAPE/Nami1/'
left_dir = dataset_dir + 'detect_rgb/'
right_dir = left_dir


imgL = cv2.imread(left_dir + '{}.png'.format(fnum))
imgR = cv2.imread(right_dir + '{}.png'.format(fnum + 1))

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

dist_coef = np.array([0, 0, 0, 0, 0])

# 2d points
imgL = cv2.undistort(imgL, K, dist_coef)
imgR = cv2.undistort(imgR, K, dist_coef)

# ORB detector
detector = cv2.ORB_create()

# matcher
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

# detect
kp1, des1 = detector.detectAndCompute(imgL, None)
kp2, des2 = detector.detectAndCompute(imgR, None)

matches = matcher.match(des1, des2)

good = []
pts1 = []  # left
pts2 = []  # right

matches = sorted(matches, key=lambda x: x.distance)

count = 0
for m in matches:
    count += 1
    if count <= 60:
        good.append([m])
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        
        
pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
# find fundamental matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# normalize points
pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K, distCoeffs=None)
pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K, distCoeffs=None)

# find essential matrix
E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=3.0)

points, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)

print(R)
print(t)





