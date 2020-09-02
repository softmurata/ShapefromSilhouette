#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import scipy.io
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import open3d as o3d
import mcubes

# Load camera matrices
data = scipy.io.loadmat("data/dino_Ps.mat")
data = data["P"]
projections = [data[0, i] for i in range(data.shape[1])]

# projection matrix = K * [R|t]  #(3, 4) shape
# K = (3, 3) shape
# proj mat = K.T @ [R|t] or np.dot(K.T, [R|t])
"""
# load images
files = sorted(glob.glob("data/*.ppm"))
images = []
for f in files:
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(float)
    im /= 255
    images.append(im[:, :, ::-1])
    
print('image shape:', images[0].shape)    

# get silouhette from images
imgH, imgW, __ = images[0].shape
silhouette = []
for im in images:
    temp = np.abs(im - [0.0, 0.0, 0.75])
    temp = np.sum(temp, axis=2)
    y, x = np.where(temp <= 1.1)
    im[y, x, :] = [0.0, 0.0, 0.0]
    im = im[:, :, 0]
    im[im > 0] = 1.0
    im = im.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    # plt.imshow(im)
    # plt.show()
    silhouette.append(im)

#    plt.figure()
#    plt.imshow(im)
"""

silhouette = []
sil_dir = './sil_images/'
camera_dir = './camera/'
sil_image_files = os.listdir(sil_dir)
sil_image_numbers = sorted([int(a.split('.')[0]) for a in sil_image_files])

coef = 15
sil_image_files = [sil_dir + str(a) + '.png' for a in sil_image_numbers]
cam_intr_files = [camera_dir + 'intrinsic' + str(a * 15) + '.npy' for a in sil_image_numbers]
cam_extr_files = [camera_dir + 'extrinsic' + str(a * 15) + '.npy' for a in sil_image_numbers]

# div = 10
# sil_image_files = [sil_dir + str(a) + '.png' for a in sil_image_numbers if a % div == 0]
# cam_intr_files = [camera_dir + 'intrinsic' + str(a) + '.npy' for a in sil_image_numbers if a % div == 0]
# cam_extr_files = [camera_dir + 'extrinsic' + str(a) + '.npy' for a in sil_image_numbers if a % div == 0]

for sil_image_path in sil_image_files:
    sil = cv2.imread(sil_image_path, 0) / 255.0
    sil = cv2.resize(sil, (512, 512))
    silhouette.append(sil)
    
imgH, imgW = silhouette[0].shape


#%%
# create voxel grid
s = 150
x, y, z = np.mgrid[:s, :s, :s]
pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float)
pts = pts.T
print('pts shape:', pts.shape)
nb_points_init = pts.shape[0]
xmax, ymax, zmax = np.max(pts, axis=0)
pts[:, 0] /= xmax
pts[:, 1] /= ymax
pts[:, 2] /= zmax
center = pts.mean(axis=0)
pts -= center
# Hyparameters (?) => yes hyparameters
ortho_ratio = 0.3
pts /= ortho_ratio
# pts[:, 2] -= 0.1

pts = np.vstack((pts.T, np.ones((1, nb_points_init))))

filled = []
count = 0
for idx, (Pt, im) in enumerate(zip(projections, silhouette)):
    intrinsic = np.load(cam_intr_files[idx])
    extrinsic = np.load(cam_extr_files[idx])[:3, :]
    P = np.matmul(intrinsic, extrinsic)
    print()
    print('---projection matrix---')
    print(P.shape, Pt.shape, pts.shape)
    uvs = P @ pts
    uvs /= uvs[2, :]
    uvs = np.round(uvs).astype(int)
    x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < imgW)
    y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < imgH)
    good = np.logical_and(x_good, y_good)
    indices = np.where(good)[0]
    fill = np.zeros(uvs.shape[1])
    sub_uvs = uvs[:2, indices]
    res = im[sub_uvs[1, :], sub_uvs[0, :]]
    fill[indices] = res
    print() 
    
    filled.append(fill)
    count += 1
    
filled = np.vstack(filled)

# the occupancy is computed as the number of camera in which the point "seems" not empty
occupancy = np.sum(filled, axis=0)
print('occupancy:', np.max(occupancy))

# Select occupied voxels
pts = pts.T
good_points = pts[occupancy > 0, :]

occ = occupancy.reshape((s, s, s))
vertices, triangles = mcubes.marching_cubes(occ, 14.8)

# create mesh
mesh = o3d.geometry.TriangleMesh()
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh('result.ply', mesh)
print('mesh save ok')

"""
# coloring
# mesh to point cloud
number_of_points = 5000
# normalization
scale = np.max(mesh.get_max_bound() - mesh.get_min_bound())
mesh.scale(1 / scale, center=mesh.get_center())

pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)


points = np.asarray(pcd.points).T
print(np.min(points), np.max(points), points.mean(0))
points = np.vstack((points, np.ones((1, number_of_points))))
print(points.shape)


intrinsic = np.load(cam_intr_files[0])
extrinsic = np.load(cam_extr_files[0])[:3, :]
P = np.matmul(intrinsic, extrinsic)
print()
print('---projection matrix---')
print(P.shape)
uvs = P @ points
uvs /= uvs[2, :]
uvs = np.round(uvs).astype(int)
print(uvs[:, :])
print(uvs.shape)

sil_image_path = './sil_images/0.png'
rgb = cv2.imread(sil_image_path) / 255.0
rgb = cv2.resize(rgb, (512, 512))

colors = rgb.transpose(2, 0, 1)
colors = colors.reshape((3, -1))
target_colors = []
target_points = []
for idx in range(uvs.shape[1]):
    uv_coord = uvs[:, idx]
    if 0 <= uv_coord[0] <= 512 and 0 <= uv_coord[1] <= 512:
        c = colors[:, uv_coord[0], uv_coord[1]]
        p = np.linalg.inv(P) @ uv_coord
        target_colors.append(c)
        target_points.append(p)
        
print(np.array(target_points).shape)
"""


"""    
#%% save point cloud with occupancy scalar 
filename = "shape.txt"
with open(filename, "w") as fout:
    fout.write("x,y,z,occ\n")
    for occ, p in zip(occupancy, pts[:, :3]):
        fout.write(",".join(p.astype(str)) + "," + str(occ) + "\n")


# x = pts[::s * s, 0]
# y = pts[:s*s:s, 1]
# z = pts[:s, 2]
"""





""" 
#%% save as rectilinear grid (this enables paraview to display its iso-volume as a mesh)
import vtk

xCoords = vtk.vtkFloatArray()
x = pts[::s*s, 0]
y = pts[:s*s:s, 1]
z = pts[:s, 2]
for i in x:
    xCoords.InsertNextValue(i)
yCoords = vtk.vtkFloatArray()
for i in y:
    yCoords.InsertNextValue(i)
zCoords = vtk.vtkFloatArray()
for i in z:
    zCoords.InsertNextValue(i)
values = vtk.vtkFloatArray()
for i in occupancy:
    values.InsertNextValue(i)
rgrid = vtk.vtkRectilinearGrid()
rgrid.SetDimensions(len(x), len(y), len(z))
rgrid.SetXCoordinates(xCoords)
rgrid.SetYCoordinates(yCoords)
rgrid.SetZCoordinates(zCoords)
rgrid.GetPointData().SetScalars(values)

writer = vtk.vtkXMLRectilinearGridWriter()
writer.SetFileName("shape.vtr")
writer.SetInputData(rgrid)
writer.Write()
"""
