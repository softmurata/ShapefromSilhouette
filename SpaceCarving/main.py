import scipy.io
import numpy as np
import cv2
import glob
import open3d as o3d
import mcubes
import matplotlib.pyplot as plt

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

def change_rotation_mat_for_open3d(R, translations):
    R_s = np.zeros((4, 4))
    R_s[:3, :3] = R
    R_s[3, :3] = np.array(translations).T
    R_s[3, 3] = 1.0
    
    return R_s


dataset_dir = '../../rgbd_dataset/SHAPE/'
dataset_name = 'Nami1'
dataset_dir = dataset_dir + dataset_name + '/'

silhouette_dir = '../../rgbd_dataset/MASK/{}/rgb/'.format(dataset_name)

sil_files = glob.glob(silhouette_dir + '*.png')
calib_files = glob.glob(dataset_dir + 'calib/*.npy')
numbers = [int(c.split('/')[-1].split('.')[0]) for c in calib_files]
sorted_index = np.argsort(numbers)

numbers = [numbers[idx] for idx in sorted_index]
sil_files = [silhouette_dir + str(n) + '.png' for n in numbers]
calib_files = [dataset_dir + 'calib/' + str(n) + '.npy' for n in numbers]

silhouette = []

new_numbers = []

for num, sil_path in zip(numbers, sil_files):
    sil_img = cv2.imread(sil_path, 0)
    if sil_img is not None:
        sil_img = sil_img / 255
        # plt.imshow(sil_img)
        # plt.show()
        silhouette.append(sil_img)
        new_numbers.append(num)
    
H, W = silhouette[0].shape
 
# create voxel grid
s = 120
x, y, z = np.mgrid[:s, :s, :s]
pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float)
pts = pts.T

nb_points_init = pts.shape[0]
xmax, ymax, zmax = np.max(pts, axis=0)

pts[:, 0] /= xmax
pts[:, 1] /= ymax
pts[:, 2] /= zmax

center = pts.mean(axis=0)
pts -= center


# Hyparameters(?)
# pts /= 5
# pts[:, 2] -=0.65

pts = np.vstack((pts.T, np.ones((1, nb_points_init))))


# create rotation matrix

# Target
start = 11
finish = 661

diff_angle = 360 / (finish - start)
angle = 0

angles = []

for i in range(len(new_numbers)):
    angles.append([i, angle])
    angle += diff_angle


R = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
# R = np.load(calib_files[0])
Rs = []
for ang in angles:
    target_num, anglez = ang
    axis_angles = [0, 0, anglez]
    # axis_angles = [anglez, 0, 0]
    # axis_angles = [0, anglez, 0]
    Rc = create_rotation_matrix(axis_angles)
    translations = [0, 0, 0]
    Rc = change_rotation_mat_for_open3d(Rc, translations)
    # fixed version
    R_real = np.dot(R, Rc)
    translations_v = [0, 0, 0]
    Rcv = change_rotation_mat_for_open3d(R[:3, :3], translations_v)
    R_real = np.dot(R_real, Rcv)
    Rs.append(R_real)

print('finish rotation matrix')

# Rs = Rs[::-1]

indicies = [2 + 18 * i for i in range(36)]

Rs = [Rs[idx] for idx in indicies]
silhouette = [silhouette[idx] for idx in indicies]


filled = []

data = scipy.io.loadmat("data/dino_Ps.mat")
data = data["P"]
projections = [data[0, i] for i in range(data.shape[1])]

K_dino = np.array([[3310, 0.0, 316.73],
                   [0.0, 3325.5, 200.55],
                   [0, 0, 1]])

for rotation_mat, P, sil in zip(Rs, projections, silhouette):
    rotation_mat = rotation_mat[:3, :]
    # proj_mat = np.dot(K, R[:3, :])
    print()
    print('--projection matrix---')
    print(P)
    print()
    print(np.linalg.inv(K_dino).dot(P))
    rotation_mat = np.linalg.inv(K_dino).dot(P)
    proj_mat = np.dot(K, rotation_mat)
    print(proj_mat)
    uvs = proj_mat @ pts
    # uvs = P @ pts
    uvs /= uvs[2, :]  # normalization?
    uvs = np.round(uvs).astype(int)
    
    x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < W)
    y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < H)
    
    good = np.logical_and(x_good, y_good)
    indices = np.where(good)[0]
    fill = np.zeros(uvs.shape[1])
    sub_uvs = uvs[:2, indices]
    # print(sub_uvs)
    # print(sil)
    print()
    print()
    res = sil[sub_uvs[1, :], sub_uvs[0, :]]
    fill[indices] = res
    
    filled.append(fill)
    
filled = np.vstack(filled)

print('finish filled occupancy')
# occupancy
occupancy = np.sum(filled, axis=0)
print('max:', np.max(occupancy))

# save mesh
occ = occupancy.reshape((s, s, s))
threshold = 30.0
vertices, triangles = mcubes.marching_cubes(occ, threshold)  # 30 => occupancy_threshold

mesh = o3d.geometry.TriangleMesh()
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh('nami_result.ply', mesh)

print('mesh save ok')







