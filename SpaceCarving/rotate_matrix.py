import numpy as np
import glob

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


dataset_dir = '../../rgbd_dataset/SHAPE/'
dataset_name = 'Nami1'
dataset_dir = dataset_dir + dataset_name + '/'

calib_dir = dataset_dir + 'calib/'
calib_files = sorted(glob.glob(calib_dir + '*.npy'))
numbers = [int(c.split('/')[-1].split('.')[0]) for c in calib_files]
sorted_index = np.argsort(numbers)

numbers = [numbers[idx] for idx in sorted_index]

calib_files = [calib_dir + str(n) + '.npy' for n in numbers]

# target Rotation matrix
start = 11
finish = 661

diff_angle = 360 / (finish - start)
angle = 0

angles = []

for i in range(len(numbers)):
    angles.append([i, angle])
    angle += diff_angle


R = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
Rs_tar = [R]
for ang in angles:
    target_num, anglez = ang
    axis_angles = [0, 0, anglez]
    Rc = create_rotation_matrix(axis_angles)
    Rc = change_rotation_mat_for_open3d(Rc)
    Rt = np.dot(R, Rc)
    # print('Rt:', Rt)
    Rs_tar.append(Rt)

# create rotation matrix
Rs = []

R = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

for idx, calib_path in enumerate(calib_files):
    R_now = np.load(calib_path)
    R_real = np.dot(R_now, Rs_tar[idx])
    Rs.append(R_real)
    print()
    print('R real:', R_real)
    print('Rs tar:', Rs_tar[idx])
    print()

# Rs = Rs[::-1]

