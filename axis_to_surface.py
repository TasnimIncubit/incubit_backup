
import sys
import numpy as np
import open3d 
import open3d.ml.torch as ml3d
import random
import os  
import torch
import math as m

dataset_folder = '/home/tasnim/Open3D-ML/try_pipe_dataset'
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
only_pipe_instance = 'pipe.ply'
pipe_and_axis_instance = 'pipe_and_axis.ply'
only_axis_instance = 'axis.ply'


def create_cylinder(radius, height, num_points):
    # Generate equally spaced angles
    angles = np.linspace(0, 2*np.pi, num_points)

    # Compute x and y coordinates of the points on the circular base
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Compute z coordinates of the points along the height of the cylinder
    z = np.linspace(0, height, num_points)

    # Combine the x, y, and z coordinates to create the point cloud
    surface_and_corresponding_axis_points_list = []
    for i in range(0, len(z), 1):
        for j in range(0,len(x),1):
            surface_and_corresponding_axis_points_list.append([[x[j],y[j],z[i]],[0, 0, z[i]]])
    surface_and_corresponding_axis_points = np.array(surface_and_corresponding_axis_points_list)

    return surface_and_corresponding_axis_points

def random_sampling_points(np_all_points):

    length, no_content, no_coordinate = np.shape(np_all_points)
    number_of_rows = length
    random_indices = np.random.choice(number_of_rows, 
                                    size=int(random.randint(500, int(length/2))), 
                                    replace=False)
    pipe_xyz= np_all_points[random_indices, :, :] # randomizing subsampling in point cloud to make it non-uniform
    print('random -> ', len(random_indices), np.shape(pipe_xyz))     

    return pipe_xyz
   

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])








# Example usage
radius = 1.0
height = 5.0
num_points = 100


tx = random.uniform(1, 10)
ty = random.uniform(1, 10)
tz = random.uniform(1, 10)

rx = random.uniform(1, 10)
ry = random.uniform(1, 10)
rz = random.uniform(1, 10)


# get cylinder surface, axis, and surface_and_axis
# cylinder_surface_and_corresponding_axis_points = create_cylinder(radius, height, num_points)
uniform_cylinder_surface_and_corresponding_axis_points = create_cylinder(radius, height, num_points)
cylinder_surface_and_corresponding_axis_points = random_sampling_points(uniform_cylinder_surface_and_corresponding_axis_points)

R = Rz(rx) * Ry(ry) * Rz(rz)
cylinder_surface_and_corresponding_axis_points[:,0] = np.matmul(cylinder_surface_and_corresponding_axis_points[:,0], R)
cylinder_surface_and_corresponding_axis_points[:,1] = np.matmul(cylinder_surface_and_corresponding_axis_points[:,1], R)


#save only pipe point cloud
pipe_pcd = open3d.geometry.PointCloud() # create a pointcloud
pipe_pcd.points = open3d.utility.Vector3dVector(cylinder_surface_and_corresponding_axis_points[:,0]) # convert np array to point cloud
#pipe_pcd.translate((tx, ty, tz)) # augmentation - translation
#pipe_pcd.rotate(pipe_pcd.get_rotation_matrix_from_xyz((rx, ry, rz)),center= pipe_pcd.get_center()) # aumentation - rotation
open3d.io.write_point_cloud(os.path.join(dataset_folder, only_pipe_instance), pipe_pcd)

# save only axis point cloud
axis_pcd = open3d.geometry.PointCloud() # create a pointcloud
axis_pcd.points = open3d.utility.Vector3dVector(cylinder_surface_and_corresponding_axis_points[:,1] ) # convert np array to point cloud
#axis_pcd.translate((tx, ty, tz)) # augmentation - translation
#axis_pcd.rotate(axis_pcd.get_rotation_matrix_from_xyz((rx, ry, rz)),center= axis_pcd.get_center()) # aumentation - rotation
open3d.io.write_point_cloud(os.path.join(dataset_folder, only_axis_instance), axis_pcd)


# save pipe and axis point cloud
pipe_and_axis_pcd = open3d.geometry.PointCloud() # create a pointcloud
pipe_and_axis_pcd.points = open3d.utility.Vector3dVector(np.concatenate((cylinder_surface_and_corresponding_axis_points[:,0],cylinder_surface_and_corresponding_axis_points[:,1]), axis = 0)) # convert np array to point cloud
#pipe_and_axis_pcd.translate((tx, ty, tz)) # augmentation - translation
#pipe_and_axis_pcd.rotate(pipe_and_axis_pcd.get_rotation_matrix_from_xyz((rx, ry, rz)),center= pipe_and_axis_pcd.get_center()) # aumentation - rotation
open3d.io.write_point_cloud(os.path.join(dataset_folder, pipe_and_axis_instance), pipe_and_axis_pcd)

# save axis point corresponding to each surface point

