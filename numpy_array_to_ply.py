import sys
import numpy as np
import open3d 
import open3d.ml.torch as ml3d
import random
import os  
import torch
import math as m

# only using mean and std
# mean = np.array([[0.44704652,  0.4966047,  -2.87343395]])
# std = np.array([[16.69963795, 17.9724519,  23.06550812]])

# new not working values only surface and distance
# max_val = np.array([[157.44165455, 174.82071387, 179.34331285]])
# min_val = np.array([[-160.78322951, -189.21882333, -167.02990417]])
# max_min_diff = np.array([[318.22488406, 364.03953721, 346.37321702]])
# mean = np.array([[0.50665515, 0.52113963, 0.47392946]]) 
# std = np.array([[0.05247747, 0.04936951, 0.06659149]])

# old working values of surface and center
max_val = np.array([[157.44165455, 174.82071387, 179.34331285]])
min_val = np.array([[-160.78322951, -189.21882333, -167.02990417]])
max_min_diff = np.array([[318.22488406, 364.03953721, 346.37321702]])
mean = np.array([[0.50736506, 0.52182236, 0.46978139]]) 
std = np.array([[0.06373268, 0.06003646, 0.08100745]])

source_folder = '/home/tasnim/pipe_center_prediction/dgcnn.pytorch/dgcnn_test_pipe_predictions'
temp_load_path = '/home/tasnim/pipe_center_prediction/dgcnn.pytorch/all_radius_train_test_dgcnn_test_pipe_predictions/output_00008_radius_1.575439.npy'
# temp_load_path = '/home/tasnim/pipe_center_prediction/dgcnn.pytorch/dgcnn_test_pipe_predictions/dgcnn_numpy_pipe_and_axis_00008.npy'
# temp_load_path = '/home/tasnim/pipe_center_prediction/dgcnn.pytorch/all_center_mean_std_min_max_predictions/dgcnn_numpy_pipe_and_axis_00005.npy'
# temp_save_path = '/home/tasnim/Open3D-ML/normalized_np_to_reconstruct_ply'
temp_save_path = '/home/tasnim/pipe_center_prediction/dgcnn.pytorch/one_train_test_ply'


###################
# pipe radius
np_load_arr = np.load(temp_load_path) # (4096,2,3)
print('load shape: ', np.shape(np_load_arr))
surface = np_load_arr
surface = (surface*std) + mean
surface = (surface*max_min_diff) + min_val

#### output radius
base=os.path.basename(temp_load_path)
str_radius = os.path.splitext(base)[0][20:]
float_radius = float(str_radius)
print('float radius: ', float_radius)
pred_radius = float_radius

pc_surface = open3d.geometry.PointCloud() # create a pointcloud
pc_surface.points = open3d.utility.Vector3dVector(surface) # convert np array to point cloud



pipe_surface = pc_surface
pipe_normal = pc_surface
pipe_reverse_normal = pc_surface

# surface points
pipe_surface_points = np.asarray(pipe_surface.points)
# print('loaded pipe surface point cloud shape: ',np.shape(np.asarray(pipe_surface.points)))

# find normals
pipe_normal.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=pred_radius, max_nn=300))
pipe_normal.orient_normals_consistent_tangent_plane(k=15)
# open3d.io.write_point_cloud(pipe_normal_directory, pipe_normal)

# flip normals
normals = np.asarray(pipe_reverse_normal.normals)
flipped_normals = - normals
pipe_reverse_normal.normals = open3d.utility.Vector3dVector(flipped_normals)
# open3d.io.write_point_cloud(pipe_flipped_normal_directory, pipe_reverse_normal)

# find points at r depth inside
r = pred_radius

center = pipe_surface_points + r * flipped_normals






#################














# old working
# np_load_arr = np.load(temp_load_path) # (4096,2,3)
# surface = np_load_arr[:,0] # (4096,3)
# center = np_load_arr[:,1] # (4096,3)



# surface = np.expand_dims(surface, axis=0) # (1,4096,3)
# center = np.expand_dims(center, axis=0) # (1,4096,3)


# delta - only mean std  and 
# surface = (surface*std) + mean
# center = (center*std) + mean
# center = surface - center


# delta -  mean std  and min max
# surface = (surface*std) + mean
# surface = (surface*max_min_diff) + min_val
# center = (center*std) + mean
# center = (center*max_min_diff) + min_val
# center = surface - center


# center - only mean std  and 
# surface = (surface*std) + mean
# center = (center*std) + mean


# center-  mean std  and min max
# surface = (surface*std) + mean
# surface = (surface*max_min_diff) + min_val
# center = (center*std) + mean
# center = (center*max_min_diff) + min_val





# forward


# surface = (surface - min_val)/max_min_diff # (1,4096,3)
# surface = (surface - mean)/std  # (1,4096,3)

# center = (center - min_val)/max_min_diff # (1,4096,3) 
# center = (center - mean)/std # (1,4096,3)


# backward 
# center_pts = surface + center
# center_pts = (center_pts*std) + mean
# center_pts = (center_pts*max_min_diff) + min_val

# surface = (surface*std) + mean
# surface = (surface*max_min_diff) + min_val

# center = (center*std) + mean
# center = (center*max_min_diff) + min_val




# # squeeze
# uncomment surface and center for direct center or delta but not radius
# surface = np.squeeze(surface, axis=0) # (4096,3)
# center_pts = np.squeeze(center_pts, axis=0) # (4096,3)
# center = np.squeeze(center, axis=0) # (4096,3)

# delta0 = np.sqrt( (surface[0][0]-center_pts[0][0])**2 + (surface[0][1]-center_pts[0][1])**2 + (surface[0][2]-center_pts[0][2])**2 )
# print(surface[0], center_pts[0],'delta 0: ', delta0)




pipe_and_axis_pcd = open3d.geometry.PointCloud() # create a pointcloud
pipe_and_axis_pcd.points = open3d.utility.Vector3dVector(np.concatenate((surface, center), axis = 0)) # convert np array to point cloud
# pipe_and_axis_pcd.points = open3d.utility.Vector3dVector(np.concatenate((surface, center), axis = 0)) # convert np array to point cloud
pipe_and_axis_instance = 'normalized_np_to_reconstruct_pipe_and_axis.ply'
open3d.io.write_point_cloud(os.path.join(temp_save_path, pipe_and_axis_instance), pipe_and_axis_pcd)


pipe_pcd = open3d.geometry.PointCloud() # create a pointcloud
pipe_pcd.points = open3d.utility.Vector3dVector(surface) # convert np array to point cloud
# pipe_and_axis_pcd.points = open3d.utility.Vector3dVector(np.concatenate((surface, center), axis = 0)) # convert np array to point cloud
pipe_instance = 'normalized_np_to_reconstruct_pipe.ply'
open3d.io.write_point_cloud(os.path.join(temp_save_path, pipe_instance), pipe_pcd)


axis_pcd = open3d.geometry.PointCloud() # create a pointcloud
axis_pcd.points = open3d.utility.Vector3dVector(center) # convert np array to point cloud
# pipe_and_axis_pcd.points = open3d.utility.Vector3dVector(np.concatenate((surface, center), axis = 0)) # convert np array to point cloud
axis_instance = 'normalized_np_to_reconstruct_axis.ply'
open3d.io.write_point_cloud(os.path.join(temp_save_path, axis_instance), axis_pcd)

# before normalization
# start
# print('before normalization')
# delta0 = np.sqrt( (surface[0][0]-center[0][0])**2 + (surface[0][1]-center[0][1])**2 + (surface[0][2]-center[0][2])**2 )
# print('delta 0: ', delta0)
# delta1 = np.sqrt( (surface[1][0]-center[1][0])**2 + (surface[1][1]-center[1][1])**2 + (surface[1][2]-center[1][2])**2 )
# print('delta 1: ', delta1)
# delta2 = np.sqrt( (surface[2][0]-center[2][0])**2 + (surface[2][1]-center[2][1])**2 + (surface[2][2]-center[2][2])**2 )
# print('delta 2: ', delta2)
# delta3 = np.sqrt( (surface[3][0]-center[3][0])**2 + (surface[3][1]-center[3][1])**2 + (surface[3][2]-center[3][2])**2 )
# print('delta 3: ', delta3)
# print(surface[0],center[0])
# print('simple delta', (surface[0]-center[0]))

# surface = np.expand_dims(surface, axis=0) # (1,4096,3)
# center = np.expand_dims(center, axis=0) # (1,4096,3)
# print(surface - center)
# # end



# surface = np.expand_dims(surface, axis=0) # (1,4096,3)
# center = np.expand_dims(center, axis=0) # (1,4096,3)

# forward
# surface = (surface - min_val)/max_min_diff # (1,4096,3)
# surface = (surface - mean)/std  # (1,4096,3)

# center = (center - min_val)/max_min_diff # (1,4096,3) 
# center = (center - mean)/std # (1,4096,3)



# new dist
# start
# surface = np.squeeze(surface, axis=0) # (4096,3)
# center = np.squeeze(center, axis=0) # (4096,3)
# print('after normalization')
# delta0 = np.sqrt( (surface[0][0]-center[0][0])**2 + (surface[0][1]-center[0][1])**2 + (surface[0][2]-center[0][2])**2 )
# print('delta 0: ', delta0)
# delta1 = np.sqrt( (surface[1][0]-center[1][0])**2 + (surface[1][1]-center[1][1])**2 + (surface[1][2]-center[1][2])**2 )
# print('delta 1: ', delta1)
# delta2 = np.sqrt( (surface[2][0]-center[2][0])**2 + (surface[2][1]-center[2][1])**2 + (surface[2][2]-center[2][2])**2 )
# print('delta 2: ', delta2)
# delta3 = np.sqrt( (surface[3][0]-center[3][0])**2 + (surface[3][1]-center[3][1])**2 + (surface[3][2]-center[3][2])**2 )
# print('delta 3: ', delta2)

#print(surface [1], center [1])
#dist= np.linalg.norm(surface - center)
#print(dist)
# print(np.max(dist, axis = 1))
# end




# backward 
# surface = (surface*std) + mean
# surface = (surface*max_min_diff) + min_val

# center = (center*std) + mean
# center = (center*max_min_diff) + min_val

# # squeeze
# surface = np.squeeze(surface, axis=0) # (4096,3)
# center = np.squeeze(center, axis=0) # (4096,3)

# pipe_and_axis_pcd = open3d.geometry.PointCloud() # create a pointcloud
# pipe_and_axis_pcd.points = open3d.utility.Vector3dVector(np.concatenate((surface, center), axis = 0)) # convert np array to point cloud
# pipe_and_axis_instance = 'normalized_np_to_reconstruct_pipe_and_axis.ply'
# open3d.io.write_point_cloud(os.path.join(temp_save_path, pipe_and_axis_instance), pipe_and_axis_pcd)


