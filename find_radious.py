import sys
import numpy as np
import open3d 
import open3d.ml.torch as ml3d
import random
import os  

# i = 0.1 directing along reverse normals to avoind conflict with other pipes
# find new iterpolated line in 3d
# find d
# minimize d 

pipe_instance = 'pipe.ply'
pipe_base = '/home/tasnim/Open3D-ML/pipe_dataset'
pipe_load_directory = os.path.join(pipe_base, pipe_instance)

point_cloud = open3d.io.read_point_cloud(pipe_load_directory) 
print('loaded pipe point cloud shape: ', np.shape(np.asarray(point_cloud.points)))

surface_points = np.asarray(point_cloud.points)

# find normals
point_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=300))
point_cloud.orient_normals_consistent_tangent_plane(k=15)

# flip normals
normals = np.asarray(point_cloud.normals)
flipped_normals = - normals
point_cloud.normals = open3d.utility.Vector3dVector(flipped_normals)


# find incremental points
r = 0.1
points_in_axis_direction = surface_points + r * flipped_normals 

# interpolate a line from points in axis direction
