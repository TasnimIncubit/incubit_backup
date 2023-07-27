import sys
import numpy as np
import open3d 
import open3d.ml.torch as ml3d
import random
import os  

# find radious 
#   - create a new pipe with known radious
# reverse the normals - done
# create points along the normals to the center at radious distance


pipe_folder = '/home/tasnim/Open3D-ML/pipe_dataset'

random_number_of_points = 10000
h = 20.0
pipe1 = open3d.geometry.TriangleMesh.create_cylinder(radius=3.0, height=h, resolution=20, split=20, create_uv_map=False) # generate pipe mesh
open3d.io.write_triangle_mesh('/home/tasnim/Open3D-ML/pipe_dataset/pipe_mesh.ply', pipe1, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
#cropping the mesh
# pipe1.triangles = open3d.utility.Vector3iVector(np.asarray(pipe1.triangles)[10:len(pipe1.triangles)-10, :])
# pipe1.triangle_normals = open3d.utility.Vector3dVector(np.asarray(pipe1.triangle_normals)[10:len(pipe1.triangle_normals)-10, :])

pipe_pcd1 = pipe1.sample_points_uniformly(number_of_points=random_number_of_points) # sample points uniformly
pipe_xyz1 = np.asarray(pipe_pcd1.points) # convert to np array
number_of_rows = pipe_xyz1.shape[0] 
random_indices = np.random.choice(number_of_rows, 
                                    size=int(random_number_of_points/2), 
                                    replace=False)
    

#pipe_xyz = pipe_xyz1[random_indices, :] 

pipe_xyz_before = pipe_xyz1[random_indices, :] # randomizing subsampling in point cloud to make it non-uniform
print('before shape: ', np.shape(pipe_xyz_before))

points_length = len(pipe_xyz_before)
indices_to_be_removed = []
c = 0

list_pipe_xyz = []

for i in range(0, points_length, 1):
    if pipe_xyz_before[i][2] > ((h/2)-1) or pipe_xyz_before[i][2] < -((h/2)-1):
        print(pipe_xyz_before[i])
        indices_to_be_removed.append(i)
        c += 1
    else:
        list_pipe_xyz.append(pipe_xyz_before[i])
        
pipe_xyz = np.array(list_pipe_xyz)

print('before shape: ', np.shape(pipe_xyz_before))
print('need to cut :', c)
#print('remove indices: ', indices_to_be_removed)
print('after shape: ', np.shape(pipe_xyz))



pipe_pcd = open3d.geometry.PointCloud() # create a pointcloud
pipe_pcd.points = open3d.utility.Vector3dVector(pipe_xyz) # convert np array to point cloud


    # save pipe pointcloud
pipe_instance = 'pipe.ply'
pipe_save_filename = os.path.join(pipe_folder,pipe_instance )
print(pipe_save_filename)
open3d.io.write_point_cloud(pipe_save_filename, pipe_pcd)



pipe_base = '/home/tasnim/Open3D-ML/pipe_dataset'
pipe_load_directory = os.path.join(pipe_base, pipe_instance)

original_pipe_save_directory = '/home/tasnim/Open3D-ML/pipe_dataset/original_pipe.ply'
pipe_normal_directory = '/home/tasnim/Open3D-ML/pipe_dataset/pipe_normal.ply'
pipe_flipped_normal_directory = '/home/tasnim/Open3D-ML/pipe_dataset/pipe_flipped_normal.ply'
pipe_estimated_axis_directory = '/home/tasnim/Open3D-ML/pipe_dataset/pipe_estimated_axis.ply'
pipe_and_estimated_axis_directory = '/home/tasnim/Open3D-ML/pipe_dataset/pipe_and_estimated_axis.ply'

pipe_pcd_loaded = open3d.io.read_point_cloud(pipe_load_directory ) 
open3d.io.write_point_cloud(original_pipe_save_directory, pipe_pcd_loaded)
print('loaded pipe point cloud shape: ', np.shape(np.asarray(pipe_pcd_loaded.points)))

pipe_surface = pipe_pcd_loaded
pipe_normal = pipe_pcd_loaded
pipe_reverse_normal = pipe_pcd_loaded

# surface points
pipe_surface_points = np.asarray(pipe_surface.points)
print('loaded pipe surface point cloud shape: ',np.shape(np.asarray(pipe_surface.points)))

# find normals
pipe_normal.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=300))
pipe_normal.orient_normals_consistent_tangent_plane(k=15)
open3d.io.write_point_cloud(pipe_normal_directory, pipe_normal)

# flip normals
normals = np.asarray(pipe_reverse_normal.normals)
flipped_normals = - normals
pipe_reverse_normal.normals = open3d.utility.Vector3dVector(flipped_normals)
open3d.io.write_point_cloud(pipe_flipped_normal_directory, pipe_reverse_normal)

# find points at r depth inside
r = 3

point_in_axis_direction = pipe_surface_points + r * flipped_normals 
print('estimated axis points shape: ', np.shape(point_in_axis_direction))
estimated_axis = open3d.geometry.PointCloud() 
estimated_axis.points = open3d.utility.Vector3dVector(point_in_axis_direction)
open3d.io.write_point_cloud(pipe_estimated_axis_directory, estimated_axis)

# pipe and axis combined
pipe_and_estimated_axis = pipe_pcd_loaded + estimated_axis
open3d.io.write_point_cloud(pipe_and_estimated_axis_directory, pipe_and_estimated_axis)



# estimate radious 
# _, inliers = point_cloud.segment_plane(distance_threshold=max_distance, ransac_n=3, num_iterations=1000)