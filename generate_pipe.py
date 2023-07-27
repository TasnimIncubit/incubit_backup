import sys
import numpy as np
import open3d 
import open3d.ml.torch as ml3d
import random
import os 

print('start')

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
                                    size=int(random_number_of_points/5), 
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
current_pipe_filename = 'pipe.ply'
pipe_save_filename = os.path.join(pipe_folder,current_pipe_filename)
print(pipe_save_filename)
open3d.io.write_point_cloud(pipe_save_filename, pipe_pcd)
