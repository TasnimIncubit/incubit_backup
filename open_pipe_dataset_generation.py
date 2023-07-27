import sys
import numpy as np
import open3d 
import open3d.ml.torch as ml3d
import random
import os 

# create folders for dataset
pipe_folder = '/home/tasnim/Open3D-ML/pipe_dataset/pipe_only'
if not os.path.exists(pipe_folder):
    os.makedirs(pipe_folder)
axis_folder = '/home/tasnim/Open3D-ML/pipe_dataset/axis_only'
if not os.path.exists(axis_folder):
    os.makedirs(axis_folder)
pipe_axis_combined_folder = '/home/tasnim/Open3D-ML/pipe_dataset/pipe_axis_combined'
if not os.path.exists(pipe_axis_combined_folder):
    os.makedirs(pipe_axis_combined_folder)
    


#total number of sample straight pipes
dataset_length = 1000

for count in range(0, dataset_length, 1):

    random_radious = random.uniform(1, 10)
    random_height = random.uniform(random_radious*5, random_radious*20)
    h = random_height
    random_resolution = random.randint(int(random_radious)*10, int(random_radious)*50)
    random_split = random.randint(int(random_height), int(random_height)*10)
    #random_number_of_points = random.randint(int(random_height)*3, int(random_height)*5) * random.randint(int(random_height)*3, int(random_height)*5)
    random_number_of_points = random.randint(15000, 50000)
    # aug is causing trouble
    tx = random.uniform(1, 10)
    ty = random.uniform(1, 10)
    tz = random.uniform(1, 10)

    rx = random.uniform(1, 10)
    ry = random.uniform(1, 10)
    rz = random.uniform(1, 10)

    # pipe generation

    pipe1 = open3d.geometry.TriangleMesh.create_cylinder(radius=random_radious, height=random_height, resolution=random_resolution, split=random_split, create_uv_map=False) # generate pipe mesh
    
    # dont aug before cropping the ends

    #pipe1.translate((tx, ty, tz)) # augmentation - translation
    #pipe1.rotate(pipe1.get_rotation_matrix_from_xyz((rx, ry, rz)),center= pipe1.get_center()) # aumentation - rotation


    pipe_pcd1 = pipe1.sample_points_uniformly(number_of_points=random_number_of_points) # sample points uniformly
    #print(np.shape(np.asarray(pipe_pcd1.points)))
    pipe_xyz1 = np.asarray(pipe_pcd1.points) # convert to np array
    number_of_rows = pipe_xyz1.shape[0] 
    random_indices = np.random.choice(number_of_rows, 
                                    size=int(random.randint(500, 10000)), 
                                    replace=False)
    pipe_xyz_before = pipe_xyz1[random_indices, :] # randomizing subsampling in point cloud to make it non-uniform

    

    points_length = len(pipe_xyz_before)
    indices_to_be_removed = []
    c = 0

    list_pipe_xyz = []

    for i in range(0, points_length, 1):
        if pipe_xyz_before[i][2] > ((h/2)-1) or pipe_xyz_before[i][2] < -((h/2)-1):
            #print(pipe_xyz_before[i])
            indices_to_be_removed.append(i)
            c += 1
        else:
            list_pipe_xyz.append(pipe_xyz_before[i])
            
    pipe_xyz = np.array(list_pipe_xyz)

   
    #print('before shape: ', np.shape(pipe_xyz_before))
    #print('need to cut :', c)
    #print('remove indices: ', indices_to_be_removed)
    #print('after shape: ', np.shape(pipe_xyz))

    pipe_pcd = open3d.geometry.PointCloud() # create a pointcloud
    pipe_pcd.points = open3d.utility.Vector3dVector(pipe_xyz) # convert np array to point cloud

    # try
    pipe_pcd.translate((tx, ty, tz)) # augmentation - translation
    pipe_pcd.rotate(pipe_pcd.get_rotation_matrix_from_xyz((rx, ry, rz)),center= pipe_pcd.get_center()) # aumentation - rotation


    # save pipe pointcloud
    current_pipe_filename = 'pipe_' + format(count, '05d') + '.ply'
    pipe_save_filename = os.path.join(pipe_folder,current_pipe_filename)
    #print(pipe_save_filename)
    open3d.io.write_point_cloud(pipe_save_filename, pipe_pcd)


    # pipe axis
    axis1 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.0001, height=random_height, resolution=random_resolution, split=random_split, create_uv_map=False) # generating axis mesh
    axis1.translate((tx, ty, tz)) # augmentation - translation 
    axis1.rotate(axis1.get_rotation_matrix_from_xyz((rx, ry, rz)),
                center= axis1.get_center()) # augmentation - rotation
    axis_pcd1 = axis1.sample_points_uniformly(number_of_points=random_number_of_points) # uniform sampling of mash to point cloud
    # print(np.shape(np.asarray(axis_pcd1.points)))
    axis_xyz1 = np.asarray(axis_pcd1.points) # np array of point cloud
    number_of_rows = axis_xyz1.shape[0]
    random_indices = np.random.choice(number_of_rows, 
                                    size=int(random_number_of_points/10), 
                                    replace=False) # random sampling of point clouds
    
    axis_xyz = axis_xyz1[random_indices, :] # random sampling of point clouds
    #print("random sampled ", np.shape(axis_xyz))
    axis_pcd = open3d.geometry.PointCloud() # create axis point cloud 
    axis_pcd.points = open3d.utility.Vector3dVector(axis_xyz) # convert np array to point cloud
    # savee point cloud as ply
    current_axis_filename = 'axis_' + format(count, '05d') + '.ply'
    axis_save_filename = os.path.join(axis_folder,current_axis_filename)
    #print(axis_save_filename)
    open3d.io.write_point_cloud(axis_save_filename, axis_pcd)


    # pipe and axis combined

    pipe_and_axis_pcd = pipe_pcd + axis_pcd # combine pipe and axis
    #print(np.shape(np.asarray(pipe_and_axis_pcd.points)))
    # save pipe and axis together 
    current_pipe_and_axis_filename = 'pipe_and_axis_' + format(count, '05d') + '.ply'
    pipe_and_axis_save_filename = os.path.join(pipe_axis_combined_folder,current_pipe_and_axis_filename)
    #print(pipe_and_axis_save_filename)
    open3d.io.write_point_cloud(pipe_and_axis_save_filename, pipe_and_axis_pcd)

    print('dataset generation progress ', count*100/dataset_length, '%', end = '\r')




#pipe_pcd_loaded = open3d.io.read_point_cloud(pipe_save_filename) # not working
#print(np.shape(np.asarray(pipe_pcd_loaded.points)))


