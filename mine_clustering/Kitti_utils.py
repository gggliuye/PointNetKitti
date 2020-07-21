import numpy as np
import glob
import random
import os

import open3d as o3d
from pyntcloud import PyntCloud

import mine_clustering.clustering_random_sample as mine_seg

def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def PCA(data):
    # normalize
    normalized_data = data - np.mean(data ,0)
    # SVD
    U,sigma,VT = np.linalg.svd(normalized_data)
    eigenvalues = sigma
    eigenvectors = np.transpose(VT)
    return eigenvalues, eigenvectors


def get_3d_box(ry, t1, t2, t3, w, h, l, color_y = 0):
    # print(ry, t1, t2, t3, w, h, l)
    ry_rod = ry
    points_vis_axis = np.array([[l/2, w/2, 0],[l/2,-w/2, 0],[-l/2, w/2, 0], [-l/2, -w/2, 0], [l/2, w/2, h],[l/2,-w/2, h],[-l/2, w/2, h], [-l/2, -w/2, h]])
    rotation_matrix = np.array([[np.cos(ry_rod), np.sin(ry_rod), 0], [-np.sin(ry_rod), np.cos(ry_rod), 0], [0,0,1]])
    rotated_cloud = np.dot(rotation_matrix, (points_vis_axis.transpose())).transpose()
    points_vis_axis = rotated_cloud + np.array([t1, t3, -t2])

    # need to rotate another 90 degrees -> coordinate change
    ry_rod = np.pi/2
    rotation_matrix = np.array([[np.cos(ry_rod), np.sin(ry_rod), 0], [-np.sin(ry_rod), np.cos(ry_rod), 0], [0,0,1]])
    points_vis_axis = np.dot(rotation_matrix, (points_vis_axis.transpose())).transpose()

    lines_vis_axis = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0,4],[1,5],[2,6],[3,7]]

    colors_box = [[0, color_y, 1] for i in range(len(lines_vis_axis))]
    box_set = o3d.geometry.LineSet()
    box_set.lines = o3d.utility.Vector2iVector(lines_vis_axis)
    box_set.points  = o3d.utility.Vector3dVector(points_vis_axis.tolist())
    box_set.colors = o3d.utility.Vector3dVector(colors_box)

    return box_set


def rotate_and_threshold_pointcloud(pointcloud, ry, t1, t2, t3, w, h, l):
    pointcloud_tmp = pointcloud[:, 0:3]

    # need to rotate another 90 degrees -> coordinate change
    ry_rod = - np.pi/2
    rotation_matrix = np.array([[np.cos(ry_rod), np.sin(ry_rod), 0], [-np.sin(ry_rod), np.cos(ry_rod), 0], [0,0,1]])
    pointcloud_tmp = np.dot(rotation_matrix, (pointcloud_tmp.transpose())).transpose()

    ry_rod = -ry
    rotation_matrix = np.array([[np.cos(ry_rod), np.sin(ry_rod), 0], [-np.sin(ry_rod), np.cos(ry_rod), 0], [0,0,1]])
    # translate
    translated_cloud = pointcloud_tmp - np.array([t1,t3,-t2])
    # rotate
    rotated_cloud = np.dot(rotation_matrix, (translated_cloud.transpose())).transpose()
    # filter
    x_flag = np.logical_and([rotated_cloud[:,0] <= l/2], [rotated_cloud[:,0] >= -l/2])
    y_flag = np.logical_and([rotated_cloud[:,1] <= w/2], [rotated_cloud[:,1] >= -w/2])

    # cut more points from the round
    z_flag = np.logical_and([rotated_cloud[:,2] <= h], [rotated_cloud[:,2] >= 0.1])
    final_flag = np.logical_and(x_flag, y_flag)
    final_flag = np.logical_and(final_flag, z_flag)
    output_points = pointcloud[final_flag[0]]
    return output_points

def save_points_to_ply(cloud, file_name):
    file_out = open(file_name, "w")

    file_out.write('ply\nformat ascii 1.0\ncomment Created by LIUYE\n')
    str_size = 'element vertex ' + str(cloud.shape[0]) + '\n'
    file_out.write(str_size)
    file_out.write('property float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n')

    for i in range(cloud.shape[0]):
        str_pt = str(cloud[i,0]) + ' ' + str(cloud[i,1]) + ' ' + str(cloud[i,2]) + ' ' + str(cloud[i,3]) + '\n'
        file_out.write(str_pt)

    file_out.close()

# as from the ground truth, we know we only care about the obstracles in the front view
def take_the_cloud_in_front_view(input_cloud):
    front_pts_indices = []
    for i in range(input_cloud.shape[0]):
        # points in the back
        if (input_cloud[i, 0] < 0):
            continue

        distance = np.sqrt(input_cloud[i, 1]*input_cloud[i, 1]+input_cloud[i, 0]*input_cloud[i, 0])
        if (input_cloud[i, 1] == 0):
            front_pts_indices.append(i)
            continue

        # filter points aside
        if np.abs(input_cloud[i, 0] / input_cloud[i, 1]) > 2:
            front_pts_indices.append(i)
        # and keep points close
        elif (distance) < 10:
            front_pts_indices.append(i)

    return input_cloud[front_pts_indices]

# in most case we cannot find the good whole ground, so I sepeate the cloud into multiple parts to search
def seperate_the_cloud(input_cloud):
    clouds = []
    # seperate by x distance
    cut_threshold = 10.0
    cloud_1 = input_cloud[input_cloud[:,0] < cut_threshold]
    cloud_2 = input_cloud[input_cloud[:,0] > cut_threshold]
    clouds.append(cloud_1)
    clouds.append(cloud_2)
    return clouds

def fusion_cloud_parts(cloud_parts):
    result_cloud = cloud_parts[0]
    for i in range(len(cloud_parts)-1):
        result_cloud = np.concatenate((result_cloud, cloud_parts[i+1]), axis=0)
    return result_cloud


def segmentation_cloud(folder_point_cloud, file_gt, eps = 0.25):
    file_name = file_gt[-10:-4]
    file_velodyne = folder_point_cloud + '/' + file_name + '.bin'

    point_cloud_test = take_the_cloud_in_front_view(load_velo_scan(file_velodyne))

    cloud_parts = seperate_the_cloud(point_cloud_test)

    segmented_points_parts = []
    floor_cloud_parts = []
    final_thresholds = [0.2,0.2]
    num_iterations = [100, 100]

    for i in range(len(cloud_parts)):
        segmented_points_part, floor_cloud_part = mine_seg.ground_segmentation(cloud_parts[i],final_thresholds[i], num_iterations[i])
        segmented_points_parts.append(segmented_points_part)
        floor_cloud_parts.append(floor_cloud_part)

    segmented_points = fusion_cloud_parts(segmented_points_parts)
    floor_cloud = fusion_cloud_parts(floor_cloud_parts)

    # use DBSCAN to cluster the data, as we have no prior on the number of classes
    cluster_index = mine_seg.clustering(segmented_points, eps)

    return segmented_points, cluster_index, floor_cloud

# some cluster is a single line, we should filter these
def is_single_line_or_flat(cluster_pts):
    normalized_pts = cluster_pts - np.mean(cluster_pts ,0)
    w_nn, v_nn = mine_seg.PCA(normalized_pts)
    main_direction = np.transpose(v_nn[:, 0])

    sum_dot_product = 0
    for i in range(normalized_pts.shape[0]):
        thre = np.abs(np.dot(normalized_pts[i], main_direction)) / np.linalg.norm(normalized_pts[i])
        sum_dot_product += thre

    average_dot_product = sum_dot_product / normalized_pts.shape[0]
    #print(average_dot_product)

    if average_dot_product > 0.9:
        return True

    normal_direction = np.transpose(v_nn[:, 2])

    sum_dot_product = 0
    for i in range(normalized_pts.shape[0]):
        thre = np.abs(np.dot(normalized_pts[i], main_direction)) / np.linalg.norm(normalized_pts[i])
        sum_dot_product += thre

    average_dot_product = sum_dot_product / normalized_pts.shape[0]

    if average_dot_product < 0.1:
        return True

    return False

# whether the cloud is too closely grouped (into a single ball)
def is_too_close(cluster_pts):
    normalized_pts = cluster_pts - np.mean(cluster_pts ,0)
    norms = np.linalg.norm(normalized_pts, axis=1)
    #average_norm = np.mean(norms)
    #print(average_norm)
    check_distance = max(norms)

    if(check_distance < 0.3):
        return True
    return False

#process the cloud result of DBSCAN
def judge_the_dbscan_cloud(cluster_pts):
    #cluster_pts = pointcloud[cluster_index==index]
    ret = True
    max_size = 5.0
    max_tall = 2.0

    max_z = max(cluster_pts[:,2])
    min_z = min(cluster_pts[:,2])

    if(max_z > 0.8):
        return False

    if(max_z < -1):
        return False

    range_x = max(cluster_pts[:,0]) - min(cluster_pts[:,0])
    range_y = max(cluster_pts[:,1]) - min(cluster_pts[:,1])
    range_z = max_z - min_z

    #print(range_x, range_y)
    if range_x > max_size :
        ret = False
    elif range_y > max_size :
        ret = False
    elif range_z > max_tall:
        ret = False
    elif (is_single_line_or_flat(cluster_pts)):
        ret = False
    elif (is_too_close(cluster_pts)):
        ret = False

    return ret


def get_object_boxes(file_gt):
    file_name = file_gt[-10:-4]
    file_velodyne = folder_point_cloud + '/' + file_name + '.bin'

    file = open(file_gt, 'r')
    box_sets = []
    while True:
        text_line = file.readline()
        if text_line:
            label, h, w, l, t1, t2, t3, ry = decode_ground_truth_line_and_choose(text_line)
            if(t1 < -100):
                continue
            if(label == -1):
                continue

            box_set = [ry, t1, t2, t3, w, h, l]
            box_sets.append(box_set)
        else:
            break
    file.close()
    return box_sets, file_velodyne

def if_overlapping_the_box(cluster_pts, box_sets):
    for box_set in box_sets:
        points_in_box = rotate_and_threshold_pointcloud(cluster_pts, box_set[0], box_set[1], box_set[2], box_set[3], box_set[4], box_set[5], box_set[6])
        #print(points_in_box.shape[0])
        if(points_in_box.shape[0] > 0 ):
            return True
    return False

def decode_ground_truth_line(gt_line):
    # str, &trash, &trash, &d.box.alpha, &d.box.x1, &d.box.y1, &d.box.x2, &d.box.y2, &d.h, &d.w, &d.l, &d.t1, &d.t2, &d.t3, &d.ry
    splits_gt = gt_line.split(' ')
#     print('label is : ', splits_gt[0])
#     print('box height : ', splits_gt[8])
#     print('box width : ', splits_gt[9])
#     print('box length : ', splits_gt[10])
#     print('box coordinat : ',splits_gt[11], splits_gt[12], splits_gt[13] )
#     print('box angle : ', splits_gt[14])
    # return h, w, l, t1, t2, t3, ry
    return float(splits_gt[8]), float(splits_gt[9]), float(splits_gt[10]), float(splits_gt[11]), float(splits_gt[12]), float(splits_gt[13]), float(splits_gt[14]),


def load_ground_true(folder_point_cloud, file_gt):
    file_name = file_gt[-10:-4]
    file_velodyne = folder_point_cloud + '/' + file_name + '.bin'
    #print(file_velodyne)

    file = open(file_gt, 'r')
    box_sets = []
    while True:
        text_line = file.readline()
        if text_line:
            h, w, l, t1, t2, t3, ry = decode_ground_truth_line(text_line)
            if(t1 < -100):
                continue
            box_set = get_3d_box(ry, t1, t2, t3, w, h, l)
            #print(type(text_line), text_line)
            box_sets.append(box_set)
        else:
            break
    file.close()
    return box_sets, file_velodyne

    
