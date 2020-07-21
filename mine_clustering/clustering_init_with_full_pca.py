# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d
from pyntcloud import PyntCloud
from collections import Counter

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def PCA(data):
    # SVD
    U,sigma,VT = np.linalg.svd(data)
    eigenvalues = sigma
    eigenvectors = np.transpose(VT)
    return eigenvalues, eigenvectors

def init_estimation_with_filtering(data):
    # randomly filter the points
    init_ids = np.random.choice(data.shape[0], int(data.shape[0]/10), replace=False)
    filtered_points = data[init_ids]
    #print('filtered data points num:', filtered_points.shape[0])

    # find the main plane
    w_nn, v_nn = PCA(filtered_points)
    point_cloud_main_normal = np.transpose(v_nn[:, 2])

    # filter the outliers (by clustering method)
    projections_on_normal = np.dot(filtered_points, point_cloud_main_normal).reshape(-1,1)
    n_clusters = 5
    cluster_algo = cluster.MiniBatchKMeans(n_clusters=n_clusters)
    cluster_algo.fit(projections_on_normal)
    normal_cluster_indices = cluster_algo.labels_.astype(np.int)
    counter_knn_result = Counter(normal_cluster_indices)
    # find the index with the most elements
    num_element = 0
    best_index = 0
    for i in range(n_clusters):
        num = counter_knn_result.get(i)
        if num > num_element:
            num_element = num
            best_index = i
    print(counter_knn_result, best_index)

    # refine the floor estimation
    # estimate the floor by SVD (PCA)
    floor_points = filtered_points[normal_cluster_indices==best_index]
    w_nn, v_nn = PCA(floor_points)
    point_cloud_main_normal = np.transpose(v_nn[:, 2])

    projections_on_normal = np.dot(floor_points, point_cloud_main_normal)
    mean_distance = np.mean((projections_on_normal))

    threshold = 0.4
    # filter outliers
    floor_flag = np.logical_and([projections_on_normal <= mean_distance + threshold], [projections_on_normal >= mean_distance - threshold])
    floor_points_2 = floor_points[floor_flag[0]]
    print("filter cloud , from ", floor_points.shape[0], " to ", floor_points_2.shape[0])
    w_nn, v_nn = PCA(floor_points_2)
    point_cloud_main_normal = np.transpose(v_nn[:, 2])
    projections_on_normal = np.dot(floor_points_2, point_cloud_main_normal)
    mean_distance = np.mean(projections_on_normal)

    return point_cloud_main_normal, mean_distance


# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # normalization of the data
    data = data - np.mean(data ,0)

    # to filter the groud from the point cloud
    # will use RANSAC method

    # as we are aimed at the outside car lidar data, we can assume the lidar
    # data is in some main direction, where the groud lies
    # as a result, the first step is to find the main component of the cloud
    # and try to model a plane based on this direction
    point_cloud_main_normal, mean_distance = init_estimation_with_filtering(data)
    print(point_cloud_main_normal, mean_distance)
    #mean_distance = 0.0
    threshold = 0.4

    projections_on_normal = (np.dot(data, point_cloud_main_normal))

    floor_flag = np.logical_and([projections_on_normal <= mean_distance + threshold], [projections_on_normal >= mean_distance - threshold])
    other_flag = np.logical_not(floor_flag)
    segmengted_cloud = data[other_flag[0]]
    floor_cloud = data[floor_flag[0]]

    #segmengted_cloud = filtered_points
    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])

    return segmengted_cloud, floor_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # use DBSCAN to cluster the data
    cluster_algo = cluster.DBSCAN(eps=1.0)
    #cluster_algo = cluster.MiniBatchKMeans(n_clusters=9)
    cluster_algo.fit(data)
    if hasattr(cluster_algo, 'labels_'):
        clusters_index = cluster_algo.labels_.astype(np.int)
    else:
        clusters_index = cluster_algo.predict(data)

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def plot_clusters_o3d(data, cluster_index, floor_cloud):
    #point_cloud_pynt = PyntCloud.from_file(file_name)
    #point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    colors = np.array(list(islice(cycle([[0,1,0], [1,1,0], [1,0,1],
                                         [0,0,1], [0,1,1], [0.5,0.5,0],
                                         [1,0,0], [0.3,0,0.6], [0.8,0,0.1]]),
                                  int(max(cluster_index) + 1))))
    colors = np.append(colors, [[0.1,0.9,0]]).reshape(-1,3)
    #print(colors)
    color = colors[cluster_index]

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(data[:,0:3].reshape(-1,3))
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(color.tolist())

    point_cloud_floor = o3d.geometry.PointCloud()
    point_cloud_floor.points = o3d.utility.Vector3dVector(floor_cloud[:,0:3].reshape(-1,3))
    colors_floor = [[0, 0, 0] for i in range(floor_cloud.shape[0])]
    point_cloud_floor.colors = o3d.utility.Vector3dVector(colors_floor)

    o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_floor])


def main():
    root_dir = 'data/'
    cat = os.listdir(root_dir)
    cat = cat[1:]
    iteration_num = len(cat)
    #iteration_num = 1

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points, floor_cloud = ground_segmentation(data=origin_points)
        cluster_index = clustering(segmented_points)

        #plot_clusters(segmented_points, cluster_index)
        plot_clusters_o3d(segmented_points, cluster_index, floor_cloud)

if __name__ == '__main__':
    main()
