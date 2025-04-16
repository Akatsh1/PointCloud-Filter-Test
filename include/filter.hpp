#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

pcl::PointCloud<pcl::PointXYZ>::Ptr readPointCloudFromBin(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "无法打开文件 " << filename << std::endl;
        return nullptr;
    }
    struct PointXYZI {
        float x, y, z, intensity;
    };
    PointXYZI point;
    while (infile.read(reinterpret_cast<char*>(&point), sizeof(PointXYZI))) {
        cloud->points.emplace_back(point.x, point.y, point.z);
    }
    infile.close();
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud){
    // 参数设置
    double ground_distance_threshold_ = 0.2; // 地面分割的距离阈值
    double voxel_leaf_size_ = 0.1;           // 体素滤波的叶子大小
    int statistical_mean_k_ = 50;           // 统计滤波的邻域点数
    double statistical_stddev_ = 1.0;       // 统计滤波的标准差倍数
    double height_threshold_ = 2.0;         // 高度截取滤波的阈值
    double cluster_tolerance_ = 0.5;        // 欧式聚类的距离容忍度
    int min_cluster_size_ = 100;            // 欧式聚类的最小点数
    int max_cluster_size_ = 25000;          // 欧式聚类的最大点数

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>(*cloud));

    // 地面分割
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(ground_distance_threshold_);
    seg.setInputCloud(filtered);
    seg.segment(*inliers, *coefficients);

    // 提取非地面点
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(filtered);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*filtered);

    // 体素滤波
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(filtered);
    voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel.filter(*filtered);

    // 统计滤波去除离群点
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> statistical;
    statistical.setInputCloud(filtered);
    statistical.setMeanK(statistical_mean_k_);
    statistical.setStddevMulThresh(statistical_stddev_);
    statistical.filter(*filtered);

    // 高度截取滤波
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(filtered);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-height_threshold_, height_threshold_);
    pass.filter(*filtered);

    // 欧式聚类
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(filtered);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(filtered);
    ec.extract(cluster_indices);

    // // 可选：只保留最大聚类
    // if (!cluster_indices.empty()) {
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr largest_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    //     size_t max_size = 0;
    //     int max_index = 0;
    //     for (size_t i = 0; i < cluster_indices.size(); ++i) {
    //         if (cluster_indices[i].indices.size() > max_size) {
    //             max_size = cluster_indices[i].indices.size();
    //             max_index = i;
    //         }
    //     }
    //     for (int idx : cluster_indices[max_index].indices) {
    //         largest_cluster->points.push_back(filtered->points[idx]);
    //     }
    //     largest_cluster->width = largest_cluster->points.size();
    //     largest_cluster->height = 1;
    //     largest_cluster->is_dense = false;
    //     return largest_cluster;
    // }

    return filtered;
}

// 提取所有聚类簇
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> extractClusters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    double cluster_tolerance = 0.5,
    int min_cluster_size = 100,
    int max_cluster_size = 25000
) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_cluster_size);
    ec.setMaxClusterSize(max_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (int idx : indices.indices) {
            cluster->points.push_back(cloud->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = false;
        clusters.push_back(cluster);
    }
    return clusters;
}
