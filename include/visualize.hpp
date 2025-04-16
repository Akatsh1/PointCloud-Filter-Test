#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <random>

void printCloudInfo(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    std::cout << "读取点数: " << cloud->points.size() << std::endl;
    if (!cloud->points.empty()) {
        std::cout << "前5个点:" << std::endl;
        for (size_t i = 0; i < std::min<size_t>(5, cloud->points.size()); ++i) {
            std::cout << cloud->points[i].x << " "
                      << cloud->points[i].y << " "
                      << cloud->points[i].z << std::endl;
        }
    }
}

void setViewerCameraToCloudCenter(
    pcl::visualization::PCLVisualizer::Ptr viewer,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud
) {
    if (!cloud->points.empty()) {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, centroid);
        viewer->setCameraPosition(
            centroid[0], centroid[1], centroid[2] + 30,
            centroid[0], centroid[1], centroid[2],
            0, -1, 0
        );
    }
}

void visualizeCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
    viewer->initCameraParameters();
    setViewerCameraToCloudCenter(viewer, cloud);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void visualizeClusters(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Clusters Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    std::mt19937 rng(42); // 固定种子，保证颜色一致
    std::uniform_int_distribution<int> dist(0, 255);

    for (size_t i = 0; i < clusters.size(); ++i) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(
            clusters[i], dist(rng), dist(rng), dist(rng));
        std::string cloud_name = "cluster_" + std::to_string(i);
        viewer->addPointCloud<pcl::PointXYZ>(clusters[i], color, cloud_name);
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);
    }

    viewer->initCameraParameters();
    // 可选：将相机设置到第一个聚类中心
    if (!clusters.empty()) {
        setViewerCameraToCloudCenter(viewer, clusters[0]);
    }
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void visualizeCloudsLinked(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud1,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud2
) {
    // 创建两个窗口
    pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Raw Cloud"));
    pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Filtered Cloud"));

    viewer1->setBackgroundColor(0, 0, 0);
    viewer2->setBackgroundColor(0, 0, 0);

    viewer1->addPointCloud<pcl::PointXYZ>(cloud1, "cloud1");
    viewer2->addPointCloud<pcl::PointXYZ>(cloud2, "cloud2");

    viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud1");
    viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");

    viewer1->initCameraParameters();
    viewer2->initCameraParameters();

    // 设置初始相机位置为点云中心
    setViewerCameraToCloudCenter(viewer1, cloud1);
    setViewerCameraToCloudCenter(viewer2, cloud2);

    // 联动相机参数
    while (!viewer1->wasStopped() && !viewer2->wasStopped()) {
        // 获取viewer1的相机参数
        pcl::visualization::Camera cam;
        viewer1->getCameraParameters(cam);
        // 设置到viewer2
        viewer2->setCameraParameters(cam);

        viewer1->spinOnce(50);
        viewer2->spinOnce(50);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// 支持普通点云与聚类点云混合输入的联动可视化
inline void visualizeCloudsLinked(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud1,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clusters2
) {
    pcl::visualization::PCLVisualizer::Ptr viewer1(new pcl::visualization::PCLVisualizer("Raw Cloud"));
    pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Clusters Cloud"));

    viewer1->setBackgroundColor(0, 0, 0);
    viewer2->setBackgroundColor(0, 0, 0);

    viewer1->addPointCloud<pcl::PointXYZ>(cloud1, "cloud1");
    viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud1");

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < clusters2.size(); ++i) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(
            clusters2[i], dist(rng), dist(rng), dist(rng));
        std::string cloud_name = "cluster_" + std::to_string(i);
        viewer2->addPointCloud<pcl::PointXYZ>(clusters2[i], color, cloud_name);
        viewer2->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);
    }

    viewer1->initCameraParameters();
    viewer2->initCameraParameters();

    setViewerCameraToCloudCenter(viewer1, cloud1);
    if (!clusters2.empty()) setViewerCameraToCloudCenter(viewer2, clusters2[0]);

    while (true) {
        if (viewer1->wasStopped() || viewer2->wasStopped()) break;
        pcl::visualization::Camera cam;
        viewer1->getCameraParameters(cam);
        viewer2->setCameraParameters(cam);
        viewer1->spinOnce(50);
        viewer2->spinOnce(50);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// 支持任意数量点云（普通点云或聚类点云）的联动可视化
inline void visualizeCloudsLinked(
    const std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>>& clouds_vec
) {
    // clouds_vec: 每个元素为一个窗口的数据（可为单一普通点云或聚类点云集合）
    std::vector<pcl::visualization::PCLVisualizer::Ptr> viewers;
    std::vector<std::string> titles;
    for (size_t i = 0; i < clouds_vec.size(); ++i) {
        std::string title = "Cloud Window " + std::to_string(i);
        titles.push_back(title);
        viewers.emplace_back(new pcl::visualization::PCLVisualizer(title));
        viewers.back()->setBackgroundColor(0, 0, 0);
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    // 添加点云到各自窗口
    for (size_t i = 0; i < clouds_vec.size(); ++i) {
        auto& viewer = viewers[i];
        const auto& clouds = clouds_vec[i];
        if (clouds.size() == 1) {
            // 单一普通点云
            viewer->addPointCloud<pcl::PointXYZ>(clouds[0], "cloud");
            viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
            setViewerCameraToCloudCenter(viewer, clouds[0]);
        } else {
            // 多个聚类点云
            for (size_t j = 0; j < clouds.size(); ++j) {
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(
                    clouds[j], dist(rng), dist(rng), dist(rng));
                std::string cname = "cluster_" + std::to_string(j);
                viewer->addPointCloud<pcl::PointXYZ>(clouds[j], color, cname);
                viewer->setPointCloudRenderingProperties(
                    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cname);
            }
            if (!clouds.empty()) setViewerCameraToCloudCenter(viewer, clouds[0]);
        }
        viewer->initCameraParameters();
    }

    // 联动相机参数
    while (true) {
        // 任意窗口关闭则全部退出
        bool anyStopped = false;
        for (auto& v : viewers) {
            if (v->wasStopped()) {
                anyStopped = true;
                break;
            }
        }
        if (anyStopped) break;

        // 以第一个窗口为主同步相机
        pcl::visualization::Camera cam;
        viewers[0]->getCameraParameters(cam);
        for (size_t i = 1; i < viewers.size(); ++i) {
            viewers[i]->setCameraParameters(cam);
        }

        for (auto& v : viewers) v->spinOnce(50);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

