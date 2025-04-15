#include <iostream>
#include <fstream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

inline pcl::PointCloud<pcl::PointXYZ>::Ptr filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
inline pcl::PointCloud<pcl::PointXYZ>::Ptr readPointCloudFromBin(const std::string& filename);
inline void printCloudInfo(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
inline void setViewerCameraToCloudCenter();
inline void visualizeCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
inline void visualizeCloudsLinked(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud1,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud2
);

int main(int argc, char** argv) {
    auto cloud_raw = readPointCloudFromBin("a.bin");
    if (!cloud_raw) return -1;
    printCloudInfo(cloud_raw);

    auto cloud_filtered = filter(cloud_raw);
    std::cout << "滤波处理后点数: " << cloud_filtered->points.size() << std::endl;

    visualizeCloudsLinked(cloud_raw, cloud_filtered);

    return 0;
}

inline pcl::PointCloud<pcl::PointXYZ>::Ptr readPointCloudFromBin(const std::string& filename) {
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

inline void printCloudInfo(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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

inline void setViewerCameraToCloudCenter(
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

inline void visualizeCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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

inline void visualizeCloudsLinked(
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

inline pcl::PointCloud<pcl::PointXYZ>::Ptr filter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud){
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