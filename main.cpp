#include "include/filter.hpp"
#include "include/visualize.hpp"
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    auto cloud_raw = readPointCloudFromBin("a.bin");
    if (!cloud_raw) return -1;
    printCloudInfo(cloud_raw);

    auto cloud_filtered = filter(cloud_raw);
    std::cout << "滤波处理后点数: " << cloud_filtered->points.size() << std::endl;

    // 提取聚类簇
    auto clusters = extractClusters(cloud_filtered);
    std::cout << "聚类簇数量: " << clusters.size() << std::endl;

    // 构建窗口输入序列
    std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> clouds_vec;
    // raw点云
    clouds_vec.push_back({cloud_raw});
    // filtered点云
    clouds_vec.push_back({cloud_filtered});
    // 聚类点云
    clouds_vec.push_back(clusters);

    visualizeCloudsLinked(clouds_vec);

    return 0;
}