//
// Created by zhao on 02.10.22.
//

#include <set>
#include <stdint.h>
#include <vector>

#include <algorithm>
#include <chrono>
#include <limits>
#include <math.h>
#include <random>
#include <thread>
#include <time.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/don.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>


#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

#include <ceres/ceres.h>

#include <nlopt.hpp>

#include <google_mv/levenberg_marquardt.h>

#include <ros/console.h>

#include <rviz_visual_tools/rviz_visual_tools.h>

#include <glog/logging.h>
#include "utility/dbscan.h"

int main() {
    std::cout << "******************mapping start!********************" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudColored(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::io::loadPCDFile("/home/zhao/zhd_ws/src/localization_lidar/primitiveMaps/new_map/SAVE/whole_pole_map_ori.pcd", *cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(1.07); // 2cm
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        //创建临时保存点云族的点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZI>);
        //通过下标，逐个填充
        int color = rand()%100000;
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++) {
            pcl::PointXYZI p;
            p.x = cloud->points[*pit].x;
            p.y = cloud->points[*pit].y;
            p.z = cloud->points[*pit].z;
            p.intensity = color;
            cloud_cluster->points.push_back(p);
        }
        *cloudColored += *cloud_cluster;
    }
    cloudColored->width = 1;
    cloudColored->height = cloudColored->points.size();
    pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/show/colored_poles_map.pcd", *cloudColored);

    std::cout << "******************mapping finish!********************" << std::endl;

    return 0;
}