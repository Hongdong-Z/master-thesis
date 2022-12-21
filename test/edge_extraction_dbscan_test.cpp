//
// Created by zhao on 23.03.22.
//

#include "utility/dbscan.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <boost/thread/thread.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>

#include <calib_storage/calib_storage.h>
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"

#include "common.h"


using std::sin;
using std::cos;
using std::atan2;


int main() {
    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/input/ground_3900.pcd";
    std::filesystem::path outPath = "/home/zhao/zhd_ws/src/localization_lidar/output/clustering";

    std::shared_ptr<pcl::PointCloud<PointType>> inputCloud = std::make_shared<pcl::PointCloud<PointType>>();
    if (pcl::io::loadPCDFile<PointType>(inPath, *inputCloud) == -1) {
        PCL_ERROR("Couldn't read file from path : ");
        return (-1);
    }
    std::cout << "Loaded " << inputCloud->size()
              << " data points from 3600 with ground.pcd with the following fields: " << std::endl;


    ////initialization and definition
    std::shared_ptr<pcl::PointCloud<PointType>> nonGroundCloud = std::make_shared<pcl::PointCloud<PointType>>();
    nonGroundCloud = inputCloud;

    std::cout << " Cloud size of edge input Cloud: " << nonGroundCloud->points.size() << std::endl;

    //////////////////////////////////////////////////////////////////////////
    ////Edge Features Extraction
    /////////////////////////////////////////////////////////////////////////
    std::filesystem::path& savePath = outPath;
    std::pair<float, pcl::PointCloud<PointType>> edgeInputCloud(0.12*0.12, *nonGroundCloud);
    unsigned int minPoints = 3;

    utility::DBSCAN ds(minPoints);
//    std::vector<PointType> centroidVector;
//    centroidVector = ds.getClustering(savePath, edgeInputCloud); // save PointCloud in savePath return Centroid Vector
    std::shared_ptr<pcl::PointCloud<PointType>> centroidCloud = std::make_shared<pcl::PointCloud<PointType>>();
    *centroidCloud = ds.getClustering_curb(edgeInputCloud);
    std::cout << "The number of Clusters: " <<centroidCloud->points.size() << std::endl;
//    for (int i = 0; i < centroidVector.size(); ++i) {
//        centroidCloud->push_back(centroidVector[i]);
//    }
    pcl::io::savePCDFileASCII(outPath / "3900_curb_cluster.pcd", *centroidCloud);

    std::cout << "*****************End processing***************" << std::endl;
    return 0;
}