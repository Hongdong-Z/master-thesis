//
// Created by zhao on 23.03.22.
//
//

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
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>


#include <Eigen/Dense>

#include <calib_storage/calib_storage.h>
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"

#include "common.h"
#include <feature_extraction/curbDetection.h>
#define random(x) rand()%(x)

using std::sin;
using std::cos;
using std::atan2;


int main() {
    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/input/3700.pcd";
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/crossroad.pcd";
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/3800.pcd";
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/groundCloud_3600.pcd";
    std::filesystem::path outPath = "/home/zhao/zhd_ws/src/localization_lidar/output/curbOutput";
    PointType nanPoint;




//    std::shared_ptr<pcl::PointCloud<PointType>> inputCloud = std::make_shared<pcl::PointCloud<PointType>>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(inPath, *inputCloud) == -1) {
        PCL_ERROR("Couldn't read file from path : ");
        return (-1);
    }
    std::cout << "Loaded " << inputCloud->size()
              << " data points from 3700.pcd with the following fields: " << std::endl;

    std::vector<pcl::PointCloud<pcl::PointXYZ>> input;
    for (int i = 0; i < 65; ++i) {
        for (int j = 0; j < 1809; ++j) {
            input[i].push_back(inputCloud->points[ i + j * 128]);
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr curbCloud(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::PointCloud<pcl::PointXYZ>::Ptr nonCurbCloud(new pcl::PointCloud<pcl::PointXYZ>);

    llo::curbDetector cD;
    *curbCloud = cD.detector(input);
    std::cout << "*****************************" << std::endl;
    pcl::visualization::CloudViewer viewer("curb");
    viewer.showCloud(curbCloud);

    pcl::io::savePCDFileASCII(outPath / "curbCloud_new_method.pcd", *curbCloud);
//    pcl::io::savePCDFileASCII(outPath / "nonCurbCloud_3900_new.pcd", *nonCurbCloud);


    std::cout << "*****************End processing***************" << std::endl;
    return 0;
}