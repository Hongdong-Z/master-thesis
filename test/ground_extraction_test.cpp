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
    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/input/3600.pcd";
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/3800.pcd";
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/crossroad.pcd";
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/groundPointCloud.pcd";
    std::filesystem::path outPath = "/home/zhao/zhd_ws/src/localization_lidar/output/groundOutput";
    PointType nanPoint;


    std::shared_ptr<pcl::PointCloud<PointType>> inputCloud = std::make_shared<pcl::PointCloud<PointType>>();
    if (pcl::io::loadPCDFile<PointType>(inPath, *inputCloud) == -1) {
        PCL_ERROR("Couldn't read file from path : ");
        return (-1);
    }
    std::cout << "Loaded " << inputCloud->size()
              << " data points from 3600.pcd with the following fields: " << std::endl;


    //// Default parameter of VLS-128
    // Number of laser lines and columns
    const int N_SCAN = 128;
    const int Horizon_SCAN = inputCloud->points.size()/N_SCAN;
    // Vertical Angle
    // -25 deg bis +15 deg
    const float ang_bottom = 25.0+0.1; // -25 is laser line 0
//    const int groundScanInd = 70;
    float heightLaser = 1.73; // The height of laser 1.73
    float horizontalRevolution = 0.1; // 0.1 deg
    // Revolution between two scans
    float verticalRevolution = 0.3125;

    ////initialization and definition
    std::shared_ptr<pcl::PointCloud<PointType>> fullCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> fullInfoCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> groundCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> nonGroundCloud = std::make_shared<pcl::PointCloud<PointType>>();

    fullCloud->points.resize(N_SCAN*Horizon_SCAN);
    fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

    cv::Mat rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    cv::Mat groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    cv::Mat curbMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    cv::Mat surfaceMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    cv::Mat labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));

    nanPoint.x = std::numeric_limits<float>::quiet_NaN();
    nanPoint.y = std::numeric_limits<float>::quiet_NaN();
    nanPoint.z = std::numeric_limits<float>::quiet_NaN();
    nanPoint.intensity = -1;

    ////Initial
    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);

    std::cout << "*****************Start Process****************" << std::endl;


    //////////////////////////////////////////////////////////////////////////
    ////Ground Features Extraction
    /////////////////////////////////////////////////////////////////////////

    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle, distance; // For ground extraction
    Eigen::Vector3f vectorTwoSc(0,0,0);
    Eigen::Vector3f zAxis(0,0,1);
    for (size_t j = 0; j < Horizon_SCAN ; ++j) {
        for (size_t i = 0; i < groundScanInd - 1; ++i) {
            // //Sequenced point cloud access methods: j + i * N_SCAN
            lowerInd = i + j * N_SCAN; // row + column
            upperInd = (i + 1) + j * N_SCAN;

            //param for ground extraction
            diffX = inputCloud->points[upperInd].x - inputCloud->points[lowerInd].x;
            diffY = inputCloud->points[upperInd].y - inputCloud->points[lowerInd].y;
            diffZ = inputCloud->points[upperInd].z - inputCloud->points[lowerInd].z;
            vectorTwoSc[0] = diffX;
            vectorTwoSc[1] = diffY;
            vectorTwoSc[2] = diffZ;
//            angle = acos(vectorTwoSc.dot(zAxis)) * 180 / M_PI;
            angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;
//            distance = sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
            // proposed methode to extract ground && inputCloud->points[lowerInd].z < -1.4 && inputCloud->points[upperInd].z < -1.4
            if (angle < 10 ) {
                groundMat.at<int8_t>(i, j) = 1;
                groundMat.at<int8_t>(i + 1, j) = 1;
            }
        }
    }

    for (size_t i = 0; i < N_SCAN; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (groundMat.at<int8_t>(i,j) == 1 ){
                groundCloud->push_back(inputCloud->points[i + j * N_SCAN]);
            }else{
                nonGroundCloud->push_back(inputCloud->points[i + j * N_SCAN]);
            }
        }
    }

    std::cout << " Cloud size of Ground Cloud: " << groundCloud->points.size() << std::endl;
    std::cout << " Cloud size of nonGround Cloud: " << nonGroundCloud->points.size() << std::endl;

    pcl::io::savePCDFileASCII(outPath / "groundCloud_3600.pcd", *groundCloud);
    pcl::io::savePCDFileASCII(outPath / "nonGroundCloud_3600.pcd", *nonGroundCloud);


    std::cout << "*****************End processing***************" << std::endl;
    return 0;
}