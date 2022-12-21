//
// Created by zhao on 28.03.22.
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
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <vtkAutoInit.h>

#include <Eigen/Dense>

#include <calib_storage/calib_storage.h>
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"

#include "common.h"


using std::sin;
using std::cos;
using std::atan2;




int main() {
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/output/segmentation/segmentedCloudPure.pcd";
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/crossroad.pcd";
    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/input/3800.pcd";
    std::filesystem::path outPath = "/home/zhao/zhd_ws/src/localization_lidar/output/surfaceOutput";

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
    const int Horizon_SCAN = 1800;
    // Vertical Angle
    // -25 deg bis +15 deg
    const float ang_bottom = 25.0+0.1; // -25 is laser line 0
    const int groundScanInd = 65;
    float heightLaser = 1.73; // The height of laser 1.73
    float horizontalRevolution = 0.2; // 0.1 - 0.4 deg
    // Revolution between two scans
    float verticalRevolution = 0.11;

    ////initialization and definition
    std::shared_ptr<pcl::PointCloud<PointType>> surfaceCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> nonSurfaceCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> testCloud = std::make_shared<pcl::PointCloud<PointType>>();



    cv::Mat groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    cv::Mat surfaceMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    cv::Mat curbMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    cv::Mat rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

    nanPoint.x = std::numeric_limits<float>::quiet_NaN();
    nanPoint.y = std::numeric_limits<float>::quiet_NaN();
    nanPoint.z = std::numeric_limits<float>::quiet_NaN();
    nanPoint.intensity = -1;

    std::cout << "*****************Start Process****************" << std::endl;
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle, distance; // For ground extraction

//    for (size_t i = 0; i < Horizon_SCAN ; ++i) {
//        for (size_t j = 0; j < groundScanInd; ++j) {
//            // //Sequenced point cloud access methods: j + i * N_SCAN
//            lowerInd = j + i * N_SCAN; // row + column
//            upperInd = (j + 1) + i * N_SCAN;
//
//            //param for ground extraction
//            diffX = inputCloud->points[upperInd].x - inputCloud->points[lowerInd].x;
//            diffY = inputCloud->points[upperInd].y - inputCloud->points[lowerInd].y;
//            diffZ = inputCloud->points[upperInd].z - inputCloud->points[lowerInd].z;
//
//            angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;
////            distance = sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ);
//            // proposed methode to extract ground
//            if (abs(angle) < 10 && inputCloud->points[lowerInd].z < -1.4 && inputCloud->points[upperInd].z < -1.4) {
//                groundMat.at<int8_t>(j, i) = 1;
//                groundMat.at<int8_t>(j + 1, i) = 1;
//            }
//        }
//    }

    //////////////////////////////////////////////////////////////////////////
    ////Curb Features Extraction
    /////////////////////////////////////////////////////////////////////////

//    size_t tempInd, maxIndex, minIndex, slidingWindowSize = 20;
//    float dX, dY, dZ, alphaX, alphaY, alphaXInv, alphaYInv;
//    PointType maxZPoint, minZPoint;
//    std::vector<float> tempPointsX(slidingWindowSize,0);
//    std::vector<float> tempPointsY(slidingWindowSize,0);
//    std::vector<float> tempPointsZ(slidingWindowSize,0);
//    std::vector<std::pair<size_t, size_t>> indexPair(slidingWindowSize, std::pair<size_t, size_t>(0,0));
//    std::vector<size_t> indexList(slidingWindowSize,0);
//    int count = 0;
//    for (size_t i = 0; i < groundScanInd ; ++i) {
//        for (size_t j = 0; j < Horizon_SCAN - slidingWindowSize; j+=slidingWindowSize) {
//
//            for (size_t k = 0; k < slidingWindowSize; ++k) {
//                tempInd = i + (j + k) * N_SCAN;
////                count++;
//
//                indexList.push_back(tempInd); // store position in point cloud
//                tempPointsX.push_back(inputCloud->points[tempInd].x); // store Y value in sliding window
//                tempPointsY.push_back(inputCloud->points[tempInd].y); // store Y value in sliding window
//                tempPointsZ.push_back(inputCloud->points[tempInd].z); // store Z value in sliding window
//
//                indexPair.push_back(std::make_pair(i,j+k)); // store position in curb mat
//            }
//
//            auto maxX = std::max_element(tempPointsX.begin(), tempPointsX.end());
//            auto minX = std::min_element(tempPointsX.begin(), tempPointsX.end());
//
//            auto maxY = std::max_element(tempPointsY.begin(), tempPointsY.end());
//            auto minY = std::min_element(tempPointsY.begin(), tempPointsY.end());
//
//            auto maxZ = std::max_element(tempPointsZ.begin(), tempPointsZ.end());
//            auto minZ = std::min_element(tempPointsZ.begin(), tempPointsZ.end());
//
//            maxIndex = maxZ - tempPointsZ.begin();
//            minIndex = minZ - tempPointsZ.begin();
//
//            maxZPoint = inputCloud->points[indexList[maxIndex]];
//            minZPoint = inputCloud->points[indexList[minIndex]];
//
//            if ((*maxZ - *minZ) > 0.05 || (*maxY - *minY) < 0.1 || (*maxX - *minX) < 0.1){
//                for (size_t l = 0; l < indexList.size()-5; ++l) {
//                    dX = inputCloud->points[indexList[l+5]].x - inputCloud->points[indexList[l]].x;
//                    dY = inputCloud->points[indexList[l+5]].y - inputCloud->points[indexList[l]].y;
//                    dZ = inputCloud->points[indexList[l+5]].z - inputCloud->points[indexList[l]].z;
//                    alphaX = atan2(abs(dY),abs(dZ)) * 180 / M_PI;
//                    alphaXInv = atan2(abs(dZ),abs(dY)) * 180 / M_PI;
//                    alphaY = atan2(abs(dX),abs(dZ)) * 180 / M_PI;
//                    alphaYInv = atan2(abs(dZ),abs(dX)) * 180 / M_PI;
//
//                    // curb in x direction && abs(dZ) > 0.008
//                    if (abs(dY) < 0.1 && alphaX < 30 && alphaYInv > 3){
//                        curbMat.at<int8_t>(indexPair[l].first, indexPair[l].second) = 1;
//                        curbMat.at<int8_t>(indexPair[l].first, indexPair[l+1].second) = 1;
//                        curbMat.at<int8_t>(indexPair[l].first, indexPair[l+2].second) = 1;
//                        curbMat.at<int8_t>(indexPair[l].first, indexPair[l+3].second) = 1;
//                        curbMat.at<int8_t>(indexPair[l].first, indexPair[l+4].second) = 1;
//                        curbMat.at<int8_t>(indexPair[l].first, indexPair[l+5].second) = 1;
//                    }
//                }
//            }
//            indexList.clear();
//            tempPointsZ.clear();
//            tempPointsY.clear();
//            indexPair.clear();
//        }
//    }

    //////////////////////////////////////////////////////////////////////////
    ////Surface Features Extraction
    /////////////////////////////////////////////////////////////////////////

    float thresholdSurface = 0.1;
    int rangeScan = 5;
    Eigen::Vector3f vectorLR (0,0,0);
    Eigen::Vector3f vectorDU (0,0,0);
    Eigen::Vector3f vectorP (0,0,0);

    Eigen::Vector3f vectorLR2 (0,0,0);
    Eigen::Vector3f vectorDU2 (0,0,0);

    Eigen::Vector3f surfaceNormal (0,0,0);
    Eigen::Vector3f surfaceNormal2 (0,0,0);

    Eigen::Vector3f normalProduct (0,0,0);


//    for (int i = rangeScan; i < N_SCAN-rangeScan-5; ++i) {
//        for (int j = rangeScan; j < Horizon_SCAN-rangeScan-5; ++j) {
//
//            // distance to selected line for each point
//            for (int k = rangeScan; k >= -rangeScan; --k) { // row
//                for (int l = rangeScan; l >= -rangeScan ; --l) { // column
//                    PointType topPoint = inputCloud->points[(i + rangeScan) + j * N_SCAN];
//                    PointType downPoint = inputCloud->points[(i - rangeScan) + j * N_SCAN];
//                    PointType leftPoint = inputCloud->points[i + (j - rangeScan) * N_SCAN];
//                    PointType rightPoint = inputCloud->points[i + (j + rangeScan) * N_SCAN];
//
//                    PointType topPoint2 = inputCloud->points[(i + rangeScan + 5) + j * N_SCAN];
//                    PointType downPoint2 = inputCloud->points[(i - rangeScan + 5) + j * N_SCAN];
//                    PointType leftPoint2 = inputCloud->points[i + (j - rangeScan + 5) * N_SCAN];
//                    PointType rightPoint2 = inputCloud->points[i + (j + rangeScan + 5) * N_SCAN];
//
//                    PointType  currentPoint = inputCloud->points[(i + k) + (j + l) * N_SCAN];
//
//                    vectorP(0) = currentPoint.x - leftPoint.x;
//                    vectorP(1) = currentPoint.y - leftPoint.y;
//                    vectorP(2) = currentPoint.z - leftPoint.z;
//
//                    vectorLR(0) = rightPoint.x - leftPoint.x;
//                    vectorLR(1) = rightPoint.y - leftPoint.y;
//                    vectorLR(2) = rightPoint.z - leftPoint.z;
//
//                    vectorDU(0) = topPoint.x - downPoint.x;
//                    vectorDU(1) = topPoint.y - downPoint.y;
//                    vectorDU(2) = topPoint.z - downPoint.z;
//
//                    ////
//
//                    vectorLR2(0) = rightPoint2.x - leftPoint2.x;
//                    vectorLR2(1) = rightPoint2.y - leftPoint2.y;
//                    vectorLR2(2) = rightPoint2.z - leftPoint2.z;
//
//                    vectorDU2(0) = topPoint2.x - downPoint2.x;
//                    vectorDU2(1) = topPoint2.y - downPoint2.y;
//                    vectorDU2(2) = topPoint2.z - downPoint2.z;
//
//                    surfaceNormal = vectorLR.cross(vectorDU);
//                    ////
//                    surfaceNormal2 = vectorLR2.cross(vectorDU2);
//                    float angle = atan2(surfaceNormal.dot(surfaceNormal2), surfaceNormal.norm()*surfaceNormal2.norm()) * 180 / M_PI;
//                    //abs(surfaceNormal[2]) < 0.001 &&
//                    if (abs(surfaceNormal[2]) < 0.01 && abs(angle) < 40){
//                        float distance = vectorP.dot(surfaceNormal) / surfaceNormal.norm();
//                        currentPoint.intensity = i + k + j + l;
//                        testCloud->push_back(currentPoint);
//                        if (abs(distance) < 0.04){
//                            surfaceMat.at<int8_t>(i + k, j + l) = 1;
//                        }
//                    }
//                }
//            }
//        }
//    }

    for (int i = 1; i < N_SCAN - 1; ++i) {
        for (int j = 2; j < Horizon_SCAN - 2; ++j) {
            PointType topPoint = inputCloud->points[(i + 1) + j * N_SCAN];
            PointType downPoint = inputCloud->points[(i - 1) + j * N_SCAN];
            PointType leftPoint = inputCloud->points[i + (j - 1) * N_SCAN];
            PointType rightPoint = inputCloud->points[i + (j + 1) * N_SCAN];

            PointType topPoint2 = inputCloud->points[(i + 1) + (j + 1) * N_SCAN];
            PointType downPoint2 = inputCloud->points[(i - 1) + (j + 1) * N_SCAN];
            PointType leftPoint2 = inputCloud->points[i + (j - 1 + 1) * N_SCAN];
            PointType rightPoint2 = inputCloud->points[i + (j + 2) * N_SCAN];


            vectorLR(0) = rightPoint.x - leftPoint.x;
            vectorLR(1) = rightPoint.y - leftPoint.y;
            vectorLR(2) = rightPoint.z - leftPoint.z;

            vectorDU(0) = topPoint.x - downPoint.x;
            vectorDU(1) = topPoint.y - downPoint.y;
            vectorDU(2) = topPoint.z - downPoint.z;
            surfaceNormal = vectorLR.cross(vectorDU);


            vectorLR2(0) = rightPoint2.x - leftPoint2.x;
            vectorLR2(1) = rightPoint2.y - leftPoint2.y;
            vectorLR2(2) = rightPoint2.z - leftPoint2.z;

            vectorDU2(0) = topPoint2.x - downPoint2.x;
            vectorDU2(1) = topPoint2.y - downPoint2.y;
            vectorDU2(2) = topPoint2.z - downPoint2.z;
            surfaceNormal2 = vectorLR2.cross(vectorDU2);

            normalProduct = surfaceNormal.cross(surfaceNormal2);

//            float verticalAngle = cos(atan2(surfaceNormal[2], surfaceNormal[0]*surfaceNormal[0] + surfaceNormal[1]*surfaceNormal[1]));

//            float ang = atan2(surfaceNormal.dot(surfaceNormal2), surfaceNormal.norm()*surfaceNormal2.norm()) * 180 / M_PI;
//            std::cout << verticalAngle << std::endl;
//            if (abs(surfaceNormal[2]) < 0.03){
//                testCloud->push_back(inputCloud->points[i + j * N_SCAN]);
//            }
            if (normalProduct.norm() < 0.01 && abs(surfaceNormal[2]) < 0.03){
                surfaceMat.at<int8_t>(i, j) = 1;
            }
        }
    }


    for (size_t i = 0; i < Horizon_SCAN; ++i){
        for (size_t j = 0; j < N_SCAN; ++j){
            if (surfaceMat.at<int8_t>(j,i) == 1 && groundMat.at<int8_t>(j,i) != 1 && curbMat.at<int8_t>(j,i) != 1){
                surfaceCloud->push_back(inputCloud->points[j + i * N_SCAN]);
            }else if (groundMat.at<int8_t>(j,i) != 1 && curbMat.at<int8_t>(j,i) != 1) {
                nonSurfaceCloud->push_back(inputCloud->points[j + i * N_SCAN]);
            }
        }
    }


    std::cout << " Cloud size of Surface Cloud: " << surfaceCloud->points.size() << std::endl;
    std::cout << " Cloud size of nonSurface Cloud: " << nonSurfaceCloud->points.size() << std::endl;

    pcl::io::savePCDFileASCII(outPath / "seg_sur.pcd", *surfaceCloud);
    pcl::io::savePCDFileASCII(outPath / "seg_non_sur.pcd", *nonSurfaceCloud);
//    pcl::io::savePCDFileASCII(outPath / "testCloud_3600.pcd", *testCloud);


    std::cout << "*****************End processing***************" << std::endl;
    return 0;
}

