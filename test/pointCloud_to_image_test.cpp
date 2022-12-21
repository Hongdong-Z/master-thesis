


#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

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
#define random(x) rand()%(x)

using std::sin;
using std::cos;
using std::atan2;

boost::shared_ptr<pcl::visualization::PCLVisualizer> customColourVis (pcl::PointCloud<PointType>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    //创建一个自定义的颜色处理器PointCloudColorHandlerCustom对象，并设置颜色为纯绿色
    pcl::visualization::PointCloudColorHandlerCustom<PointType> single_color(cloud, 0, 255, 0);
    //addPointCloud<>()完成对颜色处理器对象的传递
    viewer->addPointCloud<PointType> (cloud, single_color, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] == target) {
                    return {i,j};
                }
            }
        }
        return {};
    }
};

int main() {
    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/input.pcd";
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/crossroad.pcd";
//    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/groundPointCloud.pcd";
    std::filesystem::path outPath = "/home/zhao/zhd_ws/src/localization_lidar";
    PointType point;

    std::srand((int)time(0));

    std::shared_ptr<pcl::PointCloud<PointType>> inputCloud = std::make_shared<pcl::PointCloud<PointType>>();
    if (pcl::io::loadPCDFile<PointType>(inPath, *inputCloud) == -1) {
        PCL_ERROR("Couldn't read file from path : ");
        return (-1);
    }
    std::cout << "Loaded " << inputCloud->size()
              << " data points from 3600.pcd with the following fields: " << std::endl;

    //// Default parameter of VLS-128
    // Number of laser lines and columns
    const int N_SCANS = 128;
    const int N_COLUMN = 3272;
    // Vertical Angle
    const int verticalAngle = 40; // -25 deg bis +15 deg
    float heightLaser = 1.73; // The height of laser
    float horizontalRevolution = 0.11; // 0.11 deg
    // Revolution between two scans
    float verticalRevolution = 0.3125;
    // Label of picked: 0 not picked, 1 picked. 231000: The number of points in each frame
    int cloudPicked[231000];

    std::cout << "PC size before: " << inputCloud->size() << std::endl;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*inputCloud, *inputCloud, indices);

    std::cout << "PC size after removeNaN: " << inputCloud->size() << std::endl;
    ////for checking data
//    std::vector<float> showData;
//    std::ofstream debugFile;
//    debugFile.open(outPath / "threshold.txt", std::ios::out);
//    // color range
    std::vector<std::vector<int>> colorPcList(N_SCANS);
    for (int i = 0; i < colorPcList.size(); ++i) {
        int index = i / 22;
        switch (index) {
        case 0://red - orange
            colorPcList[i].push_back(255);
            colorPcList[i].push_back(int(i * 165 / 22));
            colorPcList[i].push_back(0);
            continue;
        case 1://orange - yellow
            colorPcList[i].push_back(255);
            colorPcList[i].push_back(165 + int((i - 22 * index) * 90 / 22));
            colorPcList[i].push_back(0);
            continue;
        case 2://yellow - green
            colorPcList[i].push_back(255 - int((i - 22 * index) * 255 / 22));
            colorPcList[i].push_back(255);
            colorPcList[i].push_back(0);
            continue;
        case 3://green - cyan-blue
            colorPcList[i].push_back(0);
            colorPcList[i].push_back(255 - int((i - 22 * index) * 128 / 22));
            colorPcList[i].push_back(int((i - 22 * index) * 255 / 22));
            continue;
        case 4://cyan-blue - blue
            colorPcList[i].push_back(0);
            colorPcList[i].push_back(128 - int((i - 22 * index) * 128 / 22));
            colorPcList[i].push_back(255);
            continue;
        case 5://blue - purple
            colorPcList[i].push_back(int((i - 22 * index) * 139 / 22));
            colorPcList[i].push_back(0);
            colorPcList[i].push_back(255);
            continue;
        }

    }
    bool beBug = false;
    if (beBug) {
        for (int i = 0; i < colorPcList.size(); ++i) {
            int index = i / 22;
            switch (index) {
            case 0:
                std::cout << "red - orange " << colorPcList[i][0] << "  " << colorPcList[i][1] << "  "
                          << colorPcList[i][2] << "  " << std::endl;
                continue;
            case 1:
                std::cout << "orange - yellow " << colorPcList[i][0] << "  " << colorPcList[i][1] << "  "
                          << colorPcList[i][2] << "  " << std::endl;
                continue;
            case 2:
                std::cout << "yellow - green " << colorPcList[i][0] << "  " << colorPcList[i][1] << "  "
                          << colorPcList[i][2] << "  " << std::endl;
                continue;
            case 3:
                std::cout << "green - cyan-blue " << colorPcList[i][0] << "  " << colorPcList[i][1] << "  "
                          << colorPcList[i][2] << "  " << std::endl;
                continue;
            case 4:
                std::cout << "cyan-blue - blue " << colorPcList[i][0] << "  " << colorPcList[i][1] << "  "
                          << colorPcList[i][2] << "  " << std::endl;
                continue;
            case 5:
                std::cout << "blue - purple " << colorPcList[i][0] << "  " << colorPcList[i][1] << "  "
                          << colorPcList[i][2] << "  " << std::endl;
                continue;
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////
    ////Store Point Cloud in 2D Matrix according to scans and horizontal angle
    /////////////////////////////////////////////////////////////////////////

    std::cout << "*****************Start processing***************" << std::endl;

    int cloudSize = inputCloud->size();
//    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
//    std::vector<std::vector<PointType>> projectedPoints(N_SCANS, std::vector<PointType>(N_COLUMN));
    Eigen::Matrix<PointType, Eigen::Dynamic, Eigen::Dynamic> projectedMatrix(N_SCANS, N_COLUMN);
//    std::cout << projectedPoints.capacity() << std::endl;
    int count = cloudSize;

    for (int i = 0; i < cloudSize; i++) {
        // identical coordinate system no need for transform
        point.x = inputCloud->points[i].x;
        point.y = inputCloud->points[i].y;
        point.z = inputCloud->points[i].z;
        point.intensity = inputCloud->points[i].intensity;
        // Calculate the elevation angle of points
        float angle = atan2(point.z , sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID;
//        showData.push_back(angle);
        // angle range in -25 deg bis 15 deg
        if (angle > -25.0 && angle < 15.0) {
            if (angle > 0){
                scanID = (int) (((15.0 - angle) / verticalRevolution) * 10 + 5) / 10;
            }else{
                scanID = (int) (((15.0 - angle) / verticalRevolution) * 10 - 5) / 10;
            }
//            point.r = inputCloud->points[i].r = colorPcList[scanID][0];
//            point.g = inputCloud->points[i].g = colorPcList[scanID][1];
//            point.b = inputCloud->points[i].b = colorPcList[scanID][2];
        }
        else{
            count--;
            continue;
        }

        //calculate horizontal angle
        float ori = rad2deg(atan2(point.y, point.x));
        int horizontalID;
        horizontalID = (int) ((ori + 180.0) / horizontalRevolution * 10 + 5) / 10;

//        laserCloudScans[scanID].push_back(point);
        projectedMatrix(scanID, horizontalID) = point;
    }

    std::cout << "The size of projectedMatrix: " << projectedMatrix.size() << std::endl;
    std::cout << "The number of points after projecting(points in -25 ~ +15 deg): " << count << std::endl;

    //////////////////////////////////////////////////////////////
    ////Extraction Ground Feature
    //////////////////////////////////////////////////////////////
    //to store ground points
    std::shared_ptr<pcl::PointCloud<PointType>> groundPointCloud = std::make_shared<pcl::PointCloud<PointType>>();
    //to store non ground points
    std::shared_ptr<pcl::PointCloud<PointType>> nonGroundPointCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::vector<float> disXYList;
    // count for null points
    int invalidCount = 0;
    for (int i = 0; i < N_COLUMN; ++i) { // Compute the features of two adjacent points in each column
        for (int j = 0; j < N_SCANS; ++j) {// Iterate over all rows, starting finding from row 0 forward
            // Remove invalid points
            if (projectedMatrix(j,i).x == 0 && projectedMatrix(j,i).y == 0 &&
                     projectedMatrix(j,i).z == 0 && projectedMatrix(j,i).intensity == 0){
                invalidCount++;
                continue; // do not process null points
            }

            PointType currentRowPoint = projectedMatrix(j, i);
            PointType nextRowPoint = projectedMatrix(j + 1 , i);
            float deltaX = nextRowPoint.x - currentRowPoint.x;
            float deltaY = nextRowPoint.y - currentRowPoint.y;
            float deltaZ = nextRowPoint.z - currentRowPoint.z;
            float absDeltaZ = sqrt(deltaZ * deltaZ);
            float alpha = atan2(absDeltaZ, sqrt(deltaX * deltaX + deltaY * deltaY)) * 180 / M_PI;
            float absXY = sqrt(deltaX * deltaX + deltaY * deltaY);

//            disXYList.push_back(absXY);
            //j > 50 && absXY > 1 && absDeltaZ < 0.05
//            if (currentRowPoint.z < -1.5 ){
//                if (absXY > 0.5){
//                    groundPointCloud->push_back(currentRowPoint);
//                }
//            }else{
//                nonGroundPointCloud->push_back(currentRowPoint);
//            }
            if (alpha < 0.625 && j > 50){
                groundPointCloud->push_back(currentRowPoint);
            }
        }
    }
    //////////////////////////////////////////////////////////////
    ////New Methode To Extraction Ground Feature
    //////////////////////////////////////////////////////////////

    for (int i = 0; i < N_SCANS; ++i) {
        for (int j = 20; j < N_COLUMN - 20; ++j) {

            PointType currentColumnPoint = projectedMatrix(i, j);
            PointType nextColumnPoint = projectedMatrix(i, j+20);
            PointType vorColumnPoint = projectedMatrix(i, j-20);

            float deltaX = projectedMatrix(i + 1, j).x - projectedMatrix(i, j).x;
            float deltaY = projectedMatrix(i + 1, j).y - projectedMatrix(i, j).y;
            float deltaZ = projectedMatrix(i + 1, j).z - projectedMatrix(i, j).z;
            float absDeltaZ = sqrt(deltaZ * deltaZ);
            float alpha = atan2(absDeltaZ, sqrt(deltaX * deltaX + deltaY * deltaY)) * 180 / M_PI;
            float absXY = sqrt(deltaX * deltaX + deltaY * deltaY);

            float h = nextColumnPoint.x - currentColumnPoint.x;
            float dY = (nextColumnPoint.y - currentColumnPoint.y) / h;
            float ddY = (nextColumnPoint.y + vorColumnPoint.y - 2 * currentColumnPoint.y) / (h * h);
            float currentCurvature = sqrt(ddY * ddY) / pow(1 + dY * dY, 3.0/2.0);

            float averageRange = (sqrt(currentColumnPoint.x * currentColumnPoint.x + currentColumnPoint.y * currentColumnPoint.y + currentColumnPoint.z * currentColumnPoint.z)
                                  + sqrt(nextColumnPoint.x * nextColumnPoint.x + nextColumnPoint.y * nextColumnPoint.y + nextColumnPoint.z * nextColumnPoint.z)
                                  + sqrt(vorColumnPoint.x * vorColumnPoint.x + vorColumnPoint.y * vorColumnPoint.y + vorColumnPoint.z * vorColumnPoint.z)) / 3;

            // new methode to extract ground feature
            if (currentColumnPoint.z < -1.5){
                if (1 / currentCurvature > averageRange - 5 && 1 / currentCurvature <= averageRange + 5){
//                    groundPointCloud->push_back(currentColumnPoint);
                }
            }
        }
    }



    //////////////////////////////////////////////////////////////
    ////Extraction Road-Curb Feature
    //////////////////////////////////////////////////////////////
    std::shared_ptr<pcl::PointCloud<PointType>> curbPointCloud = std::make_shared<pcl::PointCloud<PointType>>();
    for (int i = 0; i < N_SCANS; ++i) { // Compute the features of two adjacent points in each column
        for (int j = 0; j < N_COLUMN; ++j) {// Iterate over all rows, starting finding from row 0 forward

            PointType currentColumnPoint = projectedMatrix(i, j);
            PointType nextColumnPoint = projectedMatrix(i, j + 5);
//            float thresholdV =
//                atan2(currentColumnPoint.z , sqrt(currentColumnPoint.x * currentColumnPoint.x + currentColumnPoint.y * currentColumnPoint.y)) * 180 / M_PI;
            float thresholdV = 15 - verticalRevolution * i;
            float thresholdX = heightLaser * 1 / tan(thresholdV) * (1 - cos(horizontalRevolution));
            float thresholdY = heightLaser * 1 / tan(thresholdV) * sin(horizontalRevolution);
            float thresholdZ = 0.009;

            float deltaX = currentColumnPoint.x - nextColumnPoint.x;
            float deltaY = currentColumnPoint.y - nextColumnPoint.y;
            float deltaZ = currentColumnPoint.z - nextColumnPoint.z;
//
//            if (deltaX > thresholdX && deltaY > thresholdY && deltaZ > thresholdZ){
//                curbPointCloud->push_back(currentColumnPoint);
//            }
            if (currentColumnPoint.z < -1.5){
                if (deltaX > thresholdX && deltaY > thresholdY && deltaZ > thresholdZ){
                    curbPointCloud->push_back(currentColumnPoint);
                }
            }
//            if (deltaZ >0.003){
//                curbPointCloud->push_back(currentColumnPoint);
//            }
        }
    }


    //////////////////////////////////////////////////////////////
    ////Extraction Surface Feature
    //////////////////////////////////////////////////////////////
    std::shared_ptr<pcl::PointCloud<PointType>> surfacePointCloud = std::make_shared<pcl::PointCloud<PointType>>();
    float thresholdSurface = 0.001;
//    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> scansThreshold(N_SCANS, N_COLUMN);
//    std::vector<std::vector<float>> scansThreshold(N_SCANS, std::vector<float>());
    for (int i = 0; i < N_SCANS; ++i) {
        for (int j = 5; j < N_COLUMN - 5; ++j) {
            if (projectedMatrix(i,j).x == 0 && projectedMatrix(i,j).y == 0 && projectedMatrix(i,j).z == 0 && projectedMatrix(i,j).intensity == 0){
                continue;
            }
            float diffX = projectedMatrix(i, j - 5).x + projectedMatrix(i, j - 4).x
                          + projectedMatrix(i, j - 3).x + projectedMatrix(i, j - 2).x
                          + projectedMatrix(i, j - 1).x - 10 * projectedMatrix(i, j).x
                          + projectedMatrix(i, j + 1).x + projectedMatrix(i, j + 2).x
                          + projectedMatrix(i, j + 3).x + projectedMatrix(i, j + 4).x
                          + projectedMatrix(i, j + 5).x;
            float diffY = projectedMatrix(i, j - 5).y + projectedMatrix(i, j - 4).y
                          + projectedMatrix(i, j - 3).y + projectedMatrix(i, j - 2).y
                          + projectedMatrix(i, j - 1).y - 10 * projectedMatrix(i, j).y
                          + projectedMatrix(i, j + 1).y + projectedMatrix(i, j + 2).y
                          + projectedMatrix(i, j + 3).y + projectedMatrix(i, j + 4).y
                          + projectedMatrix(i, j + 5).y;
//            float diffZ = projectedMatrix(i, j - 5).z + projectedMatrix(i, j - 4).z
//                          + projectedMatrix(i, j - 3).z + projectedMatrix(i, j - 2).z
//                          + projectedMatrix(i, j - 1).z - 10 * projectedMatrix(i, j).z
//                          + projectedMatrix(i, j + 1).z + projectedMatrix(i, j + 2).z
//                          + projectedMatrix(i, j + 3).z + projectedMatrix(i, j + 4).z
//                          + projectedMatrix(i, j + 5).z;

//            scansThreshold[i].push_back(diffX * diffX + diffY * diffY);
            float scanCurvature = diffX * diffX + diffY * diffY;
            if (scanCurvature < 1 && projectedMatrix(i,j).z > -1.5){
                surfacePointCloud->push_back(projectedMatrix(i,j));
            }
        }
    }

//    for (int i = 0; i < N_SCANS; ++i) {
//        std::sort(scansThreshold[i].begin(), scansThreshold[i].end());
//        }

//    for (int i = 0; i < N_SCANS; ++i) {
//        for (int j = 5; j < N_COLUMN - 5; j++) {
//            if (projectedMatrix(i,j).x == 0 && projectedMatrix(i,j).y == 0 && projectedMatrix(i,j).z == 0 && projectedMatrix(i,j).intensity == 0){
//                continue;
//            }
//            PointType currentColumnPoint = projectedMatrix(i, j);
//            PointType nextColumnPoint = projectedMatrix(i, j+5);
//            PointType vorColumnPoint = projectedMatrix(i, j-5);
//
//            float pX = (vorColumnPoint.x + currentColumnPoint.x + nextColumnPoint.x) / 3 - currentColumnPoint.x;
//            float pY = (vorColumnPoint.y + currentColumnPoint.y + nextColumnPoint.y) / 3 - currentColumnPoint.y;
//            float pZ = (vorColumnPoint.z + currentColumnPoint.z + nextColumnPoint.z) / 3 - currentColumnPoint.z;
//
//            float h = nextColumnPoint.x - currentColumnPoint.x;
//            float dY = (nextColumnPoint.y - currentColumnPoint.y) / h;
//            float ddY = (nextColumnPoint.y + vorColumnPoint.y - 2 * currentColumnPoint.y) / (h * h);
//            float currentCurvature = sqrt(ddY * ddY) / pow(1 + dY * dY, 3.0/2.0);

            // proposed method
//            if (currentCurvature < 0.005){
//                surfacePointCloud->push_back(currentColumnPoint);
//            }
            // papier proposed method
//            if (pX < thresholdSurface && pY < thresholdSurface & pZ < thresholdSurface){
//                surfacePointCloud->push_back(currentColumnPoint);
//            }
//
//        }
//    }


    std::cout << "Null points number in projected Matrix: " << invalidCount << std::endl;



//
//    for (int i = 0; i < N_SCANS; ++i) {
//        for (int j = 0; j < scansThreshold[i].size(); ++j) {
//            debugFile << projectedMatrix(i,j) << "  ";
//            debugFile << scansThreshold[i][j] << "  ";
//        }
//        debugFile << std::endl;
//    }
//    debugFile.close();

    std::cout << "*****************Finish Processing**************" << std::endl;
//    pcl::PointCloud<PointType> fusionPointCloud;
//    for (int i = 0; i < N_SCANS; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            fusionPointCloud.push_back(projectedMatrix(i, 50));
//        }
//        //        fusionPointCloud += laserCloudScans[i];
//    }
//    pcl::io::savePCDFileASCII(outPath / "output.pcd", fusionPointCloud);

    pcl::io::savePCDFileASCII(outPath / "groundPointCloud.pcd", *groundPointCloud); //Ground Points
//    pcl::io::savePCDFileASCII(outPath / "nonGroundPointCloud.pcd", *nonGroundPointCloud); // nonGround Points
//    pcl::io::savePCDFileASCII(outPath / "curbPointCloud.pcd", *curbPointCloud); // Curb Points
//    pcl::io::savePCDFileASCII(outPath / "surfacePointCloud.pcd", *surfacePointCloud); // Surface Points
//    pcl::PointCloud<PointType>::ConstPtr showPointCloud (&*groundPointCloud);

//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
//    viewer = customColourVis(showPointCloud);
//    while (!viewer->wasStopped ())
//    {
//        viewer->spinOnce (100);
//        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//    }


    return 0;
}