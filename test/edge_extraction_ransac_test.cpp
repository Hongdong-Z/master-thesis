//
// Created by zhao on 16.05.22.
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
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>

#include <Eigen/Dense>

#include <calib_storage/calib_storage.h>
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"

#include "common.h"


using std::sin;
using std::cos;
using std::atan2;

class mesh {
public:
    mesh() {}
    float high = 0.7;
    int density = 0; // the number of point in this mesh
    int state = 0; // state of this mesh 1: active, 0: inactive
    std::vector<int> pointsIndex;
};


int main() {
    std::filesystem::path inPath = "/home/zhao/zhd_ws/src/localization_lidar/output/surfaceOutput/nonSurfaceCloud_3600.pcd";
    std::filesystem::path outPath = "/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/edge";

    std::shared_ptr<pcl::PointCloud<PointType>> inputCloud = std::make_shared<pcl::PointCloud<PointType>>();
    if (pcl::io::loadPCDFile<PointType>(inPath, *inputCloud) == -1) {
        PCL_ERROR("Couldn't read file from path : ");
        return (-1);
    }
    std::cout << "Loaded " << inputCloud->size()
              << " data points from 3600 with ground.pcd with the following fields: " << std::endl;


    int length = 300; // x in [-150, 150]
    int width = 80; // y in [-40, 40]
    float meshSize = 0.15; // size of mesh
    int minNumPoint = 60; // min number of point in cell

    int row = std::floor(width/meshSize);
    int column = std::floor(length/meshSize);

//    cv::Mat edgeMat = cv::Mat(row, column, CV_8S, cv::Scalar::all(0)); //(y, x) mat for show the result
    std::vector<std::vector<mesh>> allMesh(column + 1); // column for x row for y
    for (int i = 0; i < column + 1; ++i) {
        allMesh[i].resize(row + 1);
    }


    // fill the mesh with points
    for (int i = 0; i < inputCloud->points.size(); ++i) {
        float x = inputCloud->points[i].x;
        float y = inputCloud->points[i].y;
        float z = inputCloud->points[i].z;
        if (abs(x) < length/2 && abs(y) < width/2 && z < 0.7) {  // range of selected area
            int c = std::floor((x - (-length / 2)) / meshSize); // x
            int r = std::floor((y - (-width / 2)) / meshSize); // y
            allMesh[c][r].pointsIndex.push_back(i);
            allMesh[c][r].density++;
        }
    }

    // label valid mesh
    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].density > minNumPoint) {
                allMesh[i][j].state = 1;
//                edgeMat.at<int8_t>(j,i) = allMesh[i][j].density;
            }
        }
    }

    pcl::PointCloud<PointType>::Ptr edgeCloud(new pcl::PointCloud<PointType>);
    for (int i = 0; i <= column; ++i) {
        for (int j = 0; j <= row; ++j) {

            if (allMesh[i][j].state == 1) {

                pcl::PointCloud<PointType>::Ptr edgeCandidate(new pcl::PointCloud<PointType>);
                pcl::PointCloud<pcl::PointXYZ>::Ptr candidateXYZ(new pcl::PointCloud<pcl::PointXYZ>);
                for (std::vector<int>::iterator iter = allMesh[i][j].pointsIndex.begin(); iter != allMesh[i][j].pointsIndex.end(); iter++) {
                    edgeCandidate->push_back(inputCloud->points[*iter]); // push in candidate for cylinder ransac
                }

                pcl::copyPointCloud(*edgeCandidate, *candidateXYZ);

                pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
                pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
                pcl::ExtractIndices<pcl::PointXYZ> extract;
                pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
                pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
                pcl::PointIndices::Ptr inlineCylinder(new pcl::PointIndices);
                pcl::ModelCoefficients::Ptr coefficientsCylinder(new pcl::ModelCoefficients);
                pcl::PCDWriter writer;
                ne.setSearchMethod(tree);
                ne.setInputCloud(candidateXYZ);
                ne.setKSearch(10);
                ne.compute(*cloud_normals);

                seg.setOptimizeCoefficients(true);
                seg.setModelType(pcl::SACMODEL_CYLINDER);
                seg.setMethodType(pcl::SAC_RANSAC);
                seg.setAxis({0,0,1});//z轴方向
                seg.setEpsAngle(0.08);//偏离角度（弧度制）
                seg.setMaxIterations(1000);//最大迭代次数
                seg.setNormalDistanceWeight(0.1);
                seg.setDistanceThreshold(0.2);//判断内外点的距离阈值
                seg.setRadiusLimits(0, 0.1);//cylinder max radius
                seg.setInputCloud(candidateXYZ);
                seg.setInputNormals(cloud_normals);
                seg.segment(*inlineCylinder, *coefficientsCylinder);

                extract.setInputCloud(candidateXYZ);
                extract.setIndices(inlineCylinder);
                extract.setNegative(false);
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCylinder(new pcl::PointCloud<pcl::PointXYZ>());
                extract.filter(*cloudCylinder);
//
//
                if (cloudCylinder->points.size() < 10){
                    continue;
                }
//                pcl::io::savePCDFileASCII(outPath / ("edge_candi_" + std::to_string(k) + ".pcd"), *cloudCylinder);
                pcl::copyPointCloud(*cloudCylinder, *edgeCandidate);
                *edgeCloud += *edgeCandidate;

            }

        }
    }
    cv::imwrite("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/edge/grid_edge_3600.png", edgeMat);
    pcl::io::savePCDFileASCII(outPath / "edge_cloud.pcd", *edgeCloud);

    std::cout << "*****************End processing***************" << std::endl;
    return 0;
}