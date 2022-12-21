//
// Created by zhao on 01.09.22.
//


#include <primitives_extraction/primitivesExtraction.h>
#include <fstream>
#include <iostream>
#include "primitives_mapping/primitivesMapping.h"
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "utility/kittiLoader.h"
#include "feature_extraction/load_pose.h"
#include "localization_lidar/utility/tic_toc.h"
using Point = pcl::PointXYZI;
using Cloud = pcl::PointCloud<Point>;
using PointColor = pcl::PointXYZRGB;
using CloudColor = pcl::PointCloud<PointColor>;
int main() {
    KittiLoader kitti("/mrtstorage/datasets/public/kitti/kitti-odometry/dataset/sequences/00/velodyne");
    // load new pose file
    std::string poseConfig = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::shared_ptr<localization_lidar::PoseLoader> poseLoader = std::make_shared<localization_lidar::PoseLoader>(poseConfig);
    poseLoader->loadNewPoseFile();

    for (int i = 0; i < 1; ++i) {
//        auto pose = poseLoader->getPose(i);
        auto cloudCurrent = kitti.cloud(i);

        primitivesExtraction::PrimitiveExtractor PriExtractor;

        PriExtractor.setInputCloud(cloudCurrent);
        PriExtractor.setRange(-200, 200,
                              -200, 200,
                              -2.5, 10);
//    PriExtractor.run();

//    PriExtractor.testGetGround();
        PriExtractor.testNewPoleExtractor();
        std::vector<primitivesExtraction::Cylinder_Fin_Param> poles;
        std::vector<primitivesExtraction::Plane_Param> facades;
        std::vector<primitivesExtraction::Plane_Param> grounds;
        PriExtractor.getPolesFinParam(poles);
        PriExtractor.getFacadesFinParam(facades);
        PriExtractor.getGroundFinParam(grounds);

        std::cout << "Poles number: " << poles.size() << std::endl;
        std::cout << "Facades number: " << facades.size() << std::endl;
        std::cout << "Grounds number: " << grounds.size() << std::endl;
        primitivesMapping::PrimitivesMapping PM(50);
        Cloud::Ptr test(new Cloud());
//        CloudColor::Ptr testColor(new CloudColor());
//        std::vector<CloudColor::Ptr> grounds;
        PM.getPrimitivePoleCloud(test, poles);
        PM.getPrimitiveFacadeCloud(test, grounds, 3000);
        PM.getPrimitiveFacadeCloud(test, facades, 2000);
        PriExtractor.getPoleRoughCloud(test);
        PriExtractor.getFacadeRoughCloud(test);
        PriExtractor.getGroundCloud(test);
//        PriExtractor.getNonFacadeRoughCloud(test);
//        PriExtractor.get3DGroundCandidateCloudsColored(grounds);
//        for (int j = 0; j < grounds.size(); ++j) {
//            std::cout << "ground points: " << grounds[j]->points.size() << std::endl;
//            for (auto& p : grounds[j]->points) {
//                testColor->points.push_back(p);
//            }
//        }
//        testColor->width = 1;
//        testColor->height = testColor->points.size();
        pcl::io::savePCDFile("/home/zhao/zhd_ws/src/localization_lidar/kitti/FE/"
                             + std::to_string(i) + "_.pcd", *test);



        std::cerr << "Frame: " << std::to_string(i) << " Finish " << std::endl;

        std::cout << "********************************************************************" << std::endl;
    }
    return 0;
}