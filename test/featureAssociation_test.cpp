//
// Created by zhao on 17.05.22.
//

#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <iostream>
#include <typeinfo>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <seamseg_fbds/reader.h>
#include <calib_storage/calib_storage.h>
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"
#include "localization_lidar/load_datastorage.hpp"
#include "localization_lidar/parameter_server.hpp"

#include "feature_extraction/feature_extraction.h"
#include "feature_extraction/load_pose.h"
#include <feature_association/featureAssociation.h>
//#include <pangolin/pangolin.h>
//#include <Eigen/Core>
//#include <unistd.h>

using json = nlohmann::json;
using Config = localization_lidar::ParameterServer;
//using Config = ParameterServer;




int main(){
    //////////////////////////////////////////////////////////////
    // Initialization
    //////////////////////////////////////////////////////////////
    auto configPath = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::cout << "Loading config file from " << configPath << std::endl;
    Config::initialize(configPath);

    std::filesystem::path dataRootPath(Config::get()["main"]["dataRoot"]);

    auto mainDataStorage = loadDataStorage(dataRootPath);

    uint64_t dataRange[2] = {Config::get()["main"]["startDataRange"], Config::get()["main"]["stopDataRange"]};

    std::srand((int)time(nullptr)); // Seed random number generator for renderer colors

    std::vector<size_t> validDataIndices; // Only frames with both point cloud is used


    //////////////////////////////////////////////////////////////
    // Iterate data frames

    TicToc t_odo;
    for(size_t i = dataRange[0]; i < dataRange[1]; i++) {
        //////////////////////////////////////////////////////////////
        // Extract features
        //////////////////////////////////////////////////////////////

        std::optional<fbds::FrameData> frameData = mainDataStorage[i];
        std::optional<fbds::FrameData> frameData1 = mainDataStorage[i+1];
        if(!frameData || !frameData1) { std::cerr << "\nFrameData or lastFrameData " << i << " and " << i+1 << " not found " << std::endl; continue; }

        std::cout << "\nProcessing Frame " << i << std::endl;

        if(!frameData->pose() || !frameData1->pose()) { std::cerr << "Did not find pose. ~> Skipping frame. " << std::endl; continue; }

        validDataIndices.emplace_back(i);


        auto velodyneFLDatum = (*frameData)[fbds::FrameSource::VelodyneFL];
        auto velodyneFLDatum1 = (*frameData1)[fbds::FrameSource::VelodyneFL];
        //point cloud of current frame

        std::shared_ptr<pcl::PointCloud<PointType>> lastPC = std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatum->asPointcloud<PointType>());
        std::shared_ptr<pcl::PointCloud<PointType>> currPC = std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatum1->asPointcloud<PointType>());



        odo::FeatureAssociation FA(lastPC->makeShared(), currPC->makeShared());
        FA.run();
        pcl::PointCloud<PointType>::Ptr pcGround(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr pcCurb(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr pcSurface(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr pcEdge(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr pcTwoFrame(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr lastGround(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr currentGround(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr currentEdge(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr lastEdge(new pcl::PointCloud<PointType>);
        pcGround = FA.getFeatureAssociationGround();
        pcCurb = FA.getFeatureAssociationCurb();
        pcSurface = FA.getFeatureAssociationSurface();
        pcEdge = FA.getFeatureAssociationEdge();
        pcTwoFrame = FA.getFusionFrame();
        lastGround = FA.getGroundFeatureLast();
        currentGround = FA.getGroundFeature();
        lastEdge = FA.getEdgeFeatureLast();
        currentEdge = FA.getEdgeFeature();
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/" + std::to_string(i)+ "ground.pcd", *pcGround);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/" + std::to_string(i)+ "curb.pcd", *pcCurb);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/" + std::to_string(i)+ "surface.pcd", *pcSurface);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/" + std::to_string(i)+ "edge.pcd", *pcEdge);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/" + std::to_string(i)+ "fusion.pcd", *pcTwoFrame);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/" + std::to_string(i)+ "lastGround.pcd", *lastGround);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/" + std::to_string(i)+ "currentGround.pcd", *currentGround);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/" + std::to_string(i)+ "lastEdge.pcd", *lastEdge);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_association/debug/" + std::to_string(i)+ "currentEdge.pcd", *currentEdge);

        std::cerr << "Frame: " << std::to_string(i) << " Finish " << std::endl;
        std::cout << std::endl;

    }

//    DrawTrajectory(poses);
    std::cerr << " Finish Odometry" << std::endl;
    std::cerr << " whole time consumption: " << t_odo.toc() << std::endl;
    return 0;
}// End main



