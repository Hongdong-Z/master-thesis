//
// Created by zhao on 23.03.22.
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



    //////////////////////////////////////////////////////////////
    // Iterate data frames
    //////////////////////////////////////////////////////////////

    TicToc t_odo;

    for(size_t i = dataRange[0]; i <= dataRange[1]; i++) {
        //////////////////////////////////////////////////////////////
        // Extract features
        //////////////////////////////////////////////////////////////

        std::optional<fbds::FrameData> frameData = mainDataStorage[i];
        if(!frameData) { std::cerr << "\nFrameData or lastFrameData " << i << " and " << i+1 << " not found " << std::endl; continue; }

        std::cout << "\nProcessing Frame " << i << std::endl;

        if(!frameData->pose()) { std::cerr << "Did not find pose. ~> Skipping frame. " << std::endl; continue; }


        auto velodyneFLDatum = (*frameData)[fbds::FrameSource::VelodyneFL];
        //point cloud of current frame

        std::shared_ptr<pcl::PointCloud<PointType>> lastPC = std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatum->asPointcloud<PointType>());

        llo::ImageProjection IP;
        IP.run(*lastPC);
        pcl::PointCloud<PointType>::Ptr ground(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr curb(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surface(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr nonSurface(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr edge(new pcl::PointCloud<PointType>());
        ground = IP.getGroundFeature();
        curb = IP.getCurbFeature();
        surface = IP.getSurfaceFeature();
        nonSurface = IP.getNonSurfaceFeature();
        edge = IP.getEdgeFeature();

        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_extraction/debug/ground" + std::to_string(i)+ ".pcd", *ground);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_extraction/debug/curb" + std::to_string(i)+ ".pcd", *curb);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_extraction/debug/surface" + std::to_string(i)+ ".pcd", *surface);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_extraction/debug/nonSurface" + std::to_string(i)+ ".pcd", *nonSurface);
        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/feature_extraction/debug/edge" + std::to_string(i)+ ".pcd", *edge);

        std::cerr << "Frame: " << std::to_string(i) << " Finish " << std::endl;
        std::cout << std::endl;
    }

//    DrawTrajectory(poses);
    std::cerr << " Finish Processing" << std::endl;
    std::cerr << "TIME CONSUMPTION: " << t_odo.toc() << std::endl;

    return 0;
}// End main




