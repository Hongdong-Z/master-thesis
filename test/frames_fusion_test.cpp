//
// Created by zhao on 08.03.22.
//
//test function of feature extraction and feature map building
// feature extraction: extraction four features from point cloud :
// ground, surface, edge, curb.
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

using json = nlohmann::json;
using Config = localization_lidar::ParameterServer;
//using Config = ParameterServer;


/**
 * The "main" method is structured loosely as follows:
 * 1. Initialization: Load config, FBDS initialization, load calibration
 * 2. Iterate data frames:
 *      1. Extract features
 *      2. mapping with extracted features
 * 3. store the processed maps
 */

int main(){
    //////////////////////////////////////////////////////////////
    // Initialization
    //////////////////////////////////////////////////////////////
    auto configPath = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::cout << "Loading config file from " << configPath << std::endl;
    Config::initialize(configPath);

    std::filesystem::path dataRootPath(Config::get()["main"]["dataRoot"]);
    std::filesystem::path outputPath(Config::get()["main"]["outputFolder"]);
    std::filesystem::create_directory(outputPath);

    std::filesystem::path outputFramePathBase = outputPath / "per_frame_feature";
    std::filesystem::create_directory(outputFramePathBase);
    std::filesystem::path mapPath = outputPath / "maps";
    std::filesystem::create_directory(mapPath);

    auto mainDataStorage = loadDataStorage(dataRootPath);
    auto calibrationDatum = mainDataStorage.getStaticDatum("calibration");
    if(!calibrationDatum) {throw std::runtime_error("No calibration file found. Quitting..."); }

    cs::CalibStorage calibStorage(calibrationDatum->path());
    auto tf_vehicle_lidar = calibStorage.getTransformation("vehicle", "sensor/lidar/velodyne/c").src2dest; // tf from vehicle to lidar

    //    std::cout << "\nBebug: name of tf_lidar_vehicle: " << typeid(tf_lidar_vehicle.matrix()).name() << std::endl;
    std::cout << "\nLidar coordinate system to vehicle coordinate system transform: " << "\n" << tf_vehicle_lidar.matrix() << std::endl;

    uint64_t dataRange[2] = {Config::get()["main"]["startDataRange"], Config::get()["main"]["stopDataRange"]};

    std::srand((int)time(nullptr)); // Seed random number generator for renderer colors

    std::vector<size_t> validDataIndices; // Only frames with both point cloud is used

    pcl::PointCloud<pcl::PointXYZI> outputPCMap; // output point cloud map after feature extraction
    std::shared_ptr<pcl::PointCloud<PointType>> groundMap = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> curbMap = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> surfaceMap = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> edgeMap = std::make_shared<pcl::PointCloud<PointType>>();

    // load new pose file
//    std::string posePath = "/home/zhao/zhd_ws/src/localization_lidar/input/odom_os_opt_final.txt";
    std::string poseConfig = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::shared_ptr<localization_lidar::PoseLoader> poseLoader = std::make_shared<localization_lidar::PoseLoader>(poseConfig);
//    std::cout << "++++++++++" << std::endl;
    poseLoader->loadNewPoseFile();

    //////////////////////////////////////////////////////////////
    // Iterate data frames
    //////////////////////////////////////////////////////////////
    for(size_t i = dataRange[1]-1; i >= dataRange[0]; i--) {
        //////////////////////////////////////////////////////////////
        // Extract features
        //////////////////////////////////////////////////////////////
        std::optional<fbds::FrameData> firstFrameData = mainDataStorage[dataRange[1]];

        std::optional<fbds::FrameData> frameData = mainDataStorage[i];
        if(!frameData) { std::cerr << "\nFrameData or lastFrameData " << i << " not found " << std::endl; continue; }

        std::cout << "\nProcessing Frame " << i << std::endl;

        if(!frameData->pose()) { std::cerr << "Did not find pose. ~> Skipping frame. " << std::endl; continue; }

        validDataIndices.emplace_back(i);

//        auto poseVehicleCurrentFrame = firstFrameData->pose()->pose;
//        auto poseVehicleLastFrame = frameData->pose()->pose;
        auto poseVehicleCurrentFrame = poseLoader->getPose(dataRange[1]);
        auto poseVehicleLastFrame = poseLoader->getPose(i);

        // Transform between two frames
        Eigen::Transform<double, 3, Eigen::Affine> transformTwoFrame = poseVehicleCurrentFrame.inverse() * poseVehicleLastFrame;

        auto velodyneFLDatum = (*frameData)[fbds::FrameSource::VelodyneFL];
//        auto velodyneFLDatum = (*frameData).operator[](fbds::FrameSource::VelodyneFL);
        //point cloud of current frame
        auto thisPointCloud = velodyneFLDatum->asPointcloud<PointType>();

        std::shared_ptr<pcl::PointCloud<PointType>> inputCloud = std::make_shared<pcl::PointCloud<PointType>>();
//        pcl::PointCloud<PointType> inputPC; // input point cloud for feature extraction function
        *inputCloud = thisPointCloud;

        std::shared_ptr<pcl::PointCloud<PointType>> groundCloud = std::make_shared<pcl::PointCloud<PointType>>();
        std::shared_ptr<pcl::PointCloud<PointType>> curbCloud = std::make_shared<pcl::PointCloud<PointType>>();
        std::shared_ptr<pcl::PointCloud<PointType>> surfaceCloud = std::make_shared<pcl::PointCloud<PointType>>();
        std::shared_ptr<pcl::PointCloud<PointType>> edgeCloud = std::make_shared<pcl::PointCloud<PointType>>();

        // Extraction of features in current frame
        llo::FeaturesExtraction fe(inputCloud);
        *groundCloud = fe.getGround();
        *curbCloud = fe.getCurb();
        *surfaceCloud = fe.getSurface();
        *edgeCloud = fe.getEdge();

        //first frame point cloud, which other frames based on
        auto firstDatum = (*firstFrameData)[fbds::FrameSource::VelodyneFL];
        pcl::PointCloud<PointType> firstPointCloud = firstDatum->asPointcloud<PointType>();

        // first frame features extraction
        std::shared_ptr<pcl::PointCloud<PointType>> firstCloud = std::make_shared<pcl::PointCloud<PointType>>();
        *firstCloud = firstPointCloud;

        llo::FeaturesExtraction firstFe(firstCloud);

        std::shared_ptr<pcl::PointCloud<PointType>> firstGroundCloud = std::make_shared<pcl::PointCloud<PointType>>();
        std::shared_ptr<pcl::PointCloud<PointType>> firstCurbCloud = std::make_shared<pcl::PointCloud<PointType>>();
        std::shared_ptr<pcl::PointCloud<PointType>> firstSurfaceCloud = std::make_shared<pcl::PointCloud<PointType>>();
        std::shared_ptr<pcl::PointCloud<PointType>> firstEdgeCloud = std::make_shared<pcl::PointCloud<PointType>>();

        *firstGroundCloud = firstFe.getGround();
        *firstCurbCloud = firstFe.getCurb();
        *firstSurfaceCloud = firstFe.getSurface();
        *firstEdgeCloud = firstFe.getEdge();


        pcl::transformPointCloud(*groundCloud, *firstGroundCloud, transformTwoFrame);
        pcl::transformPointCloud(*curbCloud, *firstCurbCloud, transformTwoFrame);
        pcl::transformPointCloud(*surfaceCloud, *firstSurfaceCloud, transformTwoFrame);
        pcl::transformPointCloud(*edgeCloud, *firstEdgeCloud, transformTwoFrame);

        pcl::transformPointCloud(*inputCloud, *firstCloud, transformTwoFrame);

        *groundMap += *firstGroundCloud;
        *curbMap += *firstCurbCloud;
        *surfaceMap += *firstSurfaceCloud;
        *edgeMap += *firstEdgeCloud;
        outputPCMap += *firstCloud;


//        std::filesystem::path perFrameFeaturePath = outputFramePathBase / std::to_string(i);
//        std::filesystem::create_directory(perFrameFeaturePath);
//        pcl::io::savePCDFileASCII(perFrameFeaturePath / ("GroundFeature_" + std::to_string(i)+ ".pcd"), *firstGroundCloud);
//        pcl::io::savePCDFileASCII(perFrameFeaturePath / ("CurbFeature_" + std::to_string(i)+ ".pcd"), *firstCurbCloud);
//        pcl::io::savePCDFileASCII(perFrameFeaturePath / ("SurfaceFeature_" + std::to_string(i)+ ".pcd"), *firstSurfaceCloud);
//        pcl::io::savePCDFileASCII(perFrameFeaturePath / ("EdgeFeature_" + std::to_string(i)+ ".pcd"), *firstEdgeCloud);

        std::cout << "Frame: " << std::to_string(i) << " Finish " << std::endl;
        std::cout << std::endl;
    }
//    pcl::io::savePCDFileASCII(mapPath / "integrated_ground_map.pcd", *groundMap);
//    pcl::io::savePCDFileASCII(mapPath / "integrated_curb_map.pcd", *curbMap);
//    pcl::io::savePCDFileASCII(mapPath / "integrated_surface_map.pcd", *surfaceMap);
//    pcl::io::savePCDFileASCII(mapPath / "integrated_edge_map.pcd", *edgeMap);
    pcl::io::savePCDFileASCII(mapPath / "integrated_100_101.pcd", outputPCMap);

    std::cerr << " Finish Mapping" << std::endl;
    return 0;
}// End main