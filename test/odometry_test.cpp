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
    //////////////////////////////////////////////////////////////
    std::ofstream poseFile("/home/zhao/zhd_ws/src/localization_lidar/pose_odo.txt");


    poseFile << 1 << " " << 0 << " " << 0 << " " << 0
            << " " << 0 << " " << 1 << " " << 0 << " "
            << 0 << " " << 0 << " " << 0 << " " << 1
            << " " << 0 << std::endl;

    Eigen::Quaterniond qCurr(1,0,0,0);
    Eigen::Vector3d tCurr(0,0,0);

    Eigen::Quaterniond q_last_curr(1,0,0,0);
    Eigen::Vector3d t_last_curr(0,0,0);


    std::optional<fbds::FrameData> temFrame = mainDataStorage[1];
    TicToc t_odo;
    for(size_t i = dataRange[0]; i < dataRange[1]; i++) {
        //////////////////////////////////////////////////////////////
        // Extract features
        //////////////////////////////////////////////////////////////

        std::optional<fbds::FrameData> frameDataLast = mainDataStorage[i-1];
        std::optional<fbds::FrameData> frameDataCurrent = mainDataStorage[i];
        if(!frameDataLast || !frameDataCurrent) { std::cerr << "\nFrameData or lastFrameData " << i << " and " << i+1 << " not found " << std::endl; continue; }

        std::cout << "\nProcessing Frame " << i << std::endl;

        if(!frameDataCurrent->pose() || !frameDataLast->pose()) {
            if (!frameDataCurrent->pose() && frameDataLast->pose()) {
                temFrame = frameDataLast;
                continue;
            } else if (!frameDataCurrent->pose() && !frameDataLast->pose()) {
                continue;
            } else if (frameDataCurrent->pose() && !frameDataLast->pose()) {
                frameDataLast = temFrame;
            }
        }

        validDataIndices.emplace_back(i);


        auto velodyneFLDatumLast = (*frameDataLast)[fbds::FrameSource::VelodyneFL];
        auto velodyneFLDatumCurrent = (*frameDataCurrent)[fbds::FrameSource::VelodyneFL];
        //point cloud of current frame

        std::shared_ptr<pcl::PointCloud<PointType>> lastPC = std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatumLast->asPointcloud<PointType>());
        std::shared_ptr<pcl::PointCloud<PointType>> currPC = std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatumCurrent->asPointcloud<PointType>());


        // start frame special process

        TicToc t_frame;
        odo::FeatureAssociation FA(lastPC->makeShared(), currPC->makeShared(), qCurr, tCurr, q_last_curr, t_last_curr);
        FA.run();
        qCurr = FA.publishQuaterniond();
        tCurr = FA.publishTranslation();
        q_last_curr = FA.quaterniodGuess();
        t_last_curr = FA.translationGuess();
        std::cerr << "Frames odometry time consumption: " << t_frame.toc() << std::endl;
        Eigen::Isometry3d T = Eigen::Isometry3d ::Identity();
        T.rotate(qCurr.matrix());
        T.pretranslate(tCurr);
        poseFile << T.matrix()(0, 0) << " " << T.matrix()(0, 1) << " " << T.matrix()(0, 2) << " " << T.matrix()(0, 3)
                 << " " << T.matrix()(1, 0) << " " << T.matrix()(1, 1) << " " << T.matrix()(1, 2) << " " << T.matrix()(1, 3) << " "
                 << T.matrix()(2, 0) << " " << T.matrix()(2, 1) << " " << T.matrix()(2, 2) << " " << T.matrix()(2, 3) << std::endl;

//        poses.push_back(T);

        std::cerr << "Frame: " << std::to_string(i) << " Finish " << std::endl;

        std::cout << "********************************************************************" << std::endl;
    }

//    DrawTrajectory(poses);
    std::cerr << " Finish Odometry" << std::endl;
    std::cerr << " whole time consumption: " << t_odo.toc() << std::endl;
    return 0;
}// End main



