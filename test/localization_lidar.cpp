// google test docs
// wiki page: https://code.google.com/p/googletest/w/list
// primer: https://code.google.com/p/googletest/wiki/V1_7_Primer
// FAQ: https://code.google.com/p/googletest/wiki/FAQ
// advanced guide: https://code.google.com/p/googletest/wiki/V1_7_AdvancedGuide
// samples: https://code.google.com/p/googletest/wiki/V1_7_Samples
//
#include "gtest/gtest.h"
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
#include "map_matching/map_matching.h"
using json = nlohmann::json;
using Config = localization_lidar::ParameterServer;


int main() {

    auto configPath = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::cout << "Loading config file from " << configPath << std::endl;
    Config::initialize(configPath);

    std::filesystem::path dataRootPath(Config::get()["main"]["dataRoot"]);

    std::filesystem::path outputPath(localization_lidar::ParameterServer::get()["main"]["outputFolderMap"]);
    std::filesystem::path groundMapPath = outputPath / "groundMap";
    std::filesystem::path curbMapPath = outputPath / "curbMap";
    std::filesystem::path surfaceMapPath = outputPath / "surfaceMap";
    std::filesystem::path edgeMapPath = outputPath / "edgeMap";

    auto mainDataStorage = loadDataStorage(dataRootPath);

    uint64_t dataRange[2] = {Config::get()["main"]["startDataRange"], Config::get()["main"]["stopDataRange"]};


    std::vector<size_t> validDataIndices; // Only frames with both point cloud is used



    std::ofstream poseFile("/home/zhao/zhd_ws/src/localization_lidar/pose/test_new_opt.txt");


    poseFile << 1 << " " << 0 << " " << 0 << " " << 0 << " "
             << 0 << " " << 1 << " " << 0 << " " << 0 << " "
             << 0 << " " << 0 << " " << 1 << " " << 0 << std::endl;



    Eigen::Quaterniond q_w(1,0,0,0);
    Eigen::Vector3d t_w(0,0,0);

    Eigen::Quaterniond q_last_curr(1,0,0,0);
    Eigen::Vector3d t_last_curr(0,0,0);

    Eigen::Quaterniond q_w_m = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_w_m(0, 0, 0);

    pcl::PointCloud<PointType>::Ptr groundMap(new pcl::PointCloud<PointType>());

    pcl::PointCloud<PointType>::Ptr surfaceMap(new pcl::PointCloud<PointType>());

    pcl::PointCloud<PointType>::Ptr edgeMap(new pcl::PointCloud<PointType>());

    pcl::PointCloud<PointType>::Ptr curbMap(new pcl::PointCloud<PointType>());


    std::optional<fbds::FrameData> temFrame = mainDataStorage[1];
    TicToc t_loc;
    // start frame i = 1
    for(size_t i = dataRange[0]; i <= dataRange[1]; i++) {

        std::optional<fbds::FrameData> frameDataLast = mainDataStorage[i-1];
        std::optional<fbds::FrameData> frameDataCurrent = mainDataStorage[i];

        if (!frameDataCurrent || !frameDataLast) {
            std::cerr << "\nBothFrame" << i-1 << "and " << i << " not found " << std::endl;
            continue;
        }
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

        std::cout << "\nProcessing Frame " << i << std::endl;

        validDataIndices.emplace_back(i);

        //point cloud of current frame
        pcl::PointCloud<PointType>::Ptr lastPC(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr currPC(new pcl::PointCloud<PointType>());

        auto velodyneFLDatumLast = (*frameDataLast)[fbds::FrameSource::VelodyneFL];
        auto velodyneFLDatumCurrent = (*frameDataCurrent)[fbds::FrameSource::VelodyneFL];

//        std::cout << velodyneFLDatumLast.value() << std::endl;
//        std::cout << velodyneFLDatumCurrent.value() << std::endl;

        // input cloud
        *lastPC = velodyneFLDatumLast->asPointcloud<PointType>();
        *currPC = velodyneFLDatumCurrent->asPointcloud<PointType>();

        pcl::PointCloud<PointType>::Ptr groundCurrent(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCurrent(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr edgeCurrent(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr curbCurrent(new pcl::PointCloud<PointType>());



        ////***************************FRAME TO FRAME ODOMETRY**************************
        TicToc t_frame;
        odo::FeatureAssociation FA(lastPC, currPC, q_w, t_w, q_last_curr, t_last_curr);

        FA.run();
        q_w = FA.publishQuaterniond();
        t_w = FA.publishTranslation();
        q_last_curr = FA.quaterniodGuess();
        t_last_curr = FA.translationGuess();


        groundCurrent = FA.getGroundFeature();
        surfaceCurrent = FA.getSurfaceFeature();
        curbCurrent = FA.getCurbFeature();
        edgeCurrent = FA.getEdgeFeature();



        ////********************************MAP MATCHING*********************************
        if ( (i % 50 == 0) || i == 1) {
            std::cerr << "change map set\n" << std::endl;
            int startID1, endID1;


            startID1 = 50 * ((i+1) / 50);
            endID1 = 50 * ((i+1) / 50 + 1);
            pcl::io::loadPCDFile(groundMapPath / ("ground_" + std::to_string(startID1) + "_" + std::to_string(endID1) + ".pcd"), *groundMap);
            pcl::io::loadPCDFile(surfaceMapPath / ("surface_" + std::to_string(startID1) + "_" + std::to_string(endID1) + ".pcd"), *surfaceMap);
            pcl::io::loadPCDFile(edgeMapPath / ("edge_" + std::to_string(startID1) + "_" + std::to_string(endID1) + ".pcd"), *edgeMap);
            pcl::io::loadPCDFile(curbMapPath / ("curb_" + std::to_string(startID1) + "_" + std::to_string(endID1) + ".pcd"), *curbMap);
            std::cerr << "load map: " << std::to_string(startID1) << "_" << std::to_string(endID1) << "\n" << std::endl;
        }
        // for map loader

        mm::MapMatching MM(groundCurrent, curbCurrent, surfaceCurrent, edgeCurrent,
                           groundMap, curbMap, surfaceMap, edgeMap,
                           q_w, t_w,
                           q_w_m, t_w_m);
        MM.run();
//        if ( (i) % 20 == 0) {
//            pcl::PointCloud<PointType>::Ptr subGroundMap(new pcl::PointCloud<PointType>());
//            pcl::PointCloud<PointType>::Ptr subSurfaceMap(new pcl::PointCloud<PointType>());
//            pcl::PointCloud<PointType>::Ptr subEdgeMap(new pcl::PointCloud<PointType>());
//            pcl::PointCloud<PointType>::Ptr subCurbMap(new pcl::PointCloud<PointType>());
//
//            subGroundMap = MM.getSubMapGround();
//            subSurfaceMap = MM.getSubMapSurface();
//            subEdgeMap = MM.getSubMapEdge();
//            subCurbMap = MM.getSubMapCurb();
//
//
//            pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/mapMatching/subMap/" + std::to_string(i)+ "_" + "ground_map.pcd",
//                                      *subGroundMap);
//            pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/mapMatching/subMap/" + std::to_string(i)+ "_" + "curb_map.pcd",
//                                      *subCurbMap);
//            pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/mapMatching/subMap/" + std::to_string(i)+ "_" + "surface_map.pcd",
//                                      *subSurfaceMap);
//            pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/mapMatching/subMap/" + std::to_string(i)+ "_" + "edge_map.pcd", *subEdgeMap);
////
//            pcl::PointCloud<PointType>::Ptr testCloudGround(new pcl::PointCloud<PointType>());
//            pcl::PointCloud<PointType>::Ptr testCloudCurb(new pcl::PointCloud<PointType>());
//            pcl::PointCloud<PointType>::Ptr testCloudEdge(new pcl::PointCloud<PointType>());
//            pcl::PointCloud<PointType>::Ptr testCloudSurface(new pcl::PointCloud<PointType>());
//
//            testCloudGround = MM.getAssocGround();
//            testCloudEdge = MM.getAssocEdge();
//            testCloudSurface = MM.getAssocSurface();
//            testCloudCurb = MM.getAssocCurb();
//
//            pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/mapMatching/FA/" + std::to_string(i)+ "_" + "ground_fa.pcd",
//                                      *testCloudGround);
//            pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/mapMatching/FA/" + std::to_string(i)+ "_" + "curb_fa.pcd",
//                                      *testCloudCurb);
//            pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/mapMatching/FA/" + std::to_string(i)+ "_" + "surface_fa.pcd",
//                                      *testCloudSurface);
//            pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/output/mapMatching/FA/" + std::to_string(i)+ "_" + "edge_fa.pcd",
//                                      *testCloudEdge);
//        }

        q_w = MM.publishQuaterniond();
        t_w = MM.publishTranslation();

        q_w_m = MM.getQuaterniondGuess();
        t_w_m = MM.getTranslationGuess();

        std::cerr << "Frames transform time consumption: " << t_frame.toc() << std::endl;
        Eigen::Isometry3d T = Eigen::Isometry3d ::Identity();
        T.rotate(q_w.matrix());
        T.pretranslate(t_w);
        poseFile << T.matrix()(0, 0) << " " << T.matrix()(0, 1) << " " << T.matrix()(0, 2) << " " << T.matrix()(0, 3)
                 << " " << T.matrix()(1, 0) << " " << T.matrix()(1, 1) << " " << T.matrix()(1, 2) << " " << T.matrix()(1, 3) << " "
                 << T.matrix()(2, 0) << " " << T.matrix()(2, 1) << " " << T.matrix()(2, 2) << " " << T.matrix()(2, 3) << std::endl;


        std::cerr << "pose of frame after fine tuning: \n " << T.matrix() << std::endl;

        std::cout << "Frame: " << std::to_string(i) << " Finish " << std::endl;

        std::cout << "*******************************************************************************" << std::endl;
    }


    std::cerr << " Finish Localization" << std::endl;
    std::cerr << " whole time consumption: " << t_loc.toc() << std::endl;

    return 0;
}