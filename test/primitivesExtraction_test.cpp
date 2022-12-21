//
// Created by zhao on 24.06.22.
//
#include <primitives_extraction/primitivesExtraction.h>
#include <fstream>
#include <iostream>
#include "primitives_mapping/primitivesMapping.h"
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"
#include "localization_lidar/load_datastorage.hpp"
#include "localization_lidar/parameter_server.hpp"
#include "localization_lidar/utility/tic_toc.h"
using json = nlohmann::json;
using Config = localization_lidar::ParameterServer;
using Point = pcl::PointXYZI;
using Cloud = pcl::PointCloud<Point>;

using PointColor = pcl::PointXYZRGB;
using CloudColor = pcl::PointCloud<PointColor>;
int main() {

    std::cout << "*************************start processing*************************" << std::endl;

    auto configPath = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::cout << "Loading config file from " << configPath << std::endl;
    Config::initialize(configPath);

    std::filesystem::path dataRootPath(Config::get()["main"]["dataRoot"]);
    auto mainDataStorage = loadDataStorage(dataRootPath);
    uint64_t dataRange[2] = {Config::get()["main"]["startDataRange"], Config::get()["main"]["stopDataRange"]};




    std::optional<fbds::FrameData> temFrame = mainDataStorage[1];


    // loop to run
    for(size_t i = dataRange[0]; i < dataRange[1]; i++) {

        std::optional<fbds::FrameData> frameDataLast = mainDataStorage[i - 1];
        std::optional<fbds::FrameData> frameDataCurrent = mainDataStorage[i];
        if (!frameDataLast || !frameDataCurrent) {
            std::cerr << "\nFrameData or lastFrameData " << i << " and " << i + 1 << " not found " << std::endl;
            continue;
        }

        std::cout << "\nProcessing Frame " << i << std::endl;

        if (!frameDataCurrent->pose() || !frameDataLast->pose()) {
            if (!frameDataCurrent->pose() && frameDataLast->pose()) {
                temFrame = frameDataLast;
                continue;
            } else if (!frameDataCurrent->pose() && !frameDataLast->pose()) {
                continue;
            } else if (frameDataCurrent->pose() && !frameDataLast->pose()) {
                frameDataLast = temFrame;
            }
        }

        auto velodyneFLDatumCurrent = (*frameDataCurrent)[fbds::FrameSource::VelodyneFL];

        std::shared_ptr<pcl::PointCloud<PointType>> currPC =
            std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatumCurrent->asPointcloud<PointType>());


        ////***************************FRAME TO FRAME ODOMETRY**************************
        TicToc t_frame;

        primitivesExtraction::PrimitiveExtractor PriExtractor;

        PriExtractor.setInputCloud(currPC->makeShared());
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
//        std::cout << "Poles number: " << poles.size() << std::endl;
//        std::cout << "Facades number: " << facades.size() << std::endl;
        std::cout << "Grounds number: " << grounds.size() << std::endl;
        primitivesMapping::PrimitivesMapping PM(50);
        Cloud::Ptr test(new Cloud());
//        CloudColor::Ptr testColor(new CloudColor());
//        std::vector<CloudColor::Ptr> grounds;
        PM.getPrimitivePoleCloud(test, poles);
        PM.getPrimitiveFacadeCloud(test, grounds, 3000);
        PM.getPrimitiveFacadeCloud(test, facades, 2000);
//        PriExtractor.getPoleRoughCloud(test);
//        PriExtractor.getFacadeRoughCloud(test);
//        PriExtractor.getGroundCloud(test);
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
        pcl::io::savePCDFile("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/poleEstimate/test/"
                                 + std::to_string(i) + "_small.pcd", *test);



        std::cerr << "Frame: " << std::to_string(i) << " Finish " << std::endl;

        std::cout << "********************************************************************" << std::endl;

    }

    std::cout << "*************************end processing*************************" << std::endl;

    return 0;

}