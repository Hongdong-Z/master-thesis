//
// Created by zhao on 21.08.22.
//
#include "primitives_extraction/primitivesExtraction.h"
#include "primitives_association/primitives_association.h"
#include "primitives_association/primitives_map_matcher.h"

#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"
#include "localization_lidar/load_datastorage.hpp"
#include "localization_lidar/parameter_server.hpp"
#include "localization_lidar/utility/tic_toc.h"
using json = nlohmann::json;
using Config = localization_lidar::ParameterServer;
using Primitives = primitivesMapMatcher::Primitives;

int main() {

    std::cout << "*************************start processing*************************" << std::endl;

    auto configPath = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::cout << "Loading config file from " << configPath << std::endl;
    Config::initialize(configPath);

    std::filesystem::path dataRootPath(Config::get()["main"]["dataRoot"]);
    auto mainDataStorage = loadDataStorage(dataRootPath);
    uint64_t dataRange[2] = {Config::get()["main"]["startDataRange"], Config::get()["main"]["stopDataRange"]};


    std::ofstream poseFile("/home/zhao/zhd_ws/src/localization_lidar/pose_primitives/test_time.txt");



    poseFile << 1 << " " << 0 << " " << 0 << " " << 0 << " "
             << 0 << " " << 1 << " " << 0 << " " << 0 << " "
             << 0 << " " << 0 << " " << 1 << " " << 0 << std::endl;



    // true pose
    Eigen::Quaterniond q_w(1,0,0,0);
    Eigen::Vector3d t_w(0,0,0);

    // transform from scan to scan
    Eigen::Quaterniond q_last_curr(1,0,0,0);
    Eigen::Vector3d t_last_curr(0,0,0);

    // transform between odometry and world frame
    Eigen::Quaterniond q_w_m = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_w_m(0, 0, 0);

    std::optional<fbds::FrameData> temFrame = mainDataStorage[1];


    // loop to run
    pcl::PointCloud<PointType>::Ptr showCloud(new pcl::PointCloud<PointType>());
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

        auto velodyneFLDatumLast = (*frameDataLast)[fbds::FrameSource::VelodyneFL];
        auto velodyneFLDatumCurrent = (*frameDataCurrent)[fbds::FrameSource::VelodyneFL];

        // point cloud of current frame and last frame
        std::shared_ptr<pcl::PointCloud<PointType>> lastPC =
            std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatumLast->asPointcloud<PointType>());
        std::shared_ptr<pcl::PointCloud<PointType>> currPC =
            std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatumCurrent->asPointcloud<PointType>());


        ////***************************FRAME TO FRAME ODOMETRY**************************
//        TicToc t_frame;
        // from odomtry get the transform between two frame and the primitives features for current frame
        TicToc t_od;
        primitivesFa::PrimitivesAssociation PA(lastPC->makeShared(), currPC->makeShared(),
                                               q_w, t_w,
                                               q_last_curr, t_last_curr);
        TicToc t_fa;
        PA.runAS();
        std::cerr << "feature association time: " << t_fa.toc() << "ms" << std::endl;
        std::cerr << "odometry time: " << t_od.toc() << "ms" << std::endl;
        q_w = PA.publishQuaterniond();
        t_w = PA.publishTranslation();
        q_last_curr = PA.quaterniodGuess();
        t_last_curr = PA.translationGuess();
        Eigen::Isometry3d T1 = Eigen::Isometry3d ::Identity();
        T1.rotate(q_w.matrix());
        T1.pretranslate(t_w);
        std::cout << "pose of frame before fine tuning: \n " << T1.matrix() << std::endl;




        std::vector<primitivesFa::Plane> detected_planes;
        std::vector<primitivesFa::Plane> detected_ground;
        std::vector<primitivesFa::Cylinder> detected_cylinders;
        // get current detections
        PA.getCurrentDetection(detected_cylinders, detected_planes, detected_ground);
//        primitivesFa::Cloud::Ptr cloud1(new primitivesFa::Cloud());
//        PA.getAssociation(cloud1);
//        pcl::io::savePCDFile("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/mapMatcher/"
//                             + std::to_string(i) + "_PA.pcd", *cloud1);

        ////********************************MAP MATCHING*********************************

        primitivesMapMatcher::PrimitiveMatcher PM;
        // set transform from odo
        PM.set_pose_prior(t_w, q_w, t_w_m, q_w_m);
        // set detections
        PM.set_detections(detected_planes, detected_cylinders, detected_ground);

        PM.match();


        // get optimized pose
        q_w = PM.getOptimizedQuaterniond();
        t_w = PM.getOptimizedTranslation();
        // update guess
        q_w_m = PM.getQuaterniondGuess();
        t_w_m = PM.getTranslationGuess();

        // visualization
//        pcl::PointCloud<PointType>::Ptr primitives(new pcl::PointCloud<PointType>());
//        PM.getMapPrimitives(primitives, rand()%2000);
//        PointType p;
//        p.x = t_w[0];
//        p.y = t_w[1];
//        p.z = t_w[2];
//        p.intensity = 500;
//        primitives->points.push_back(p);
//        *showCloud += *primitives;
//        primitives->width = 1;
//        primitives->height = primitives->points.size();
//        pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/visualization/"
//                                      + std::to_string(i)+ ".pcd", *primitives);

        //       std::cout << "Frames transform time consumption: " << t_frame.toc() << std::endl;
        Eigen::Isometry3d T = Eigen::Isometry3d ::Identity();
        T.rotate(q_w.matrix());
        T.pretranslate(t_w);
        std::cout << "pose of frame after fine tuning: \n " << T.matrix() << std::endl;

        poseFile << T.matrix()(0, 0) << " " << T.matrix()(0, 1) << " " << T.matrix()(0, 2) << " " << T.matrix()(0, 3)
                 << " " << T.matrix()(1, 0) << " " << T.matrix()(1, 1) << " " << T.matrix()(1, 2) << " " << T.matrix()(1, 3) << " "
                 << T.matrix()(2, 0) << " " << T.matrix()(2, 1) << " " << T.matrix()(2, 2) << " " << T.matrix()(2, 3) << std::endl;



        std::cerr << "Frame: " << std::to_string(i) << " Finish " << std::endl;

        std::cout << "********************************************************************" << std::endl;



    }
//    pcl::io::savePCDFileASCII("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/visualization/total.pcd", *showCloud);

    std::cout << "*************************end processing*************************" << std::endl;

    return 0;
}