//
// Created by jli on 02.03.22.
//

#ifndef SRC_LOAD_POSE_H
#define SRC_LOAD_POSE_H
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
//#include "preprocessing/config_load_server.h"
//#include "preprocessing/data_loader.h"
#include "localization_lidar/load_datastorage.hpp"
#include "localization_lidar/parameter_server.hpp"

namespace localization_lidar{

class PoseLoader {
public:
    PoseLoader(std::string& configPath) : configPath_(configPath){
        ParameterServer::initialize(configPath_);
        auto inDataRootPath = static_cast<std::filesystem::path>(ParameterServer::get()["pose"]["dataRoot"]);
        mainDataStorage_ = loadDataStorage(inDataRootPath); // include all frames info
        startId_ = ParameterServer::get()["main"]["startDataRange"];
        endId_ = ParameterServer::get()["main"]["stopDataRange"];
    }

    void loadNewPoseFile() {
        std::string poseFilePath = ParameterServer::get()["pose"]["poseFilePath"];
        std::ifstream newPoseFile(poseFilePath);

        if (newPoseFile.is_open()) {
            std::string line;
            int i = 0;
            while (getline(newPoseFile, line) && i <= endId_) {
                if (i >= startId_) {
                    Eigen::Isometry3d pose;
                    std::stringstream ss(line);
                    std::string word;
                    int j = 0;
                    while (getline(ss, word, ' ')) {
                        pose(j / 4, j % 4) = stod(word);
                        j++;
                    }
                    pose(3, 3) = 1;
                    poses_.push_back(pose);

//                    std:: cout <<"line number: " << i << std::endl;
//                    std:: cout << poses_[i - startId_].matrix() << std::endl;
//                    std:: cout << "+******************" << std::endl;
                }
                i++;
            }
        }
    }

    /*** this function is only used for pose align and create new pose file: odom_os_opt_final.txt ***/
    void load() {
        std::string poseFilePath = ParameterServer::get()["pose"]["poseFilePath"];
        std::ifstream oldPoseFile(poseFilePath);
        if (oldPoseFile.is_open()) {
            std::string line;
            int i = 0;
            while (getline(oldPoseFile, line) && i <= endId_) {
                while (!isFrameValid(i)) {
                    if (i >= startId_) {
                        poses_.push_back(poses_.back());
                    }
                    i++;
                }
                if (i >= startId_) {
                    Eigen::Isometry3d pose;
                    std::stringstream ss(line);
                    std::string word;
                    int j = 0;
                    while (getline(ss, word, ' ')) {
                        pose(j / 4, j % 4) = stod(word);
                        j++;
                    }
                    pose(3, 3) = 1;
                    poses_.push_back(pose);
                }
                i++;
            }
        }

        const std::string path = "/home/rwang/rwang_ws/src/mapping_semantic_ground_estimator_tool/output/debug/";

        std::ofstream calib_pose_file(path + "calib_pose_file.txt");
        if (calib_pose_file.is_open()) {
            for (int i = startId_; i <= endId_; ++i) {
                calib_pose_file << poses_[i].matrix() << std::endl;
            }
        }



        std::ofstream test_file(path + "test_file.txt");
        if (test_file.is_open()) {
            for (int i = startId_; i <= endId_; ++i) {
                Eigen::MatrixXd M1(4,4);
                M1 = poses_[i].matrix();
                M1.transposeInPlace();
                Eigen::VectorXd poseVector(Eigen::Map<Eigen::VectorXd>(M1.data(), M1.cols()*M1.rows()));
                test_file << poseVector.transpose() << std::endl;
            }

        }

    }

    bool isFrameValid(int i) {
        auto frameData = mainDataStorage_[i];
        if (!frameData) {
            std::cerr << "\nFrameData " << i << " not found" << std::endl;
            return false;
        }
        if (!frameData->pose()) {
            std::cerr << "Did not find pose. ~> Skipping frame " << i << std::endl;
            return false;
        }
        auto pc = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>((*frameData)[fbds::FrameSource::VelodyneC]->asPointcloud());
        if (pc->points.empty()) {
            std::cerr << "Did not find point cloud. ~> Skipping frame" << i << std::endl;
            return false;
        }
        return true;
    }

    Eigen::Isometry3d getPose(int frameId) {
        return poses_[frameId - startId_];
    };


private:
    std::string configPath_;
    fbds::DataStorage mainDataStorage_;
    int startId_;
    int endId_;
    std::vector<Eigen::Isometry3d> poses_;
};
}



#endif // SRC_LOAD_POSE_H