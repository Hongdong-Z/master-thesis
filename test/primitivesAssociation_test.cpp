//
// Created by zhao on 04.08.22.
//
#include <folder_based_data_storage/folder_based_data_storage.hpp>
#include "nlohmann/json.hpp"
#include "localization_lidar/load_datastorage.hpp"
#include "localization_lidar/parameter_server.hpp"
#include "primitives_extraction/primitivesExtraction.h"
#include "primitives_association/primitives_association.h"
using json = nlohmann::json;
using Config = localization_lidar::ParameterServer;
int main() {

    auto configPath = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::cout << "Loading config file from " << configPath << std::endl;
    Config::initialize(configPath);

    std::filesystem::path dataRootPath(Config::get()["main"]["dataRoot"]);
    auto mainDataStorage = loadDataStorage(dataRootPath);
    uint64_t dataRange[2] = {Config::get()["main"]["startDataRange"], Config::get()["main"]["stopDataRange"]};


    std::ofstream poseFile("/home/zhao/zhd_ws/src/localization_lidar/pose_primitives/pose_odo_pri.txt");


    poseFile << 0.945379 <<-0.325877 <<-0.00790777 <<445.531
             <<0.325973 <<0.94512 <<0.0221132 <<87.0484
             <<0.000267601<< -0.0234831<< 0.999724<< -0.912769<<std::endl;





    Eigen::Quaterniond q_last_curr(0.9861825,-0.0115588,-0.0020725,0.1652458);
    Eigen::Vector3d t_last_curr(445.531,87.0484 ,-0.912769);

    Eigen::Quaterniond qCurr(1,0,0,0);
    Eigen::Vector3d tCurr(0,0,0);

    std::optional<fbds::FrameData> temFrame = mainDataStorage[1];
    std::vector<primitivesExtraction::CloudColor::Ptr> poleCandidates1, poleCandidates2;
    std::vector<primitivesExtraction::CloudColor::Ptr> facadesCandidates1, facadesCandidates2;
    primitivesExtraction::CloudColor::Ptr poleColoredSum(new primitivesExtraction::CloudColor);
    primitivesExtraction::CloudColor::Ptr facadesColoredSum(new primitivesExtraction::CloudColor);





//    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//    pcl::ModelCoefficients cylinder_coeff;
//    cylinder_coeff.values.resize(7);

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
        // point cloud of current frame

        std::shared_ptr<pcl::PointCloud<PointType>> lastPC =
            std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatumLast->asPointcloud<PointType>());
        std::shared_ptr<pcl::PointCloud<PointType>> currPC =
            std::make_shared<pcl::PointCloud<PointType>>(velodyneFLDatumCurrent->asPointcloud<PointType>());

        std::cout << "*************************start processing*************************" << std::endl;

        primitivesFa::PrimitivesAssociation PA(lastPC->makeShared(), currPC->makeShared(),
                                               qCurr, tCurr,
                                               q_last_curr, t_last_curr);
        PA.runAS();
        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
        PA.getAssociation(cloud);
        pcl::io::savePCDFile("/home/zhao/zhd_ws/src/localization_lidar/primitiveLocalization/output/cloud/primitivesAssociation/test/"
                             + std::to_string(i) + "_PA.pcd", *cloud);

        qCurr = PA.publishQuaterniond();
        tCurr = PA.publishTranslation();
        q_last_curr = PA.quaterniodGuess();
        t_last_curr = PA.translationGuess();
        Eigen::Isometry3d T = Eigen::Isometry3d ::Identity();
        T.rotate(qCurr.matrix());
        T.pretranslate(tCurr);
        std::cout << "T: \n " << T.matrix() << std::endl;
        poseFile << T.matrix()(0, 0) << " " << T.matrix()(0, 1) << " " << T.matrix()(0, 2) << " " << T.matrix()(0, 3)
                 << " " << T.matrix()(1, 0) << " " << T.matrix()(1, 1) << " " << T.matrix()(1, 2) << " " << T.matrix()(1, 3) << " "
                 << T.matrix()(2, 0) << " " << T.matrix()(2, 1) << " " << T.matrix()(2, 2) << " " << T.matrix()(2, 3) << std::endl;

        std::cerr << "Frame: " << std::to_string(i) << " Finish " << std::endl;

        std::cout << "********************************************************************" << std::endl;



    }

    std::cout << "*************************end processing*************************" << std::endl;

    return 0;
}