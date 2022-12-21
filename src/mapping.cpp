//
// Created by zhao on 09.06.22.
//

#include "map_matching/mapping.h"

namespace mapping{

Mapping::Mapping(const int lengthMap_) {
    lengthMap = lengthMap_;
    initialValue();
}

void Mapping::initialValue() {
    mapGround.reset(new pcl::PointCloud<PointType>());
    mapCurb.reset(new pcl::PointCloud<PointType>());
    mapSurface.reset(new pcl::PointCloud<PointType>());
    mapEdge.reset(new pcl::PointCloud<PointType>());

    mapGroundDS.reset(new pcl::PointCloud<PointType>());
    mapCurbDS.reset(new pcl::PointCloud<PointType>());
    mapSurfaceDS.reset(new pcl::PointCloud<PointType>());
    mapEdgeDS.reset(new pcl::PointCloud<PointType>());

    tempGround.reset(new pcl::PointCloud<PointType>());
    tempCurb.reset(new pcl::PointCloud<PointType>());
    tempSurface.reset(new pcl::PointCloud<PointType>());
    tempEdge.reset(new pcl::PointCloud<PointType>());

    tempGroundTF.reset(new pcl::PointCloud<PointType>());
    tempCurbTF.reset(new pcl::PointCloud<PointType>());
    tempSurfaceTF.reset(new pcl::PointCloud<PointType>());
    tempEdgeTF.reset(new pcl::PointCloud<PointType>());

    currentPointCloud.reset(new pcl::PointCloud<PointType>());

    downSizeFilterCurb.setLeafSize(0.4, 0.4, 0.4);
    downSizeFilterSurface.setLeafSize(0.8, 0.8, 0.8);
    downSizeFilterGround.setLeafSize(0.8, 0.8, 0.8);
    downSizeFilterEdge.setLeafSize(0.4, 0.4, 0.4);

    distanceCount = 0;

}

void Mapping::resetPara() {

    mapGroundDS->clear();
    mapCurbDS->clear();
    mapSurfaceDS->clear();
    mapEdgeDS->clear();

    tempGround->clear();
    tempCurb->clear();
    tempSurface->clear();
    tempEdge->clear();

    tempGroundTF->clear();
    tempCurbTF->clear();
    tempSurfaceTF->clear();
    tempEdgeTF->clear();

    currentPointCloud->clear();
}

void Mapping::run() {
    auto configPath = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::cout << "Loading config file from " << configPath << std::endl;
    localization_lidar::ParameterServer::initialize(configPath);

    std::filesystem::path dataRootPath(localization_lidar::ParameterServer::get()["main"]["dataRoot"]);
    std::filesystem::path outputPath(localization_lidar::ParameterServer::get()["main"]["outputFolderMap"]);
    std::filesystem::path groundMapPath = outputPath / "groundMap";
    std::filesystem::path curbMapPath = outputPath / "curbMap";
    std::filesystem::path surfaceMapPath = outputPath / "surfaceMap";
    std::filesystem::path edgeMapPath = outputPath / "edgeMap";
    std::filesystem::create_directory(outputPath);
    std::filesystem::create_directory(groundMapPath);
    std::filesystem::create_directory(curbMapPath);
    std::filesystem::create_directory(surfaceMapPath);
    std::filesystem::create_directory(edgeMapPath);


    auto mainDataStorage = loadDataStorage(dataRootPath);


    uint64_t dataRange[2] = {localization_lidar::ParameterServer::get()["main"]["startDataRange"], localization_lidar::ParameterServer::get()["main"]["stopDataRange"]};


    std::vector<size_t> validDataIndices; // Only frames with both point cloud is used

    // load new pose file
    std::string poseConfig = "/home/zhao/zhd_ws/src/localization_lidar/cfg/config.json";
    std::shared_ptr<localization_lidar::PoseLoader> poseLoader = std::make_shared<localization_lidar::PoseLoader>(poseConfig);
    poseLoader->loadNewPoseFile();


    size_t tempI = 0;
    double distanceCountName = 0;
    double distanceCountNameStart = 0;
    for (size_t i = dataRange[0]; i <= dataRange[1]; ++i) {

        // start with second frame
        std::optional<fbds::FrameData> frameData = mainDataStorage[i];
        if(!frameData) { std::cerr << "\nFrameData or lastFrameData " << i << " not found " << std::endl; continue; }

        std::cout << "\nProcessing Frame " << i << std::endl;

        if(!frameData->pose()) { std::cerr << "Did not find pose. ~> Skipping frame. " << std::endl; continue; }

        validDataIndices.emplace_back(i);
        // set start frame
        auto pose_w_first = poseLoader->getPose(tempI);
        // true first frame
        auto pose_w_first_fix = poseLoader->getPose(0);

        auto pose_w_current = poseLoader->getPose(i);

        Eigen::Vector3d firstPosition = pose_w_first.matrix().block<3, 1>(0, 3);
        Eigen::Vector3d trueFirstPosition = pose_w_first_fix.matrix().block<3, 1>(0, 3);
        Eigen::Vector3d currentPosition = pose_w_current.matrix().block<3, 1>(0, 3);

//        std::cout << "first position:  second position: " << firstPosition << currentPosition << std::endl;

        auto velodyneFLDatum = (*frameData)[fbds::FrameSource::VelodyneFL];

//        std::cout << velodyneFLDatum.value() << std::endl;
        *currentPointCloud = velodyneFLDatum->asPointcloud<PointType>();

        distanceCount = (currentPosition - firstPosition).norm();
        distanceCountName = (currentPosition - trueFirstPosition).norm();
//        std::cout << distanceCountName << std::endl;

        llo::ImageProjection IP;
        IP.run(*currentPointCloud);
        // local cloud to save map needed to be reset


        tempGround = IP.getGroundFeature();
        tempCurb = IP.getCurbFeature();
        tempSurface = IP.getSurfaceFeature();
        tempEdge = IP.getEdgeFeature();

        // Transform between first frame and current frame

        Eigen::Transform<double, 3, Eigen::Affine> transformFrame = pose_w_current;

        pcl::transformPointCloud(*tempGround, *tempGroundTF, transformFrame);
        pcl::transformPointCloud(*tempCurb, *tempCurbTF, transformFrame);
        pcl::transformPointCloud(*tempSurface, *tempSurfaceTF, transformFrame);
        pcl::transformPointCloud(*tempEdge, *tempEdgeTF, transformFrame);

        // downsize
        downSizeFilterGround.setInputCloud(tempGroundTF);
        downSizeFilterGround.filter(*mapGroundDS);

        downSizeFilterCurb.setInputCloud(tempCurbTF);
        downSizeFilterCurb.filter(*mapCurbDS);

        downSizeFilterSurface.setInputCloud(tempSurfaceTF);
        downSizeFilterSurface.filter(*mapSurfaceDS);

        downSizeFilterEdge.setInputCloud(tempEdgeTF);
        downSizeFilterEdge.filter(*mapEdgeDS);

        *mapGround += *mapGroundDS;
        *mapCurb += *mapCurbDS;
        *mapSurface += *mapSurfaceDS;
        *mapEdge += *mapEdgeDS;

        if (i % lengthMap == 0) {
            // save break frame

            pcl::io::savePCDFileASCII(groundMapPath / ("ground_" + std::to_string(tempI) + "_"
                                                    + std::to_string(i) + ".pcd"), *mapGround);
            pcl::io::savePCDFileASCII(curbMapPath / ("curb_" + std::to_string(tempI) + "_"
                                                    + std::to_string(i) + ".pcd"), *mapCurb);
            pcl::io::savePCDFileASCII(surfaceMapPath / ("surface_" + std::to_string(tempI) + "_"
                                                    + std::to_string(i) + ".pcd"), *mapSurface);
            pcl::io::savePCDFileASCII(edgeMapPath / ("edge_" + std::to_string(tempI) + "_"
                                                    + std::to_string(i) + ".pcd"), *mapEdge);
            initialValue();
            std::cerr << "save sub map between frames: " << tempI << " and " << i << std::endl;
            tempI = i;
            distanceCountNameStart = distanceCountName;

            continue;
        }
        resetPara();

    }


}





} // end of namespace mapping
