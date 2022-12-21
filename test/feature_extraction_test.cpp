#include "feature_extraction/feature_extraction.h"

int main(){

    std::filesystem::path inputPath = "/home/zhao/zhd_ws/src/localization_lidar/input/3700.pcd";
    std::filesystem::path outPath = "/home/zhao/zhd_ws/src/localization_lidar/output/feature_extraction/3700";
    std::shared_ptr<pcl::PointCloud<PointType>> inputCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> groundCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> curbCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> surfaceCloud = std::make_shared<pcl::PointCloud<PointType>>();
    std::shared_ptr<pcl::PointCloud<PointType>> edgeCloud = std::make_shared<pcl::PointCloud<PointType>>();
    if (pcl::io::loadPCDFile<PointType>(inputPath, *inputCloud) == -1) {
        PCL_ERROR("Couldn't read file from path : ");
        return (-1);
    }
    std::cout << "Loaded " << inputCloud->size()
              << " data points from 3600.pcd with the following fields: " << std::endl;

    llo::FeaturesExtraction fe(inputCloud);
    *groundCloud = fe.getGround();
    *curbCloud = fe.getCurb();
    *surfaceCloud = fe.getSurface();
    *edgeCloud = fe.getEdge();
//    fe.getEdgeFeatureSize();
//    fe.getSurfaceFeatureSize();

    pcl::io::savePCDFileASCII(outPath / "groundCloud.pcd", *groundCloud);
    pcl::io::savePCDFileASCII(outPath / "curbCloud.pcd", *curbCloud);
    pcl::io::savePCDFileASCII(outPath / "surfaceCloud.pcd", *surfaceCloud);
    pcl::io::savePCDFileASCII(outPath / "edgeCloud.pcd", *edgeCloud);
    std::cout << "-------Finish Mapping-------" << std::endl;
    return 0;
}

